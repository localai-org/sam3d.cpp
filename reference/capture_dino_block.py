#!/usr/bin/env python3
"""Pinned original DINOv3 eval block/RoPE with synthetic parameters, NOT model parity.

Hooks observe actual original module inputs/outputs. The apply_rope wrapper calls
the unchanged bound method. Exact equality with uninstrumented forward is
required. QKV views, padded prefix sin/cos, logits and probabilities are marked
auxiliary diagnostics, not internal taps of PyTorch fused SDPA.
"""
import argparse
from functools import partial
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys
import types

import numpy as np
import torch
from safetensors.numpy import save_file

HASHES = {
    "layers/block.py":"809a3615ec93019042ab440f31c21d5242186d9e503c67ac0f42a324ceae1950",
    "layers/attention.py":"dfc21def6ba17d00cba18ea23a8726e7f1dd5a0dc8e43510852c999a173d37a8",
    "layers/ffn_layers.py":"fa2caadb52ff43a7e6888966a00ee07842fed9fd05c68305962cc1577d221fbe",
    "layers/layer_scale.py":"f9a10ee763a62136f9e814bfe2f2209c2cb501dfdc5573f858ab01ba9a69c23f",
    "layers/rope_position_encoding.py":"af8ea35acb76d18ad4d53b844d5057bf01472a848375c51d70583a87f044efd1",
    "utils/__init__.py":"83e6ec1f0cea862f18ae111560e2eadf2fff1873e3fdd7d358dee8d32d0ade8d",
    "utils/utils.py":"d6fe7fe621b2ae3db89e3af588fbee0c78afe2130f4548dff87c94cbd734fe40",
    "utils/dtype.py":"0c53f5cc155ea7e9899bc611493193fef9b985f23cb7d990401e22c363ac4419",
}
PARAMETERS = ["norm1.weight","norm1.bias","attn.qkv.weight","attn.qkv.bias",
              "attn.qkv.bias_mask","attn.proj.weight","attn.proj.bias","ls1.gamma",
              "norm2.weight","norm2.bias","mlp.w1.weight","mlp.w1.bias",
              "mlp.w2.weight","mlp.w2.bias","mlp.w3.weight","mlp.w3.bias","ls2.gamma","periods"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--small-regression',action='store_true',help='capture one compact normal-test fixture')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    for path,digest in HASHES.items():
        if hashlib.sha256((args.upstream/'dinov3'/path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'upstream hash mismatch: {path}')
    # Avoid unrelated root imports/optional FP8 packages, no functions replaced.
    for name in ['dinov3','dinov3.layers']:
        namespace=types.ModuleType(name)
        namespace.__path__=[str(args.upstream/Path(*name.split('.')))]
        sys.modules[name]=namespace
    Block=importlib.import_module('dinov3.layers.block').SelfAttentionBlock
    SwiGLU=importlib.import_module('dinov3.layers.ffn_layers').SwiGLUFFN
    Rope=importlib.import_module('dinov3.layers.rope_position_encoding').RopePositionEmbedding
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.small_regression and args.device!='cpu':
        raise ValueError('normal-test fixture provenance uses CPU only')
    rng=np.random.default_rng(8401)
    args.output.mkdir(parents=True,exist_ok=True)
    cases,tensors,rules=[],{},[]
    shapes=[(1,1,1,8,2,0),(2,2,3,32,4,5),(1,3,5,1280,20,5),(1,32,32,1280,20,5)]
    if args.small_regression: shapes=[(2,2,3,8,2,5)]
    for case_index,(b,h,w,d,heads,prefix) in enumerate(shapes):
        case=f'case.{case_index:04d}'
        net=Block(d,heads,ffn_ratio=6,qkv_bias=True,proj_bias=True,ffn_bias=True,
                  init_values=1e-5,norm_layer=partial(torch.nn.LayerNorm,eps=1e-5),
                  ffn_layer=SwiGLU,mask_k_bias=True).float().to(args.device).eval()
        rope_net=Rope(d,num_heads=heads,base=100,normalize_coords='separate',
                      rescale_coords=2,dtype=torch.float32,device=args.device).eval()
        with torch.no_grad():
            for name,p in net.named_parameters():
                if p.ndim==2:
                    value=rng.normal(scale=1/np.sqrt(p.shape[1]),size=tuple(p.shape))
                elif name.endswith('weight'): value=rng.uniform(.8,1.2,size=tuple(p.shape))
                elif name.endswith('gamma'): value=rng.uniform(.2,.5,size=tuple(p.shape))
                else: value=rng.normal(scale=.1,size=tuple(p.shape))
                p.copy_(torch.from_numpy(value.astype(np.float32)))
            net.attn.qkv.bias_mask.fill_(1)
            net.attn.qkv.bias_mask[d:2*d].fill_(0)
            x=torch.from_numpy(rng.normal(size=(b,h*w+prefix,d)).astype(np.float32)).to(args.device)
            rope=rope_net(H=h,W=w)
            baseline=net(x,rope).clone()
            taps={}
            def capture(name,value): taps[name]=value.detach().clone()
            capture('00.rope_sin',torch.nn.functional.pad(rope[0],(0,0,prefix,0),value=0))
            capture('01.rope_cos',torch.nn.functional.pad(rope[1],(0,0,prefix,0),value=1))
            hooks=[]
            for attr,key in [('norm1','02.norm1'),('attn.qkv','03.qkv'),('attn.proj','12.attn_proj'),
                             ('ls1','13.ls1'),('norm2','15.norm2'),('mlp.w1','16.w1'),
                             ('mlp.w2','17.w2'),('mlp.w3','19.w3'),('ls2','20.ls2')]:
                layer=net.get_submodule(attr)
                hooks.append(layer.register_forward_hook(lambda _m,_a,v,key=key:capture(key,v)))
            for attr,key in [('attn.proj','11.attention'),('norm2','14.residual1'),('mlp.w3','18.hidden')]:
                hooks.append(net.get_submodule(attr).register_forward_pre_hook(lambda _m,a,key=key:capture(key,a[0])))
            original_apply=net.attn.apply_rope
            def observed_apply(q,k,rope):
                capture('04.q',q); capture('05.k',k)
                out=original_apply(q,k,rope)
                capture('07.q_rope',out[0]); capture('08.k_rope',out[1])
                return out
            net.attn.apply_rope=observed_apply
            observed=net(x,rope)
            if not torch.equal(baseline,observed): raise ValueError('instrumentation changed block output')
            capture('21.output',observed)
            net.attn.apply_rope=original_apply
            for hook in hooks: hook.remove()
            capture('06.v',taps['03.qkv'].reshape(b,h*w+prefix,3,heads,d//heads)[:,:,2].transpose(1,2))
            logits=(taps['07.q_rope']@taps['08.k_rope'].transpose(-1,-2))*((d//heads)**-.5)
            capture('09.logits',logits)
            capture('10.probs',logits.softmax(-1))
        state=dict(net.state_dict()); state['periods']=rope_net.periods
        filename=case+'.input'
        hidden=net.mlp.w1.out_features
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DBLK01'+struct.pack('<7I',b,h,w,d,heads,prefix,hidden))
            stream.write(x.cpu().numpy().astype('<f4').tobytes())
            for name in PARAMETERS: stream.write(state[name].cpu().numpy().astype('<f4').tobytes())
        order=sorted(taps)
        if args.small_regression:
            lines=['S3D_DINO_REGRESSION_V1',f'{b} {h} {w} {d} {heads} {prefix} {hidden}']
            def append_values(name,value):
                values=value.detach().numpy().reshape(-1)
                lines.append(f'{name} {len(values)}')
                for start in range(0,len(values),8):
                    lines.append(' '.join(format(float(v),'.9g') for v in values[start:start+8]))
            append_values('input',x)
            for name in PARAMETERS: append_values(name,state[name])
            for key in order: append_values(key,taps[key])
            (args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
        cases.append({'prefix':case,'input':filename,'order':order,
                      'shapes':{key:list(taps[key].shape) for key in order}})
        for key in order:
            name=case+'.'+key
            tensors[name]=taps[key].cpu().numpy().copy()
            rules.append({'name':name,'mode':'float','max_abs':1e-4,
                          'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {case}: {b}x{h*w+prefix}x{d}',flush=True)
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='original DINOv3 eval block/RoPE synthetic-weight contract; auxiliary attention diagnostics; NOT trained-model parity'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'6876159a11b4df116f30f667f8c9888617df0751',
              'source_hashes':HASHES,'dtype':'float32','torch':torch.__version__,'numpy':np.__version__,
              'device':args.device,'threads':1,'unobserved_vs_observed':'exact_equal_all_cases',
              'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,
              'tf32_matmul':False,'tf32_cudnn':False,
              'auxiliary_taps':['00.rope_sin','01.rope_cos','06.v','09.logits','10.probs'],
              'cases':cases,'artifacts':{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json':
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
            manifest['artifacts'][path.name]=digest.hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
