#!/usr/bin/env python3
"""Original Body decoder-layer contracts; synthetic parameters, not learned pose parity."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys
import types

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from safetensors.numpy import save_file
from capture_camera_encoder import HASHES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--small-regression',action='store_true')
    args=parser.parse_args()
    root=args.upstream/'sam_3d_body/models/modules'
    for name,digest in HASHES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest: raise ValueError(f'source changed: {name}')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.modules']:
        package=types.ModuleType(name);package.__path__=[str(args.upstream/Path(*name.split('.')))];sys.modules[name]=package
    Layer=importlib.import_module('sam_3d_body.models.modules.transformer').TransformerDecoderLayer
    torch.set_num_threads(1);torch.manual_seed(8601);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(8601)
    # repeat PE, skip first self PE, two-way, token PE, context PE, token mask.
    options=[(0,0,0,0,0,0),(1,1,1,1,1,1),(1,0,1,1,1,1),
             (1,0,0,1,0,1),(0,0,1,1,1,1),(1,0,1,0,0,0)]
    shapes=[(2,3,4,4,6,2,3,8)]*len(options)
    if not args.small_regression:
        shapes=[(2,7,11,16,24,2,8,32)]*len(options)
        shapes.append((1,143,1024,1024,1280,8,64,4096));options.append((1,0,1,1,1,0))
    args.output.mkdir(parents=True,exist_ok=True)
    cases,tensors,rules=[],{},[];lines=['S3D_DECODER_REGRESSION_V1',str(len(shapes))]
    for index,((b,n,m,d,c,h,dh,f),flags) in enumerate(zip(shapes,options)):
        repeat,skip,twoway,has_xp,has_cp,has_mask=flags
        net=Layer(d,c,h,dh,f,repeat_pe=bool(repeat),skip_first_pe=bool(skip),enable_twoway=bool(twoway)).float().to(args.device).eval()
        def random(shape,scale=1): return torch.from_numpy(rng.normal(scale=scale,size=shape).astype(np.float32)).to(args.device)
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            for name,p in net.named_parameters():
                if p.ndim==2: values=random(tuple(p.shape),1/np.sqrt(p.shape[1]))
                elif name.endswith('weight'): values=torch.from_numpy(rng.uniform(.8,1.2,size=tuple(p.shape)).astype(np.float32)).to(args.device)
                else: values=random(tuple(p.shape),.1)
                p.copy_(values)
            x=random((b,n,d));image=random((b,m,c))
            xp=random((b if index%2 else 1,n,d)) if has_xp else None
            cp=random((1,m,c)) if has_cp else None
            mask=None
            if has_mask:
                mask=torch.ones((b,n),device=args.device);mask[:,1::2]=0
                mask[-1]=0 # includes SDPA's all-masked reverse row contract
            baseline=tuple(v.clone() for v in net(x,image,xp,cp,mask));taps={};hooks=[]
            def capture(name,v): taps[name]=v.detach().clone()
            def pre(module,name,kw=False):
                if kw: raise AssertionError('not used')
                hooks.append(module.register_forward_pre_hook(lambda _m,a:capture(name,a[0])))
            def post(module,name): hooks.append(module.register_forward_hook(lambda _m,_a,v:capture(name,v)))
            if repeat and cp is not None: post(net.ln_pe_1,'00.token_pe');post(net.ln_pe_2,'01.context_pe')
            post(net.ln1,'02.ln1')
            pre(net.ln2_1,'20.self_residual');post(net.ln2_1,'21.ln2_1');post(net.ln2_2,'22.ln2_2')
            pre(net.ln3,'40.cross_residual');post(net.ln3,'41.ln3')
            post(net.ffn.layers[0][0],'42.ffn_linear1');post(net.ffn.layers[0][1],'43.ffn_gelu')
            post(net.ffn.layers[1],'44.ffn_linear2');post(net.ffn,'45.ffn_residual')
            if twoway:post(net.ln4_1,'50.ln4_1');post(net.ln4_2,'51.ln4_2')
            def attention(module,stage):
                def inputs(_m,_a,kw):
                    for i,key in enumerate(['q','k','v']):capture(f'{stage}.0{i}.{key}_input',kw[key])
                    # Diagnostic expansion only. Original module still executes
                    # its own projections and SDPA, with no monkey-patching.
                hooks.append(module.register_forward_pre_hook(inputs,with_kwargs=True))
                for i,key in enumerate(['q','k','v']):
                    hooks.append(getattr(module,key+'_proj').register_forward_hook(
                        lambda _m,_a,v,key=key,i=i:capture(f'{stage}.0{i+3}.{key}',module._separate_heads(v))))
                pre(module.proj,stage+'.08.attended');post(module,stage+'.09.output')
            attention(net.self_attn,'10.self');attention(net.cross_attn,'30.cross')
            if twoway:attention(net.cross_attn_2,'60.reverse')
            observed=net(x,image,xp,cp,mask)
            if not all(torch.equal(a,v) for a,v in zip(baseline,observed)):raise ValueError('instrumentation changed decoder output')
            for hook in hooks:hook.remove()
            capture('90.tokens',observed[0]);capture('91.context',observed[1])
            for stage in ['10.self','30.cross']+(['60.reverse'] if twoway else []):
                q,k=taps[stage+'.03.q'],taps[stage+'.04.k']
                logits=(q@k.transpose(-1,-2))*(1/np.sqrt(dh));capture(stage+'.06.logits',logits)
                allowed=None
                if mask is not None and stage=='10.self':
                    allowed=mask[:,:,None]@mask[:,None,:];allowed.diagonal(dim1=1,dim2=2).fill_(1);allowed=allowed[:,None]>0
                elif mask is not None and stage=='60.reverse':allowed=mask[:,None,None,:]>0
                if allowed is not None:logits=logits.masked_fill(~allowed,float('-inf'))
                capture(stage+'.07.probs',torch.nan_to_num(logits.softmax(-1),nan=0))
        inputs={'tokens':x,'context':image}
        if xp is not None:inputs['token_pe']=xp
        if cp is not None:inputs['context_pe']=cp
        if mask is not None:inputs['mask']=mask
        state=net.state_dict();order=sorted(taps);prefix=f'case.{index:04d}';filename=prefix+'.input'
        # Generic named tensors allow the native schema to reject unexpected
        # names and exact sizes without requiring both languages' map ordering.
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DDEC01'+struct.pack('<11I',b,n,m,d,c,h,dh,f,repeat,skip,twoway))
            all_inputs=inputs|dict(state);stream.write(struct.pack('<I',len(all_inputs)))
            for name,v in sorted(all_inputs.items()):
                encoded=name.encode('ascii');a=v.cpu().numpy().astype('<f4')
                stream.write(struct.pack('<I',len(encoded))+encoded+struct.pack('<Q',a.size)+a.tobytes())
        if args.small_regression:
            lines.append(' '.join(map(str,(b,n,m,d,c,h,dh,f,repeat,skip,twoway))))
            for group in [inputs,dict(state),taps]:
                lines.append(str(len(group)))
                for name,v in sorted(group.items()):
                    a=v.cpu().numpy().reshape(-1);lines.append(f'{name} {a.size}')
                    for start in range(0,len(a),8):lines.append(' '.join(format(float(v),'.9g') for v in a[start:start+8]))
        cases.append({'prefix':prefix,'input':filename,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()}})
        for name in order:
            key=prefix+'.'+name;tensors[key]=taps[name].cpu().numpy().copy()
            rules.append({'name':key,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {prefix} shape={(b,n,m,d,c,h,dh,f)} flags={flags} taps={len(taps)}',flush=True)
    if args.small_regression:(args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='original TransformerDecoderLayer synthetic F32/GELU/eps1e-6/no-LayerScale component; not learned or full decoder feedback parity'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'b5c765a0d89d789985e186d396315e7590887b94','source_hashes':HASHES,
        'capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':args.device,
        'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,'torch':torch.__version__,
        'numpy':np.__version__,'threads':1,'sdpa_backend':'MATH','tf32_matmul':False,'tf32_cudnn':False,
        'unobserved_vs_observed':'exact_equal_all_cases','auxiliary_taps':['*.06.logits','*.07.probs'],
        'cases':cases,'artifacts':{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json':manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
