#!/usr/bin/env python3
"""Original Body wrapper + DINO backbone synthetic contracts, not trained parity."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import types

import numpy as np
import torch
from safetensors.numpy import save_file
from capture_dino_schema import HASHES, load_factory
from capture_dino_block import PARAMETERS

BODY_PATH='sam_3d_body/models/backbones/dinov3.py'
BODY_HASH='53ae01ccdffa647f16b0fefe366f947162874a9e593d510864d52ffcd259e0e8'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--body-upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--full-shape',action='store_true',help='full 32-block H+ at 512x512; synthetic weights')
    parser.add_argument('--small-regression',action='store_true',help='compact text fixture for normal CTest')
    args=parser.parse_args()
    if args.small_regression and (args.full_shape or args.device!='cpu'):
        raise ValueError('small regression requires CPU and excludes full shape')
    factory=load_factory(args.upstream)
    path=args.body_upstream/BODY_PATH
    if digest(path)!=BODY_HASH: raise ValueError('original Body wrapper hash mismatch')
    spec=importlib.util.spec_from_file_location('original_body_backbone',path)
    body=importlib.util.module_from_spec(spec); spec.loader.exec_module(body)
    Dino=importlib.import_module('dinov3.models.vision_transformer').DinoVisionTransformer
    torch.set_num_threads(1); torch.manual_seed(8402)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(8402)
    shapes=[(2,32,48,16,32,4,3,4),(1,48,32,16,32,4,2,0)]
    if args.full_shape: shapes=[(1,512,512,16,1280,20,32,4)]
    if args.small_regression: shapes=[(2,4,6,2,8,2,3,4)]
    args.output.mkdir(parents=True,exist_ok=True)
    tensors,cases,rules={},[],[]
    for index,(batch,h,w,patch,dim,heads,depth,storage) in enumerate(shapes):
        case=f'case.{index:04d}'
        if args.full_shape:
            # Original factory and initializer; no pretrained/download branch.
            net=factory(pretrained=False).float().to(args.device).eval()
        else:
            net=Dino(img_size=h,patch_size=patch,embed_dim=dim,depth=depth,num_heads=heads,
                ffn_ratio=6,qkv_bias=True,layerscale_init=1e-5,norm_layer='layernormbf16',
                ffn_layer='swiglu',ffn_bias=True,proj_bias=True,n_storage_tokens=storage,
                mask_k_bias=True,pos_embed_rope_dtype='fp32',pos_embed_rope_rescale_coords=2)
            net.init_weights(); net=net.float().to(args.device).eval()
        hidden=net.blocks[0].mlp.w1.out_features
        with torch.no_grad():
            # Stress residual propagation, not almost-zero initialization. Each
            # tensor is stored verbatim for native; no cross-framework RNG claim.
            for name,p in net.named_parameters():
                if p.ndim>=2 and name.endswith('weight'):
                    value=rng.normal(scale=1/np.sqrt(np.prod(p.shape[1:])),size=tuple(p.shape))
                elif name.endswith('weight'): value=rng.uniform(.8,1.2,size=tuple(p.shape))
                elif name.endswith('gamma'): value=rng.uniform(.2,.5,size=tuple(p.shape))
                else: value=rng.normal(scale=.1,size=tuple(p.shape))
                p.copy_(torch.from_numpy(value.astype(np.float32))); del value
            image=torch.from_numpy(rng.normal(size=(batch,3,h,w)).astype(np.float32)).to(args.device)
            wrapper=types.SimpleNamespace(encoder=net)
            baseline=body.Dinov3Backbone.forward(wrapper,image).clone()
            taps={}
            def capture(name,value): taps[name]=value.detach().clone()
            hooks=[net.patch_embed.register_forward_hook(lambda _m,_a,v:capture('00.patch',v.flatten(1,2))),
                   net.norm.register_forward_hook(lambda _m,_a,v:capture('03.norm',v))]
            for block_index,block in enumerate(net.blocks):
                key=f'02.block.{block_index:02d}'
                hooks.append(block.register_forward_hook(lambda _m,_a,v,key=key:capture(key,v)))
            original=net.prepare_tokens_with_masks
            def observed_tokens(*a,**kw):
                result=original(*a,**kw); capture('01.tokens',result[0]); return result
            net.prepare_tokens_with_masks=observed_tokens
            observed=body.Dinov3Backbone.forward(wrapper,image)
            if not torch.equal(baseline,observed): raise ValueError('backbone instrumentation changed output')
            capture('04.features',observed)
            net.prepare_tokens_with_masks=original
            for hook in hooks: hook.remove()
        state=net.state_dict()
        parameter_order=['patch_embed.proj.weight','patch_embed.proj.bias','cls_token','mask_token']
        if storage: parameter_order.append('storage_tokens')
        parameter_order.append('rope_embed.periods')
        for block_index in range(depth):
            parameter_order.extend(f'blocks.{block_index}.{name}' for name in PARAMETERS if name!='periods')
        parameter_order+=['norm.weight','norm.bias']
        if set(parameter_order)!=set(state): raise ValueError('unhandled original state tensor')
        filename=case+'.input'
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DBBN01'+struct.pack('<9I',batch,h,w,patch,dim,heads,hidden,depth,storage))
            stream.write(image.cpu().numpy().astype('<f4').tobytes())
            for name in parameter_order: stream.write(state[name].cpu().numpy().astype('<f4').tobytes())
        order=sorted(taps)
        if args.small_regression:
            lines=['S3D_BACKBONE_REGRESSION_V1',f'{batch} {h} {w} {patch} {dim} {heads} {hidden} {depth} {storage}']
            def append(name,value):
                values=value.detach().cpu().numpy().reshape(-1)
                lines.append(f'{name} {len(values)}')
                for start in range(0,len(values),8):
                    lines.append(' '.join(format(float(v),'.9g') for v in values[start:start+8]))
            append('image',image)
            for name in parameter_order: append(name,state[name])
            for name in order: append(name,taps[name])
            (args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
        cases.append({'prefix':case,'input':filename,'order':order,
                      'shapes':{key:list(taps[key].shape) for key in order}})
        for key in order:
            name=case+'.'+key; tensors[name]=taps[key].cpu().numpy().copy()
            rules.append({'name':name,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {case}: {depth} blocks, {batch}x{h}x{w}, dim {dim}',flush=True)
        del net,state,taps,baseline,observed
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='original Body wrapper and complete DINO synthetic backbone; not trained-model parity'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'6876159a11b4df116f30f667f8c9888617df0751',
        'source_hashes':HASHES,'body_source_hashes':{BODY_PATH:BODY_HASH},
        'body_revision':'b5c765a0d89d789985e186d396315e7590887b94',
        'device':args.device,'dtype':'float32','torch':torch.__version__,'numpy':np.__version__,
        'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'capture_script_sha256':digest(Path(__file__)),
        'threads':1,'tf32_matmul':False,'tf32_cudnn':False,'synthetic_weights':True,
        'unobserved_vs_observed':'exact_equal_all_cases','full_body_backbone_shape':args.full_shape,
        'cases':cases,'artifacts':{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json': manifest['artifacts'][path.name]=digest(path)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
