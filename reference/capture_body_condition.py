#!/usr/bin/env python3
"""Original Body conditioning through first decoder layer; synthetic state, no pose heads.

The unchanged forward_decoder AST calls real original prompt/camera/decoder
modules. A forward hook stops immediately AFTER the first real decoder layer,
before pose/geometry callbacks. No fake final pose/output is returned. This is
explicitly a partial composition boundary, not complete decoder/model parity.
"""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys
import types
from typing import Optional

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.numpy import save_file
from yacs.config import CfgNode
from capture_camera_encoder import HASHES
from capture_body_prompt import PROMPT_SHA256

BODY_SHA256='851b7475f18b56891aa02606e7c0ee9e03120fa208cc85df5127b792e1abfeee'
DECODER_SHA256='73c1219c176478e52358d9448d8087017e8294202653ca9fb62e2c964f7c363a'


class BoundaryReached(Exception):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--small-regression',action='store_true')
    args=parser.parse_args();root=args.upstream/'sam_3d_body'
    hashes={f'models/modules/{name}':digest for name,digest in HASHES.items()}
    hashes|={'models/decoders/prompt_encoder.py':PROMPT_SHA256,'models/decoders/promptable_decoder.py':DECODER_SHA256,
             'models/meta_arch/sam3d_body.py':BODY_SHA256,
             'models/heads/mhr_head.py':'03793d51ef82484c5d9906358c6314981c0bff9dbcc6f6e6b175bd977a251b35'}
    for name,digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'source changed: {name}')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.modules','sam_3d_body.models.decoders']:
        package=types.ModuleType(name);package.__path__=[str(args.upstream/Path(*name.split('.')))];sys.modules[name]=package
    Prompt=importlib.import_module('sam_3d_body.models.decoders.prompt_encoder').PromptEncoder
    Camera=importlib.import_module('sam_3d_body.models.modules.camera_embed').CameraEncoder
    Decoder=importlib.import_module('sam_3d_body.models.decoders.promptable_decoder').PromptableDecoder
    source=root/'models/meta_arch/sam3d_body.py';tree=ast.parse(source.read_text(),filename=str(source))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SAM3DBody')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward_decoder')
    if method.decorator_list:raise ValueError('unexpected method decorators')
    namespace={'torch':torch,'Optional':Optional}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    torch.set_num_threads(1);torch.manual_seed(8603);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(8603)
    # B,H,W,patch,contextD,tokenD,promptN,promptJ,poseD,keypoints,3D,hand,prev,heads,headD,FFN.
    shapes=[(2,4,6,2,8,8,2,3,5,3,0,0,0,2,4,16),
            (1,6,4,2,8,8,3,3,5,3,1,0,1,2,4,16),
            (2,4,4,2,8,8,1,3,5,3,1,1,1,2,4,16)]
    if not args.small_regression:shapes.append((1,512,512,16,1280,1024,1,70,519,70,1,0,0,8,64,4096))
    args.output.mkdir(parents=True,exist_ok=True);cases,tensors,rules=[],{},[]
    lines=['S3D_CONDITION_REGRESSION_V1',str(len(shapes))]
    for index,shape in enumerate(shapes):
        b,h,w,patch,c,d,n,j,pose,kps,kps3,hand,has_prev,heads,dh,hidden=shape
        holder=torch.nn.Module();holder.cfg=CfgNode({'MODEL':{'BACKBONE':{'TYPE':'dinov3'},'DECODER':{
            'DO_KEYPOINT_TOKENS':True,'DO_KEYPOINT3D_TOKENS':bool(kps3),'DO_HAND_DETECT_TOKENS':bool(hand)}}})
        holder.init_pose=torch.nn.Embedding(1,pose);holder.init_camera=torch.nn.Embedding(1,3)
        holder.init_to_token_mhr=torch.nn.Linear(pose+6,d);holder.prev_to_token_mhr=torch.nn.Linear(pose+3,d)
        holder.prompt_encoder=Prompt(c,j);holder.prompt_to_token=torch.nn.Linear(c,d);holder.ray_cond_emb=Camera(c,patch)
        holder.keypoint_embedding=torch.nn.Embedding(kps,d)
        if kps3:holder.keypoint3d_embedding=torch.nn.Embedding(kps,d)
        if hand:holder.hand_box_embedding=torch.nn.Embedding(2,d)
        holder.keypoint_token_update_fn=None;holder.keypoint3d_token_update_fn=None
        holder.decoder=Decoder(d,c,depth=1,num_heads=heads,head_dims=dh,mlp_dims=hidden,
            repeat_pe=True,enable_twoway=True,do_interm_preds=True,keypoint_token_update=True)
        holder=holder.float().to(args.device).eval()
        def random(shape,scale=1):return torch.from_numpy(rng.normal(scale=scale,size=shape).astype(np.float32)).to(args.device)
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            for name,p in holder.state_dict().items():
                if name.endswith('positional_encoding_gaussian_matrix'):values=random(tuple(p.shape))
                elif '.ln' in name and name.endswith('weight') or name.endswith('norm.weight'):
                    values=torch.from_numpy(rng.uniform(.8,1.2,size=tuple(p.shape)).astype(np.float32)).to(args.device)
                elif p.ndim>=2 and p.shape[0]!=1 and name.endswith('weight') and 'embedding' not in name:
                    values=random(tuple(p.shape),1/np.sqrt(np.prod(p.shape[1:])))
                else:values=random(tuple(p.shape),.15)
                p.copy_(values)
            image=random((b,c,h//patch,w//patch));rays=random((b,2,h,w),.6);cliff=random((b,3),.2)
            keypoints=torch.from_numpy(rng.uniform(0,1,(b,n,3)).astype(np.float32)).to(args.device)
            keypoints[...,2]=torch.tensor((np.arange(b*n)%(j+2)-2).reshape(b,n),device=args.device)
            if index==3:keypoints.zero_();keypoints[...,2]=-2
            previous=random((b,1,pose+3)) if has_prev else None
            boundary={}
            def stop(_m,_a,value):
                boundary['40.layer_tokens']=value[0].clone();boundary['41.layer_context']=value[1].clone();raise BoundaryReached()
            stop_hook=holder.decoder.layers[0].register_forward_hook(stop)
            def run():
                try:namespace['forward_decoder'](holder,image,keypoints=keypoints,prev_estimate=previous,condition_info=cliff,batch={'ray_cond':rays})
                except BoundaryReached:return
                raise ValueError('original decoder boundary not reached')
            run();baseline={k:v.clone() for k,v in boundary.items()};taps={};hooks=[]
            def capture(name,v):taps[name]=v.detach().clone()
            def pre(module,name):hooks.append(module.register_forward_pre_hook(lambda _m,a:capture(name,a[0])))
            def post(module,name):hooks.append(module.register_forward_hook(lambda _m,_a,v:capture(name,v)))
            pre(holder.init_to_token_mhr,'00.init_input');post(holder.init_to_token_mhr,'01.pose_token')
            pre(holder.prev_to_token_mhr,'02.prev_input');post(holder.prev_to_token_mhr,'03.prev_token')
            def prompt_out(_m,_a,value):capture('10.prompt_input',value[0]);capture('11.prompt_mask',value[1])
            hooks.append(holder.prompt_encoder.register_forward_hook(prompt_out));post(holder.prompt_to_token,'12.prompt_token')
            def decoder_in(_m,a):
                if a[4] is not None:raise ValueError('original unexpectedly masks prompt tokens')
                for name,v in zip(['30.tokens','20.image','31.token_pe','21.image_pe'],a[:4]):capture(name,v)
            hooks.append(holder.decoder.register_forward_pre_hook(decoder_in));run()
            for hook in hooks:hook.remove()
            stop_hook.remove()
            if not all(torch.equal(v,boundary[k]) for k,v in baseline.items()):raise ValueError('observation changed first-layer output')
            taps.update(boundary)
        state={k:v for k,v in holder.state_dict().items() if not k.startswith('decoder.') or k.startswith('decoder.layers.0.')}
        inputs={'features':image,'rays':rays,'cliff':cliff,'keypoints':keypoints}
        if previous is not None:inputs['previous']=previous
        prefix=f'case.{index:04d}';filename=prefix+'.input'
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DCND01'+struct.pack('<16I',*shape))
            for v in [image,rays,cliff,keypoints,*([previous] if previous is not None else []),*[state[k] for k in sorted(state)]]:
                stream.write(v.cpu().numpy().astype('<f4').tobytes())
        if args.small_regression:
            lines.append(' '.join(map(str,shape)))
            for group in [inputs,state,taps]:
                lines.append(str(len(group)))
                for name,v in sorted(group.items()):
                    a=v.cpu().numpy().reshape(-1);lines.append(f'{name} {a.size}')
                    for start in range(0,len(a),8):lines.append(' '.join(format(float(v),'.9g') for v in a[start:start+8]))
        order=sorted(taps);cases.append({'prefix':prefix,'input':filename,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()}})
        for name in order:
            key=prefix+'.'+name;tensors[key]=taps[name].cpu().numpy().copy()
            rules.append({'name':key,'mode':'exact'} if name in ['00.init_input','02.prev_input','11.prompt_mask'] else
                {'name':key,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {prefix}, tokens={2+n+2*hand+kps*(1+kps3)}, pose={pose}, taps={len(taps)}',flush=True)
    if args.small_regression:(args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='unchanged Body forward_decoder AST through real original first decoder layer; synthetic inputs/state, stopped before pose/geometry feedback, not full decoder/model parity'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'b5c765a0d89d789985e186d396315e7590887b94','source_hashes':hashes,
        'selected_method':'forward_decoder','method_changes':'none; original AST body isolated to avoid unrelated model assets/imports',
        'capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':args.device,
        'torch':torch.__version__,'numpy':np.__version__,'threads':1,'tf32_matmul':False,'tf32_cudnn':False,'sdpa_backend':'MATH',
        'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'unobserved_vs_observed':'exact_equal_first_layer_outputs_all_cases','decoder_token_mask':'None, asserted',
        'termination':'explicit BoundaryReached from first-layer output hook, before pose heads or intermediate feedback',
        'cases':cases,'artifacts':{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json':manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
