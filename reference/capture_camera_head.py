#!/usr/bin/env python3
"""Original PerspectiveHead + original full-perspective projection; synthetic state/points."""
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
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.numpy import save_file
from capture_camera_encoder import HASHES

HEAD_SHA256='84a14112e5c4b216559c1f9a39c3a33d6fa277182c2037f44a76a9cb538de55e'
GEOMETRY_SHA256='2553899d9a0f781da1a1d73e8d49dcdd08376c5c0e471d6262e662229b645ef1'


class ProjectionTrace(TorchDispatchMode):
    def __init__(self,taps):super().__init__();self.taps=taps
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        value=func(*args,**(kwargs or {}))
        if func==torch.ops.aten.div.Tensor and isinstance(value,torch.Tensor) and value.ndim==3:
            if '18.normalized' in self.taps:raise ValueError('unexpected additional projection division')
            self.taps['18.normalized']=value.detach().clone()
        return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--small-regression',action='store_true')
    args=parser.parse_args();root=args.upstream/'sam_3d_body'
    hashes={f'models/modules/{k}':v for k,v in HASHES.items()}
    hashes|={'models/heads/camera_head.py':HEAD_SHA256,'models/modules/geometry_utils.py':GEOMETRY_SHA256,
        'models/modules/__init__.py':'ddad96a6a9bf09d36156ba82ebaa9ee6798045fe1fc715174cec4c0b4bd2ead1',
        'models/modules/misc.py':'f942d181ee8578c67a9ab4c0828d030aefdadd8891450ab1319b33a8237891eb'}
    for name,digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'source changed: {name}')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.heads']:
        package=types.ModuleType(name);package.__path__=[str(args.upstream/Path(*name.split('.')))];sys.modules[name]=package
    Head=importlib.import_module('sam_3d_body.models.heads.camera_head').PerspectiveHead
    torch.set_num_threads(1);torch.manual_seed(8604);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(8604)
    # B,D,hidden,depth,N,center convention,initial estimate,scale factor.
    shapes=[(2,16,8,1,5,0,0,1.),(1,16,8,2,7,0,1,.7),(2,32,8,3,3,1,1,1.6)]
    if not args.small_regression:shapes.extend([(1,1024,128,1,70,1,1,1.),(1,1024,128,2,18439,0,1,.8)])
    args.output.mkdir(parents=True,exist_ok=True);cases,tensors,rules=[],{},[]
    lines=['S3D_CAMERA_HEAD_REGRESSION_V1',str(len(shapes))]
    for index,(b,d,f,depth,n,center,has_initial,scale) in enumerate(shapes):
        scale=float(np.float32(scale));net=Head(d,(512,512),mlp_depth=depth,mlp_channel_div_factor=d//f,default_scale_factor=scale).float().to(args.device).eval()
        def random(shape,scale=1):return torch.from_numpy(rng.normal(scale=scale,size=shape).astype(np.float32)).to(args.device)
        with torch.no_grad():
            for name,p in net.named_parameters():p.copy_(random(tuple(p.shape),.2/np.sqrt(p.shape[1]) if p.ndim==2 else .03))
            net.proj.layers[-2].bias[0]=1.5 if index==2 else -1.5
            token=random((b,d));initial=random((b,3),.08) if has_initial else None
            points=random((b,n,3),.15)
            image_size=torch.tensor([[997,613] if z==0 else [1920,1080] for z in range(b)],dtype=torch.float32,device=args.device)
            bbox_center=image_size*torch.tensor([.43,.61],device=args.device);box=torch.tensor([450+50*z for z in range(b)],dtype=torch.float32,device=args.device)
            intrinsics=torch.tensor([[[850+90*z,3,.46*float(image_size[z,0])],[-2,925+80*z,.52*float(image_size[z,1])],[0,0,1]] for z in range(b)],dtype=torch.float32,device=args.device)
            def project(cam):return net.perspective_projection(points,cam,bbox_center,box,image_size,intrinsics,use_intrin_center=bool(center))
            base_cam=net(token,initial).clone();baseline={k:v.clone() for k,v in project(base_cam).items()}
            taps={};hooks=[]
            def capture(name,v):taps[name]=v.detach().clone()
            for i in range(depth):
                module=net.proj.layers[i] if i+1==depth else net.proj.layers[i][0]
                hooks.append(module.register_forward_hook(lambda _m,_a,v,i=i:capture(f'00.ffn.{i}.linear',v)))
                if i+1<depth:hooks.append(net.proj.layers[i][1].register_forward_hook(lambda _m,_a,v,i=i:capture(f'00.ffn.{i}.relu',v)))
            pred_cam=net(token,initial);capture('10.pred_cam',pred_cam)
            for hook in hooks:hook.remove()
            def profile(frame,event,value):
                if event!='return' or frame.f_code.co_name!='perspective_projection':return
                local=frame.f_locals
                if frame.f_code.co_filename==str(root/'models/heads/camera_head.py'):
                    for name,key in [('11.corrected_cam','pred_cam'),('12.scaled_box','bs'),('13.focal','focal_length'),('15.translation','pred_cam_t'),('16.camera_points','j3d_cam')]:capture(name,local[key])
                    capture('14.offset',torch.stack([local['cx'],local['cy']],dim=-1))
                    capture('17.depth',value['pred_keypoints_2d_depth']);capture('20.pixels',value['pred_keypoints_2d'])
                elif frame.f_code.co_filename==str(root/'models/modules/geometry_utils.py'):capture('19.intrinsic_projection',local['y'])
            previous_profile=sys.getprofile()
            try:
                sys.setprofile(profile)
                with ProjectionTrace(taps):observed=project(pred_cam)
            finally:sys.setprofile(previous_profile)
            if not torch.equal(base_cam,pred_cam) or not all(torch.equal(v,observed[k]) for k,v in baseline.items()):raise ValueError('camera-head instrumentation changed outputs')
        if len(taps)!=2*depth+10:raise ValueError(f'missing camera-head taps: {sorted(taps)}')
        inputs={'token':token,'points':points,'center':bbox_center,'box':box,'image_size':image_size,'intrinsics':intrinsics}
        if initial is not None:inputs['initial']=initial
        state=dict(net.state_dict());prefix=f'case.{index:04d}';filename=prefix+'.input'
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DCHD01'+struct.pack('<7If',b,d,f,depth,n,center,has_initial,scale))
            for v in [token,*([initial] if initial is not None else []),points,bbox_center,box,image_size,intrinsics,*[state[k] for k in sorted(state)]]:stream.write(v.cpu().numpy().astype('<f4').tobytes())
        if args.small_regression:
            lines.append(' '.join(map(str,(b,d,f,depth,n,center,has_initial,scale))))
            for group in [inputs,state,taps]:
                lines.append(str(len(group)))
                for name,v in sorted(group.items()):
                    a=v.cpu().numpy().reshape(-1);lines.append(f'{name} {a.size}')
                    for start in range(0,len(a),8):lines.append(' '.join(format(float(v),'.9g') for v in a[start:start+8]))
        order=sorted(taps);cases.append({'prefix':prefix,'input':filename,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()}})
        for name in order:
            key=prefix+'.'+name;tensors[key]=taps[name].cpu().numpy().copy()
            # Distinct units: sub-millipixel/box bounds, tighter camera/latent bounds.
            rules.append({'name':key,'mode':'exact'} if name=='13.focal' else {'name':key,'mode':'float',
                'max_abs':1e-3 if name in ['12.scaled_box','19.intrinsic_projection','20.pixels'] else 1e-4,
                'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {prefix}, depth={depth}, points={n}, taps={len(taps)}',flush=True)
    if args.small_regression:(args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='original PerspectiveHead FFN and complete perspective reprojection; synthetic token/state/3D points, no MHR or full Body inference'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'b5c765a0d89d789985e186d396315e7590887b94','source_hashes':hashes,
        'capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':args.device,
        'torch':torch.__version__,'numpy':np.__version__,'threads':1,'tf32_matmul':False,'tf32_cudnn':False,
        'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'unobserved_vs_observed':'exact_equal_all_head_and_projection_outputs',
        'observation':'original module hooks, Python return-local capture and non-replacing ATen division observer',
        'cases':cases,'artifacts':{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json':manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
