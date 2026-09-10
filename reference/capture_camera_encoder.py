#!/usr/bin/env python3
"""Original Body CameraEncoder component contracts; synthetic weights/features/rays."""
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
from safetensors.numpy import save_file

HASHES={
    'camera_embed.py':'42f2483368fa3249557464aaa188c5d422f4e2faeb29ecd6facc131a97141539',
    'transformer.py':'0cc343878ba8cdf28f9061de53ac0bc0fd6f829889ade96341729db5860ecbe4',
    'swiglu_ffn.py':'7a25fff5fa99c910c8daea7737f3650881a92da11388ddfbfd57794dc9de5355',
    'layer_scale.py':'ef4e0f769dd072425384f354c9527e2fdd7346843379e5f3b0cb2110a72c1c81',
    'drop_path.py':'04c2f8bf7f13382c3d5c992fca00692a0591d187a2d271f49afaa5bce2ec9525',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--small-regression',action='store_true')
    args=parser.parse_args()
    root=args.upstream/'sam_3d_body/models/modules'
    for name,digest in HASHES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'upstream hash mismatch: {name}')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.modules']:
        package=types.ModuleType(name); package.__path__=[str(args.upstream/Path(*name.split('.')))]; sys.modules[name]=package
    Camera=importlib.import_module('sam_3d_body.models.modules.camera_embed').CameraEncoder
    torch.set_num_threads(1); torch.manual_seed(8501); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.small_regression and args.device!='cpu': raise ValueError('compact regression is CPU only')
    rng=np.random.default_rng(8501)
    shapes=[(2,8,12,2,8),(1,9,15,3,32),(1,3,5,1,4),(1,512,512,16,1280)]
    if args.small_regression: shapes=[(2,4,6,2,4)]
    tensors,cases,rules={},[],[]; args.output.mkdir(parents=True,exist_ok=True)
    for index,(batch,h,w,patch,dim) in enumerate(shapes):
        net=Camera(dim,patch).float().to(args.device).eval()
        with torch.no_grad():
            for name,p in net.named_parameters():
                if p.ndim==4: values=rng.normal(scale=1/np.sqrt(dim+99),size=tuple(p.shape))
                elif name.endswith('weight'): values=rng.uniform(.8,1.2,size=tuple(p.shape))
                else: values=rng.normal(scale=.1,size=tuple(p.shape))
                p.copy_(torch.from_numpy(values.astype(np.float32)))
            features=torch.from_numpy(rng.normal(size=(batch,dim,h//patch,w//patch)).astype(np.float32)).to(args.device)
            rays=torch.from_numpy(rng.normal(scale=.6,size=(batch,2,h,w)).astype(np.float32)).to(args.device)
            baseline=net(features,rays).clone(); taps={}
            def capture(name,value): taps[name]=value.detach().clone()
            def camera_input(_m,_a,kw): capture('01.positions',kw['pos'])
            def bnd(v): return v.flatten(2).transpose(1,2).contiguous()
            hooks=[net.camera.register_forward_pre_hook(camera_input,with_kwargs=True),
                net.camera.register_forward_hook(lambda _m,_a,v:capture('03.fourier',v)),
                net.conv.register_forward_pre_hook(lambda _m,a:capture('04.joined',bnd(a[0]))),
                net.conv.register_forward_hook(lambda _m,_a,v:capture('05.projection',bnd(v))),
                net.norm.register_forward_hook(lambda _m,_a,v:capture('06.normalized',bnd(v)))]
            observed=net(features,rays)
            if not torch.equal(baseline,observed): raise ValueError('camera instrumentation changed output')
            for hook in hooks: hook.remove()
            capture('07.output',observed)
            capture('00.rays_downsampled',taps['01.positions'][...,:2].transpose(1,2).reshape(batch,2,h//patch,w//patch))
            # Auxiliary constant diagnostic, not a tap inside original Fourier.
            capture('02.frequencies',torch.stack([torch.linspace(1,32,16,device=args.device) for _ in range(3)]))
        case=f'case.{index:04d}'; filename=case+'.input'; state=net.state_dict()
        parameter_order=['conv.weight','norm.weight','norm.bias']
        if set(state)!=set(parameter_order): raise ValueError('unexpected original CameraEncoder parameters')
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DCAM01'+struct.pack('<5I',batch,h,w,patch,dim))
            for v in [features,rays,*[state[name] for name in parameter_order]]:
                stream.write(v.cpu().numpy().astype('<f4').tobytes())
        order=sorted(taps)
        if args.small_regression:
            lines=['S3D_CAMERA_REGRESSION_V1',f'{batch} {h} {w} {patch} {dim}']
            def append(name,v):
                values=v.detach().cpu().numpy().reshape(-1); lines.append(f'{name} {len(values)}')
                for start in range(0,len(values),8): lines.append(' '.join(format(float(v),'.9g') for v in values[start:start+8]))
            append('features',features); append('rays',rays)
            for name in parameter_order: append(name,state[name])
            for name in order: append(name,taps[name])
            (args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
        cases.append({'prefix':case,'input':filename,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()}})
        for key in order:
            name=case+'.'+key; tensors[name]=taps[key].cpu().numpy().copy()
            rules.append({'name':name,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {case}: {batch}x{h}x{w}, patch={patch}, channels={dim}',flush=True)
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='original CameraEncoder synthetic component; not trained or image-to-pose parity'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'b5c765a0d89d789985e186d396315e7590887b94','source_hashes':HASHES,
        'capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'device':args.device,'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'torch':torch.__version__,'numpy':np.__version__,'threads':1,'tf32_matmul':False,'tf32_cudnn':False,
        'unobserved_vs_observed':'exact_equal_all_cases','auxiliary_taps':['02.frequencies'],
        'cases':cases,'artifacts':{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json': manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
