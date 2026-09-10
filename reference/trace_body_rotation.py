#!/usr/bin/env python3
"""Isolate original global rotations on both original/native full-decoder predictions.

No learned weights or checkpoint deserialization. Run in reviewed container.
"""
import argparse,hashlib,importlib,json,struct,sys,types
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file,save_file
from capture_body_pose import ROMA_SHA
from capture_camera_head import GEOMETRY_SHA256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','roma-wheel','reference','native','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    source=a.upstream/'sam_3d_body/models/modules/geometry_utils.py';wheel=a.roma_wheel.resolve()
    if digest(source)!=GEOMETRY_SHA256 or digest(wheel)!=ROMA_SHA:raise ValueError('unverified source')
    sys.path.insert(0,str(wheel));import roma
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.modules']:
        m=types.ModuleType(name);m.__path__=[str(a.upstream/Path(*name.split('.')))];sys.modules[name]=m
    rotation=importlib.import_module('sam_3d_body.models.modules.geometry_utils').rot6d_to_rotmat
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True);tensors={};inputs=[];cases=[]
    for label,path in [('original',a.reference),('native',a.native)]:
        data=load_file(path)
        for layer in range(6):
            key=f'case.0000.layer.{layer}.pose.pose.10.pred';x=data[key][:,:6].copy();inputs.append(x);prefix=f'{label}.{layer}';taps={}
            def keep(k,v):taps[k]=v.detach().cpu().numpy().copy()
            def profile(frame,event,value):
                if event!='return':return
                name,filename=frame.f_code.co_name,frame.f_code.co_filename;l=frame.f_locals
                if name=='rot6d_to_rotmat' and filename==str(source):
                    for k,n in [('11.global.b1','b1'),('12.global.b2','b2'),('13.global.b3','b3')]:keep(k,l[n])
                    keep('14.global.matrix',value)
                if filename.startswith(str(wheel)):
                    if name=='rotmat_to_unitquat':keep('15.global.choice',l['choices'].float());keep('16.global.quaternion',value)
                    if name=='unitquat_to_euler':
                        keep('diagnostic.abcd',torch.stack([l[k] for k in ['a','b','c','d']],dim=-1))
                        keep('diagnostic.hypot',torch.stack([torch.hypot(l['c'],l['d']),torch.hypot(l['a'],l['b'])],dim=-1))
                        keep('diagnostic.middle',2*torch.atan2(torch.hypot(l['c'],l['d']),torch.hypot(l['a'],l['b'])))
            t=torch.from_numpy(x).to(a.device)
            baseline=roma.rotmat_to_euler('ZYX',rotation(t));old=sys.getprofile()
            try:sys.setprofile(profile);value=roma.rotmat_to_euler('ZYX',rotation(t))
            finally:sys.setprofile(old)
            if not torch.equal(value,baseline):raise ValueError('observer changed output')
            keep('17.global.euler',value)
            for k,v in taps.items():tensors[prefix+'.'+k]=v
            cases.append({'prefix':prefix,'source':key})
    with (a.output/'input.bin').open('wb') as f:f.write(struct.pack('<I',len(inputs))+np.concatenate(inputs).astype('<f4').tobytes())
    save_file(tensors,a.output/'upstream.safetensors')
    (a.output/'manifest.json').write_text(json.dumps({'device':a.device,'cases':cases,'keys':sorted(taps),'script_sha256':digest(Path(__file__)),'reference_sha256':digest(a.reference),'native_sha256':digest(a.native)},indent=2)+'\n')
if __name__=='__main__':main()
