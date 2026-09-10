#!/usr/bin/env python3
"""Original crop/batch/ray/CLIFF methods, not model inference.

SAM3DBody's two methods are compiled from their unchanged hash-verified ASTs to
avoid unrelated full-model imports/assets. No expression/body is rewritten.
BaseModel and preparation/transforms are original imported code. get_ray_condition
contains an upstream hardcoded .cuda(), so this capture requires NVIDIA CUDA.
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
from typing import Dict, Optional

import numpy as np
import torch
from safetensors.numpy import save_file
from torchvision.transforms import ToTensor
from yacs.config import CfgNode

HASHES={
    'models/meta_arch/sam3d_body.py':'851b7475f18b56891aa02606e7c0ee9e03120fa208cc85df5127b792e1abfeee',
    'models/meta_arch/base_model.py':'baf4c93ab865e6e9f4f498056a673698e59bafe89b17f969833884a3b352e8bc',
    'data/utils/prepare_batch.py':'c0555035944f02cfdb9d811b377a3c1ffdd2bfbfe256ee6dc501c8d9409298d0',
    'data/transforms/common.py':'165697629df2fe23a62bb90daa2ac8a0f67cac09d70f36844b3c089a9af5a45f',
    'data/transforms/bbox_utils.py':'0f49a3f857a09f98d1fd9c609df42f875899515f5e10a0ea6b25acd7a9197d79',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--small-regression',action='store_true')
    args=parser.parse_args(); root=args.upstream/'sam_3d_body'
    for name,digest in HASHES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest: raise ValueError(f'upstream hash mismatch: {name}')
    for name in ['sam_3d_body','sam_3d_body.data','sam_3d_body.data.utils','sam_3d_body.models',
                 'sam_3d_body.models.meta_arch','sam_3d_body.models.optim']:
        package=types.ModuleType(name);package.__path__=[str(args.upstream/Path(*name.split('.')))];sys.modules[name]=package
    transforms=importlib.import_module('sam_3d_body.data.transforms.common')
    prepare=importlib.import_module('sam_3d_body.data.utils.prepare_batch').prepare_batch
    Base=importlib.import_module('sam_3d_body.models.meta_arch.base_model').BaseModel
    source=root/'models/meta_arch/sam3d_body.py'
    tree=ast.parse(source.read_text(),filename=str(source))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='SAM3DBody')
    names=['get_ray_condition','_get_decoder_condition']
    selected=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(selected)!=2 or any(node.decorator_list for node in selected): raise ValueError('unexpected upstream method definition')
    namespace={'torch':torch,'Dict':Dict,'Optional':Optional}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    holder=types.SimpleNamespace(_batch_size=1,_max_num_person=1)
    holder._flatten_person=types.MethodType(Base._flatten_person,holder)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    cases,tensors,rules=[],{},[]; lines=['S3D_CAMERA_GEOMETRY_REGRESSION_V1','12']
    for index in range(12):
        w,h=[(640,480),(1920,1080),(7,9),(997,613)][index%4]
        side=[1,4,16,512][index%4]
        if args.small_regression: side=min(side,8)
        padding=float(np.float32([1.25,.9,1.3][index%3])); use_center=index%2
        box=np.array([-.15*w,.13*h,.87*w,1.1*h] if index>=6 else [.11*w,.2*h,.85*w,.9*h],dtype=np.float32)
        fx=np.float32([500,1433,3,987.3][index%4]); fy=np.float32(fx*1.13)
        intrinsics=np.array([fx,fy,w*.45,h*.56],dtype=np.float32)
        camera=torch.tensor([[[fx,0,intrinsics[2]],[0,fy,intrinsics[3]],[0,0,1]]],dtype=torch.float32)
        transform=transforms.Compose([transforms.GetBBoxCenterScale(padding=padding),
            transforms.TopdownAffine(input_size=(side,side)),transforms.VisionTransformWrapper(ToTensor())])
        batch=prepare(np.zeros((h,w,3),dtype=np.uint8),transform,box[None],cam_int=camera)
        batch={key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in batch.items()}
        holder.cfg=CfgNode({'MODEL':{'DECODER':{'CONDITION_TYPE':'cliff','USE_INTRIN_CENTER':bool(use_center)}}})
        with torch.no_grad():
            rays=namespace['get_ray_condition'](holder,batch)
            cliff=namespace['_get_decoder_condition'](holder,batch)
        case=f'case.{index:04d}'; filename=case+'.input'
        (args.output/filename).write_bytes(b'S3DRAY01'+struct.pack('<4I9f',w,h,side,use_center,padding,*box,*intrinsics))
        taps={'00.center':batch['bbox_center'][0,0], '01.scale':batch['bbox_scale'][0,0],
            '02.affine':batch['affine_trans'][0,0],'03.rays':rays[0,0],'04.cliff':cliff[0]}
        order=sorted(taps)
        cases.append({'prefix':case,'input':filename,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()}})
        for key in order:
            name=case+'.'+key;tensors[name]=taps[key].cpu().numpy().copy()
            rules.append({'name':name,'mode':'float','max_abs':1e-6,'relative_l2':1e-6,'zero_reference_floor':1e-12})
        if args.small_regression:
            lines.append(' '.join(map(str,[w,h,side,use_center,padding,*box,*intrinsics])))
            for key in order:
                values=taps[key].cpu().numpy().reshape(-1); lines.append(f'{key} {len(values)}')
                for start in range(0,len(values),8): lines.append(' '.join(format(float(v),'.9g') for v in values[start:start+8]))
    if args.small_regression: (args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='original prepare_batch/crop metadata + unmodified ray and CLIFF method ASTs; not full model'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'b5c765a0d89d789985e186d396315e7590887b94','source_hashes':HASHES,
        'selected_methods':names,'method_changes':'none; isolated unchanged AST bodies with original BaseModel flatten',
        'capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch':torch.__version__,'numpy':np.__version__,'device':torch.cuda.get_device_name(),'dtype':'float32',
        'cases':cases,'artifacts':{}}
    for path in args.output.iterdir():
        if path.is_file() and path.name!='manifest.json':manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Captured 12 original crop/batch/camera cases, 60 boundaries',flush=True)


if __name__=='__main__': main()
