#!/usr/bin/env python3
"""Actual Body PromptEncoder/PositionEmbeddingRandom contracts, synthetic state."""
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

PROMPT_SHA256='a9160aa5ccdd4d5e2ae047605d86eee9e0e558344b85f716d963fd091f9a33d1'


class PositionTrace(TorchDispatchMode):
    """Observe original aten outputs; never replace an expression or its result."""
    def __init__(self,coords):
        super().__init__();self.shape=tuple(coords.shape);self.taps={'00.coords':coords.detach().clone()};self.multiplies=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}));name=None
        if func==torch.ops.aten.mul.Tensor:
            name=['01.doubled','04.angles'][self.multiplies];self.multiplies+=1
        elif func==torch.ops.aten.sub.Tensor:name='02.centered'
        elif func==torch.ops.aten.mm.default:name='03.projected'
        elif func==torch.ops.aten.sin.default:name='05.sin'
        elif func==torch.ops.aten.cos.default:name='06.cos'
        elif func==torch.ops.aten.cat.default:name='07.encoding'
        if name:
            if name in self.taps:raise ValueError(f'duplicate operation tap {name}')
            value=result.detach().clone()
            if name=='03.projected':value=value.reshape(*self.shape[:-1],value.shape[-1])
            self.taps[name]=value
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--small-regression',action='store_true')
    args=parser.parse_args();modules=args.upstream/'sam_3d_body/models/modules'
    for name,digest in HASHES.items():
        if hashlib.sha256((modules/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'source changed: {name}')
    source=args.upstream/'sam_3d_body/models/decoders/prompt_encoder.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=PROMPT_SHA256:raise ValueError('prompt source changed')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.modules','sam_3d_body.models.decoders']:
        package=types.ModuleType(name);package.__path__=[str(args.upstream/Path(*name.split('.')))];sys.modules[name]=package
    Prompt=importlib.import_module('sam_3d_body.models.decoders.prompt_encoder').PromptEncoder
    torch.set_num_threads(1);torch.manual_seed(8602);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(8602)
    shapes=[(2,5,8,3,3,5,401,997),(1,1,4,1,1,1,7,9),(2,6,16,4,5,3,1080,1920)]
    if not args.small_regression:shapes.append((1,72,1280,70,32,32,1080,1920))
    args.output.mkdir(parents=True,exist_ok=True);cases,tensors,rules=[],{},[]
    lines=['S3D_PROMPT_REGRESSION_V1',str(len(shapes))]
    for index,(b,n,d,j,h,w,ih,iw) in enumerate(shapes):
        net=Prompt(d,j).float().to(args.device).eval()
        def random(shape):return torch.from_numpy(rng.normal(size=shape).astype(np.float32)).to(args.device)
        with torch.no_grad():
            for name,p in net.state_dict().items():p.copy_(random(tuple(p.shape)))
            coords=rng.uniform(0,1,(b,n,2)).astype(np.float32);coords.reshape(-1,2)[:3]=np.array([[0,0],[1,1],[.5,.5]])[:min(3,b*n)]
            labels=(np.arange(b*n)%(j+2)-2).astype(np.float32).reshape(b,n,1)
            points=torch.from_numpy(np.concatenate([coords,labels],axis=-1)).to(args.device)
            pixel=torch.from_numpy((rng.uniform(-.5,1.5,(b,n,2))*np.array([iw,ih])).astype(np.float32)).to(args.device)
            baseline=[net.get_dense_pe((h,w)).clone(),*[v.clone() for v in net(points)],net.pe_layer.forward_with_coords(pixel,(ih,iw)).clone()]
            taps={};original=net.pe_layer._pe_encoding;phase='00.dense'
            def observe(coords):
                trace=PositionTrace(coords)
                with trace:value=original(coords)
                if len(trace.taps)!=8:raise ValueError(f'missing original operation: {trace.taps.keys()}')
                for name,v in trace.taps.items():taps[phase+'.'+name]=v
                return value
            # Observation-only wrapper calls the unchanged bound method. Its
            # ATen dispatch calls each original op once and returns it unchanged.
            net.pe_layer._pe_encoding=observe
            dense=net.get_dense_pe((h,w));phase='10.point';embedding,mask=net(points)
            phase='30.pixel';pixels=net.pe_layer.forward_with_coords(pixel,(ih,iw))
            net.pe_layer._pe_encoding=original
            if not all(torch.equal(a,v) for a,v in zip(baseline,[dense,embedding,mask,pixels])):raise ValueError('observer changed original result')
            taps['20.embeddings']=embedding;taps['21.mask']=mask;taps['22.dense_nchw']=dense
            if not torch.equal(pixels,taps['30.pixel.07.encoding']):raise ValueError('pixel result tap mismatch')
        prefix=f'case.{index:04d}';filename=prefix+'.input';state=dict(net.state_dict());inputs={'keypoints':points,'pixels':pixel}
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DPRM01'+struct.pack('<8I',b,n,d,j,h,w,ih,iw))
            for v in [points,pixel,*[state[k] for k in sorted(state)]]:stream.write(v.cpu().numpy().astype('<f4').tobytes())
        if args.small_regression:
            lines.append(' '.join(map(str,(b,n,d,j,h,w,ih,iw))))
            for group in [inputs,state,taps]:
                lines.append(str(len(group)))
                for name,v in sorted(group.items()):
                    a=v.cpu().numpy().reshape(-1);lines.append(f'{name} {a.size}')
                    for start in range(0,len(a),8):lines.append(' '.join(format(float(v),'.9g') for v in a[start:start+8]))
        order=sorted(taps);cases.append({'prefix':prefix,'input':filename,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()}})
        for name in order:
            key=prefix+'.'+name;tensors[key]=taps[name].cpu().numpy().copy()
            rules.append({'name':key,'mode':'exact'} if name.endswith('.00.coords') or name=='21.mask' else
                {'name':key,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {prefix} shape={(b,n,d,j,h,w,ih,iw)} taps={len(taps)}',flush=True)
    if args.small_regression:(args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='original Body PromptEncoder and pixel/dense PositionEmbeddingRandom; synthetic state, not learned or image-to-pose parity'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'b5c765a0d89d789985e186d396315e7590887b94',
        'module_source_hashes':HASHES,'prompt_encoder_sha256':PROMPT_SHA256,
        'capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch':torch.__version__,'numpy':np.__version__,'device':args.device,'threads':1,
        'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'tf32_matmul':False,'tf32_cudnn':False,'unobserved_vs_observed':'exact_equal_all_cases',
        'observation':'unchanged bound _pe_encoding method with non-replacing ATen output observer; no rewritten oracle',
        'auxiliary_taps':[],'cases':cases,'artifacts':{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json':manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
