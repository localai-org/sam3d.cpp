#!/usr/bin/env python3
"""Original Objects RGBA/default image-mask pipeline, without learned models.

Unchanged AST extraction avoids unrelated rendering/model imports. Original
PreProcessor and the pipeline's actual preprocess_image methods are executed.
No pointmap methods are used. Captures run in the reviewed isolated container.
"""
import argparse,ast,hashlib,json,logging,struct,sys,types,warnings
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable,Optional,Union
import numpy as np
import torch
import torchvision
from PIL import Image
from safetensors.numpy import save_file

HASHES={
 'pipeline/preprocess_utils.py':'57de199943b3e22e776cc2baeec70caeceea848ccfd9ad82e6096d1ee3b869c4',
 'pipeline/inference_pipeline.py':'e83e9560a727b06565134ea855503034d3510cbfbeeff2b6ca6dc13013536d2f',
 'data/dataset/tdfy/preprocessor.py':'ea61061b81b2c0b08ef5b4f1092ef119fb2c20dfae30ed8f8f3e54ee1c6dd673',
 'data/dataset/tdfy/img_and_mask_transforms.py':'e5ecdb46c12568a1d91e09c83842a16afaca8d1e22d73f49f0b9ded62cc3cc26',
 'data/dataset/tdfy/img_processing.py':'a90b0ce2f96b5950176884a14162b2d4a340bbe1947cd7bf4db2bd03d0f5be3d',
}
PHOTO='0d4d55dcebda52e7be9fd60e6fe68ce3fe52fdee5f7fe086bffda7520466e9ed'
MASK='811eada8e42b6870a403bf63cbf0ba2ec06bf585c993833cc6b4a13585ec77b8'
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def original(root):
 for name,sha in HASHES.items():
  if digest(root/name)!=sha:raise ValueError('unverified original source '+name)
 ns={'torch':torch,'torchvision':torchvision,'np':np,'F':torch.nn.functional,
     'Optional':Optional,'Union':Union,'Callable':Callable,'Image':Image,'dataclass':dataclass,
     'warnings':warnings,'logger':logging.getLogger('original-objects'),'partial':partial,
     'Compose':torchvision.transforms.Compose,'Resize':torchvision.transforms.Resize,
     'InterpolationMode':torchvision.transforms.InterpolationMode,
     'SSINormalizedPointmap':namedtuple('SSINormalizedPointmap',['pointmap','scale','shift'])}
 def select(file,names,cls=None):
  path=root/file;nodes=ast.parse(path.read_text()).body
  if cls:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls).body
  nodes=[n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
  if len(nodes)!=len(names):raise ValueError('missing original definitions')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 select('data/dataset/tdfy/img_processing.py',['pad_to_square_centered'])
 select('data/dataset/tdfy/img_and_mask_transforms.py',['BoundingBoxError','check_bounding_box','concat_rgba','split_rgba','get_mask','compute_mask_bbox','crop_around_mask_with_padding','rembg','SSIPointmapNormalizer'])
 select('data/dataset/tdfy/preprocessor.py',['PreProcessor'])
 select('pipeline/preprocess_utils.py',['get_default_preprocessor'])
 select('pipeline/inference_pipeline.py',['_apply_transform','_preprocess_image_and_mask','image_to_float','preprocess_image'],'InferencePipeline')
 holder=types.SimpleNamespace(device='cpu')
 for name in ['_apply_transform','_preprocess_image_and_mask','image_to_float','preprocess_image']:setattr(holder,name,types.MethodType(ns[name],holder))
 return ns,holder

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);ns,holder=original(a.upstream/'sam3d_objects')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 rng=np.random.default_rng(60123);inputs=[]
 for i,(h,w,side,factor,padding) in enumerate([(9,13,7,1.,.1),(13,9,22,1.6,.1),(11,17,12,.65,.25),(12,12,12,1.,0.),(13,17,31,1.,.1)]):
  rgba=rng.integers(0,256,(h,w,4),dtype=np.uint8);rgba[...,3]=0
  if i==1:rgba[:8,:5,3]=255 # Expanded crop crosses two image borders.
  elif i==4:rgba[1,1,3]=1;rgba[-2,-2,3]=127 # Disconnected, nonbinary alpha.
  else:rgba[1:-1,2:-2,3]=255
  inputs.append((rgba,side,factor,padding))
 folder=a.upstream/'notebook/images/kid_box';photo=folder/'image.png';mask=folder/'0.png'
 if digest(photo)!=PHOTO or digest(mask)!=MASK:raise ValueError('unverified official Objects image/mask')
 rgb=np.asarray(Image.open(photo).convert('RGB'));m=np.asarray(Image.open(mask))
 if m.ndim==3:m=m[...,-1]
 rgba=np.concatenate([rgb,((m>0)*255).astype(np.uint8)[...,None]],axis=-1);inputs.append((rgba,518,1.,.1))
 tensors={};full={};cases=[];rules=[];lines=['S3D_OBJECTS_IMAGE_REGRESSION_V1',str(len(inputs)-1)]
 for i,(rgba,side,factor,padding) in enumerate(inputs):
  pre=ns['get_default_preprocessor']()
  if side!=518:
   pre.img_transform.transforms[-1]=torchvision.transforms.Resize(side,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
   pre.mask_transform.transforms[-1]=torchvision.transforms.Resize(side,interpolation=0)
  pre.img_mask_joint_transform[0]=partial(ns['crop_around_mask_with_padding'],box_size_factor=factor,padding_factor=padding)
  baseline=holder.preprocess_image(rgba,pre)
  for _ in range(2):
   repeat=holder.preprocess_image(rgba,pre)
   if any(not torch.equal(v,repeat[k]) for k,v in baseline.items()):raise ValueError('nonrepeatable original preprocessing')
  taps={};pad_index=0
  def keep(k,v):
   if isinstance(v,torch.Tensor):v=v.detach().cpu().numpy()
   taps[k]=np.array(v,dtype=np.float32,copy=True)
  def observe(frame,event,value):
   nonlocal pad_index
   if event!='return':return
   code=frame.f_code
   if code is ns['image_to_float'].__code__:keep('00.rgba',value.transpose(2,0,1))
   elif code is ns['compute_mask_bbox'].__code__:keep('01.binary_mask',frame.f_locals['mask'][None]);keep('02.bbox',value)
   elif code is ns['crop_around_mask_with_padding'].__code__:keep('03.crop_rgb',value[0]);keep('04.crop_mask',value[1])
   elif code is ns['rembg'].__code__:keep('05.masked_rgb',value[0]);keep('06.masked_mask',value[1])
   elif code is ns['pad_to_square_centered'].__code__:
    keep('07.pad.'+str(pad_index),value);pad_index+=1
  sys.setprofile(observe)
  try:observed=holder.preprocess_image(rgba,pre)
  finally:sys.setprofile(None)
  if pad_index!=4 or any(not torch.equal(v,observed[k]) for k,v in baseline.items()):raise ValueError('observation changed original result')
  for k,v in observed.items():keep('08.'+k,v)
  prefix=f'case.{i:04d}';name=prefix+'.input';h,w=rgba.shape[:2];stride=w*4+5
  raw=np.full((h,stride),191,dtype=np.uint8);raw[:,:w*4]=rgba.reshape(h,w*4)
  with (a.output/name).open('wb') as f:f.write(b'S3DOIM01'+struct.pack('<4I2d',w,h,stride,side,factor,padding)+raw.tobytes())
  order=sorted(taps);cases.append({'prefix':prefix,'input':name,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()},'side':side,'box_factor':factor,'padding':padding,'official_photo':i==len(inputs)-1})
  for k,v in taps.items():
   tensors[prefix+'.'+k]=v
   exact=k.startswith(('00.','01.','02.','03.','04.','05.','06.','07.')) or k.endswith('mask')
   rules.append({'name':prefix+'.'+k,'mode':'exact'} if exact else {'name':prefix+'.'+k,'mode':'float','max_abs':1e-6,'relative_l2':1e-6,'zero_reference_floor':1e-12})
  for k,v in baseline.items():full[prefix+'.'+k]=v.numpy().copy()
  if i<len(inputs)-1:
   lines.append(f'{w} {h} {stride} {side} {factor:.17g} {padding:.17g}');lines.append(' '.join(map(str,raw.reshape(-1))))
   for key in order:
    vals=taps[key].reshape(-1);lines.append(f'{key} {len(vals)}')
    for j in range(0,len(vals),8):lines.append(' '.join(format(float(x),'.9g') for x in vals[j:j+8]))
  print(json.dumps({'case':i,'taps':len(taps),'repeat_and_observation':'exact'}),flush=True)
 rejections=[]
 for label,kind in [('empty',0),('single_pixel',1),('one_row',2),('one_column',3)]:
  x=np.zeros((9,13,4),dtype=np.uint8)
  if kind==1:x[4,5,3]=255
  if kind==2:x[4,2:10,3]=255
  if kind==3:x[2:8,4,3]=255
  try:holder.preprocess_image(x,ns['get_default_preprocessor']())
  except Exception as e:rejections.append({'case':label,'exception':type(e).__name__,'message':str(e)})
  else:raise ValueError('unexpected upstream acceptance of '+label)
 save_file(tensors,a.output/'upstream.safetensors');save_file(full,a.output/'full.safetensors')
 (a.output/'regression.txt').write_text('\n'.join(lines)+'\n')
 (a.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':'Objects default image/mask preprocessing only; no pointmap or model inference','tensors':rules},indent=2)+'\n')
 manifest={'scope':'original default Objects RGBA pipeline, synthetic edge cases plus official kid_box image/mask; no pointmap/neural inference',
  'revision':'f91db411c50efee93d8db7aeb323885650f6f722','source_hashes':HASHES,'script_sha256':digest(Path(__file__)),'photo_sha256':PHOTO,'mask_sha256':MASK,
  'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'device':'cpu','threads':1,'repeats':3,'observer_exact':True,
  'cases':cases,'rejections':rejections,'artifacts':{}}
 manifest['artifacts']={p.name:digest(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
