#!/usr/bin/env python3
"""Original pointmap-aware resize/crop/rembg chain, supplied XYZ (not MoGe)."""
import argparse,ast,json,struct,sys
from functools import partial
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file
from capture_objects_image import original,digest,HASHES,PHOTO,MASK

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['upstream','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);root=a.upstream/'sam3d_objects';ns,holder=original(root)
 path=root/'data/dataset/tdfy/img_and_mask_transforms.py';tree=ast.parse(path.read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resize_all_to_same_size']
 if len(nodes)!=1:raise ValueError('missing original resize function')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(61103)
 shapes=[(9,13,5,7,1.,.1),(13,9,17,11,1.6,.1),(11,17,11,17,.65,.25),
         (12,12,12,7,1.,0.),(13,17,5,17,1.,.1),(7,19,3,5,2.,.25)]
 inputs=[]
 for index,(h,w,ph,pw,factor,padding) in enumerate(shapes):
  rgba=rng.integers(0,256,(h,w,4),dtype=np.uint8);rgba[...,3]=0
  rgba[1:-1,2:-2,3]=rng.choice([1,127,128,254,255],size=(h-2,w-4)).astype(np.uint8)
  if index in [1,5]:rgba[:5,:5,3]=127
  xyz=rng.normal(size=(3,ph,pw)).astype(np.float32);xyz[2]+=3
  xyz[0,::3,::4]=np.nan;xyz[:,1::5,1::3]=np.nan
  if index==5:xyz[1,1,2]=np.inf # NaN clearing must not silently clear infinity.
  inputs.append((rgba,xyz,factor,padding))
 folder=a.upstream/'notebook/images/kid_box'
 if digest(folder/'image.png')!=PHOTO or digest(folder/'0.png')!=MASK:raise ValueError('unverified official image/mask')
 rgb=np.asarray(Image.open(folder/'image.png').convert('RGB'));mask=np.asarray(Image.open(folder/'0.png'))
 if mask.ndim==3:mask=mask[...,-1]
 rgba=np.concatenate([rgb,((mask>0)*255).astype(np.uint8)[...,None]],axis=-1)
 xyz=rng.normal(size=(3,257,259)).astype(np.float32);xyz[2]+=3;xyz[:,::11,::7]=np.nan
 inputs.append((rgba,xyz,1.,.1));tensors={};full={};cases=[];lines=['S3D_JOINT_REGRESSION_V1','6']
 for index,(rgba,xyz,factor,padding) in enumerate(inputs):
  h,w=rgba.shape[:2];ph,pw=xyz.shape[1:];pre=ns['PreProcessor']()
  pre.img_mask_pointmap_joint_transform=[ns['resize_all_to_same_size'],partial(ns['crop_around_mask_with_padding'],box_size_factor=factor,padding_factor=padding),ns['rembg']]
  pre.img_mask_joint_transform=[ns['rembg']] # Original triple path must take priority.
  def run():
   converted=torch.from_numpy(holder.image_to_float(rgba)).permute(2,0,1).contiguous()
   m=ns['get_mask'](converted,None,'ALPHA_CHANNEL')
   return pre._preprocess_image_mask_pointmap(converted[:3],m,torch.from_numpy(xyz))
  baseline=run()
  for _ in range(2):
   repeat=run()
   if any(x.numpy().tobytes()!=y.numpy().tobytes() for x,y in zip(baseline,repeat)):raise ValueError('nonrepeatable original joint transforms')
  taps={}
  def keep(key,value):
   if key in taps:raise ValueError('duplicate original joint boundary')
   if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
   taps[key]=np.ascontiguousarray(value,dtype=np.float32)
  def profile(frame,event,value):
   if event!='return':return
   code=frame.f_code;l=frame.f_locals
   if code is ns['image_to_float'].__code__:keep('00.rgba',value.transpose(2,0,1))
   elif code is ns['get_mask'].__code__:keep('01.alpha',value)
   elif code is ns['resize_all_to_same_size'].__code__:
    for key,v in zip(['02.aligned_rgb','03.aligned_mask','04.aligned_pointmap'],value):keep(key,v)
    if 'nan_mask' in l:
     for key,name in [('05.nan_mask','nan_mask'),('06.clean','pointmap_clean'),('07.resized','pointmap_resized'),('08.resized_nan_mask','nan_mask_resized')]:keep(key,l[name])
   elif code is ns['compute_mask_bbox'].__code__:keep('09.bbox',value)
   elif code is ns['crop_around_mask_with_padding'].__code__:
    for key,v in zip(['10.crop_rgb','11.crop_mask','12.crop_pointmap'],value):keep(key,v)
   elif code is ns['rembg'].__code__:
    for key,v in zip(['20.rgb','21.mask','22.pointmap'],value):keep(key,v)
  sys.setprofile(profile)
  try:observed=run()
  finally:sys.setprofile(None)
  if any(x.numpy().tobytes()!=y.numpy().tobytes() for x,y in zip(baseline,observed)):raise ValueError('observer changed original result')
  prefix=f'case.{index:04d}';name=prefix+'.input';stride=w*4+3;raw=np.full((h,stride),181,dtype=np.uint8);raw[:,:w*4]=rgba.reshape(h,w*4)
  with (a.output/name).open('wb') as f:f.write(b'S3DPJM01'+struct.pack('<5I2d',w,h,ph,pw,stride,factor,padding)+raw.tobytes()+xyz.astype('<f4').tobytes())
  for k,v in taps.items():tensors[prefix+'.'+k]=v
  for k,v in zip(['rgb','mask','pointmap'],baseline):full[prefix+'.'+k]=np.ascontiguousarray(v.numpy())
  cases.append({'prefix':prefix,'input':name,'args':[w,h,ph,pw,stride],'options':[factor,padding],'order':sorted(taps),'shapes':{k:list(v.shape) for k,v in taps.items()},'official_image':index==6,'pointmap':'synthetic supplied XYZ'})
  if index<6:
   lines.append(f'{w} {h} {ph} {pw} {stride} {factor} {padding}');lines.append(' '.join(map(str,raw.reshape(-1))));lines.append(' '.join(format(float(v),'.9g') for v in xyz.reshape(-1)));lines.append(str(len(taps)))
   for key in sorted(taps):
    vals=taps[key].reshape(-1);lines.append(f'{key} {len(vals)}')
    for start in range(0,len(vals),8):lines.append(' '.join(format(float(v),'.9g') for v in vals[start:start+8]))
  print(json.dumps({'case':index,'taps':len(taps),'observer_exact':True}),flush=True)
 save_file(tensors,a.output/'upstream.safetensors');save_file(full,a.output/'full.safetensors');(a.output/'regression.txt').write_text('\n'.join(lines)+'\n')
 meta={'scope':'original explicit resize/crop/rembg joint transform chain, supplied synthetic XYZ; NOT SSI/MoGe/learned pipeline parity or verified checkpoint transform selection',
  'device':'cpu','torch':torch.__version__,'threads':1,'repeats':3,'observer_exact':True,'source_hashes':HASHES,'image_sha256':PHOTO,'mask_sha256':MASK,
  'script_sha256':digest(Path(__file__)),'loader_sha256':digest(Path(__file__).with_name('capture_objects_image.py')),'max_abs':1e-5,'relative_l2':2e-5,'cases':cases,
  'artifacts':{p.name:digest(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'}}
 (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
