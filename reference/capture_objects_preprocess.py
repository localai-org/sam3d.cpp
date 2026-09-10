#!/usr/bin/env python3
"""Original composed PointMap preprocessing; explicit config and supplied XYZ.

Not MoGe, learned checkpoint configuration, or full model inference. Run only
inside the reviewed isolated reference container. Original AST bodies unchanged.
"""
import argparse,ast,json,struct,sys,types
from functools import partial
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file
from capture_objects_image import original,digest,HASHES,PHOTO,MASK
from original_objects_ssi import load_original,HASHES as SSI_HASHES,P3D_REV

PIPELINE_SHA='55b69917ff0bb8b5ca6e918f516d75e9e7561a6c9396120abde863bf5757aa72'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['upstream','pytorch3d','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);root=a.upstream/'sam3d_objects'
 ns,holder=original(root);ns.update(load_original(a.upstream,a.pytorch3d))
 path=root/'data/dataset/tdfy/img_and_mask_transforms.py'
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='resize_all_to_same_size']
 if len(nodes)!=1:raise ValueError('missing resize function')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 path=root/'pipeline/inference_pipeline_pointmap.py'
 if digest(path)!=PIPELINE_SHA:raise ValueError('unverified PointMap pipeline')
 cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='InferencePipelinePointMap')
 nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_image']
 if len(nodes)!=1:raise ValueError('missing original method')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 holder.preprocess_image=types.MethodType(ns['preprocess_image'],holder)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(61203)
 specs=[(4,4,1,0),(1,1,1,0),(3,1,1,1),(0,0,0,0),(6,7,1,0),(2,1,1,0),(5,4,1,0),(1,1,0,1)]
 def normalizer(mode,override,q,clip,factor,logshift):
  if mode==0:return ns['SSIPointmapNormalizer']()
  if mode<=3:return ns['ObjectCentricSSI'](use_scene_scale=[None,True,False,'OBJECT_NORM_MEDIAN'][mode],quantile_drop_threshold=q,clip_beyond_scale=clip,scale_factor=factor,allow_scale_and_shift_override=bool(override))
  if mode<=5:return ns['ObjectApparentSizeSSI'](clip_beyond_scale=clip,use_scene_scale=mode==4,scale_factor=factor)
  return ns['NormalizedDisparitySpaceSSI'](clip_beyond_scale=clip,use_scene_scale=mode==6,log_disparity_shift=logshift)
 cases=[];reg=['S3D_PREPROCESS_REGRESSION_V1',str(len(specs))]
 for index,(om,fm,normalize,override) in enumerate(specs+[(4,4,1,0)]):
  official=index==len(specs)
  if official:
   folder=a.upstream/'notebook/images/kid_box'
   if digest(folder/'image.png')!=PHOTO or digest(folder/'0.png')!=MASK:raise ValueError('unverified official image')
   rgb=np.asarray(Image.open(folder/'image.png').convert('RGB'));mask=np.asarray(Image.open(folder/'0.png'))
   if mask.ndim==3:mask=mask[...,-1]
   rgba=np.concatenate([rgb,((mask>0)*255).astype(np.uint8)[...,None]],axis=-1)
   ph,pw,side,ps=257,259,518,256;box,pad=1.,.1
  else:
   h,w=9+index%3*2,13+index%2*2;ph,pw=(h,w) if index==3 else (5+index%2*2,7)
   side,ps=11+index,8+index%3;box,pad=(1.6,.25) if index==2 else (1.,.1)
   rgba=rng.integers(0,256,(h,w,4),dtype=np.uint8);rgba[...,3]=0
   rgba[1:-1,2:-2,3]=rng.choice([1,127,128,254,255],size=(h-2,w-4)).astype(np.uint8)
   if index==2:rgba[:4,:4,3]=255
  h,w=rgba.shape[:2];xyz=rng.normal(size=(3,ph,pw)).astype(np.float32);xyz[2]=rng.uniform(.5,5,(ph,pw));xyz[0,::4,::5]=np.nan;xyz[:,1::6,1::7]=np.nan
  nanpad=index%2;opts=[(om,0,0,.1,0.,1.3,.23),(fm,override,0,.2,0.,.9,.17)]
  nets=[normalizer(m,ov,q,clip,factor,ls) for m,ov,raise_,q,clip,factor,ls in opts]
  pre=ns['PreProcessor'](normalize_pointmap=bool(normalize),pointmap_normalizer=nets[0],rgb_pointmap_normalizer=nets[1])
  pre.img_transform=ns['Compose']([ns['pad_to_square_centered'],ns['Resize'](side,interpolation=ns['InterpolationMode'].BICUBIC)])
  pre.mask_transform=ns['Compose']([ns['pad_to_square_centered'],ns['Resize'](side,interpolation=ns['InterpolationMode'].NEAREST)])
  pre.pointmap_transform=ns['Compose']([partial(ns['pad_to_square_centered'],value=float('nan') if nanpad else 0),ns['Resize'](ps,interpolation=ns['InterpolationMode'].NEAREST)])
  pre.img_mask_pointmap_joint_transform=[ns['resize_all_to_same_size'],partial(ns['crop_around_mask_with_padding'],box_size_factor=box,padding_factor=pad),ns['rembg']]
  pre.img_mask_joint_transform=[ns['rembg']]
  def run():return holder.preprocess_image(rgba,pre,torch.from_numpy(xyz))
  baseline=run()
  def exact(other):return set(other)==set(baseline) and all(v.numpy().tobytes()==other[k].numpy().tobytes() for k,v in baseline.items())
  for _ in range(2):
   if not exact(run()):raise ValueError('nonrepeatable original result')
  prefix=f'case.{index:04d}';taps={};small={};n_norm=0;transform=0;active=False
  def keep(key,v):
   if key in taps:raise ValueError('duplicate original boundary '+key)
   if isinstance(v,torch.Tensor):v=v.detach().cpu().numpy()
   v=np.ascontiguousarray(v,dtype=np.float32);file=prefix+'.'+key+'.safetensors';save_file({'value':v},a.output/file)
   taps[key]={'file':file,'shape':list(v.shape)}
   if not official:small[key]=v.copy()
  def profile(frame,event,value):
   nonlocal n_norm,transform,active
   code=frame.f_code;l=frame.f_locals
   apply=code in (pre._apply_transform.__func__.__code__,holder._apply_transform.__func__.__code__)
   if event=='call' and apply:active=True
   if event!='return':return
   if code is ns['image_to_float'].__code__:keep('00.rgb',value[...,:3].transpose(2,0,1));keep('00.mask',value[...,3][None])
   elif code is pre._normalize_pointmap.__func__.__code__:
    pref='01.object.' if n_norm==0 else '05.full.';n_norm+=1
    for key,v in zip(['pointmap','scale','shift'],value):keep(pref+key,v)
   elif code is pre._preprocess_rgb_image_mask.__func__.__code__:keep('02.full_before.rgb',value[0]);keep('02.full_before.mask',value[1])
   elif code is ns['resize_all_to_same_size'].__code__:
    for key,v in zip(['02.aligned_rgb','03.aligned_mask','04.aligned_pointmap'],value):keep('03.'+key,v)
    if 'nan_mask' in l:
     for key,name in [('05.nan_mask','nan_mask'),('06.clean','pointmap_clean'),('07.resized','pointmap_resized'),('08.resized_nan_mask','nan_mask_resized')]:keep('03.'+key,l[name])
   elif code is ns['compute_mask_bbox'].__code__:keep('03.09.bbox',value)
   elif code is ns['crop_around_mask_with_padding'].__code__:
    for key,v in zip(['10.crop_rgb','11.crop_mask','12.crop_pointmap'],value):keep('03.'+key,v)
   elif code is ns['rembg'].__code__:
    for key,v in zip(['20.rgb','21.mask','22.pointmap'],value):keep('03.'+key,v)
   elif code is ns['pad_to_square_centered'].__code__ and active:keep(f'04.apply.{transform}.pad',value)
   elif apply:keep(f'04.apply.{transform}.output',value);transform+=1;active=False
  sys.setprofile(profile)
  try:observed=run()
  finally:sys.setprofile(None)
  if transform!=7 or n_norm!=2 or not exact(observed):raise ValueError('observer changed original result')
  for key,v in observed.items():keep('06.return.'+key,v)
  save_file({k:np.ascontiguousarray(v.numpy()) for k,v in baseline.items()},a.output/(prefix+'.full.safetensors'))
  stride=w*4+3;raw=np.full((h,stride),181,dtype=np.uint8);raw[:,:w*4]=rgba.reshape(h,w*4)
  args=[w,h,stride,ph,pw,side,ps,normalize,nanpad];name=prefix+'.input'
  with (a.output/name).open('wb') as f:
   f.write(b'S3DOPP01'+struct.pack('<9I2d',*args,box,pad))
   for opt in opts:f.write(struct.pack('<3I4d',*opt))
   f.write(raw.tobytes()+xyz.astype('<f4').tobytes())
  if not official:
   reg+=[' '.join(map(str,args)),f'{box} {pad}']+[' '.join(map(str,opt)) for opt in opts]
   reg+=[' '.join(map(str,raw.reshape(-1))),' '.join(format(float(v),'.9g') for v in xyz.reshape(-1)),str(len(small))]
   for key in sorted(small):
    v=small[key].reshape(-1);reg.append(f'{key} {len(v)}')
    for start in range(0,len(v),8):reg.append(' '.join(format(float(x),'.9g') for x in v[start:start+8]))
  cases.append({'prefix':prefix,'input':name,'taps':taps,'full':prefix+'.full.safetensors','official_image':official,'args':args,'normalizers':opts,'crop':[box,pad]})
  print(json.dumps({'case':index,'boundaries':len(taps),'observer_exact':True}),flush=True)
 (a.output/'regression.txt').write_text('\n'.join(reg)+'\n')
 meta={'scope':'original complete PointMap preprocess_image method on CPU; explicit transform config, supplied synthetic XYZ; NOT MoGe/checkpoint configuration/learned model parity',
  'device':'cpu','torch':torch.__version__,'threads':1,'repeats':3,'observer_exact':True,'max_abs':1e-4,'relative_l2':2e-5,
  'source_hashes':HASHES,'ssi_source_hashes':SSI_HASHES,'pipeline_sha256':PIPELINE_SHA,'pytorch3d_revision':P3D_REV,'photo_sha256':PHOTO,'mask_sha256':MASK,
  'script_sha256':digest(Path(__file__)),'loaders':{n:digest(Path(__file__).with_name(n)) for n in ['capture_objects_image.py','original_objects_ssi.py']},'cases':cases,
  'artifacts':{p.name:digest(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'}}
 (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
