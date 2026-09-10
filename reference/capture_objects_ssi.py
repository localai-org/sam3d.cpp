#!/usr/bin/env python3
"""Capture original SSI statistics, transforms, final normalization and own roundtrip."""
import argparse,json,struct,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import save_file
from original_objects_ssi import load_original,digest,HASHES,P3D_REV

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['upstream','pytorch3d','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
 ns=load_original(a.upstream,a.pytorch3d);a.output.mkdir(parents=True,exist_ok=True)
 # Original CUDA nanmedian returns nondeterministic tie INDICES (discarded by
 # these normalizers). Keep warnings, use its unchanged implementation, and
 # require bit-identical actual values across three runs and instrumentation.
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=a.device=='cuda');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rng=np.random.default_rng(61002);cases=[];tensors={};full={};reg=['S3D_SSI_REGRESSION_V1','15'];rejections=[]
 specs=[(m,9,13,0,0,0,0) for m in range(8)]
 specs += [(1,7,11,1,1,1,0),(1,11,9,0,1,1,0),(0,9,11,0,1,0,0),
           (4,7,13,0,1,1,0),(6,9,11,0,0,1,0),(1,7,9,0,0,0,1),(1,9,11,0,0,0,0)]
 specs += [(m,257,259,0,0,0,0) for m in range(8)]
 for index,(mode,h,w,override,has_s,has_t,all_invalid) in enumerate(specs):
  mh,mw=5,7;q,clip,factor,logshift=.1,(.85 if index==14 else 2.5),1.3,.23
  args=[h,w,mh,mw,mode,override,0,has_s,has_t];options=[q,clip,factor,logshift]
  if mode==0:net=ns['SSIPointmapNormalizer']()
  elif mode<=3:net=ns['ObjectCentricSSI'](use_scene_scale=[None,True,False,'OBJECT_NORM_MEDIAN'][mode],quantile_drop_threshold=q,clip_beyond_scale=clip,scale_factor=factor,allow_scale_and_shift_override=bool(override))
  elif mode<=5:net=ns['ObjectApparentSizeSSI'](clip_beyond_scale=clip,use_scene_scale=mode==4,scale_factor=factor)
  else:net=ns['NormalizedDisparitySpaceSSI'](clip_beyond_scale=clip,use_scene_scale=mode==6,log_disparity_shift=logshift)
  xyz=rng.normal(size=(3,h,w)).astype(np.float32);xyz[2]=rng.uniform(.15,5,size=(h,w));xyz[0,::4,::5]=np.nan;xyz[:,1::6,1::7]=np.nan
  if index==11:xyz[1,2,3]=np.inf;xyz[0,3,2]=-np.inf
  if all_invalid:xyz[:]=np.nan
  mask=rng.choice([.2,.5,.51,.8,1.],size=(1,mh,mw)).astype(np.float32)
  scale=np.array([.6,1.1,1.8],dtype=np.float32);shift=np.array([.3,-.2,1.1],dtype=np.float32)
  x=torch.from_numpy(xyz).to(a.device);m=torch.from_numpy(mask).to(a.device)
  s=torch.from_numpy(scale).to(a.device) if has_s else None;t=torch.from_numpy(shift).to(a.device) if has_t else None
  prefix=f'case.{index:04d}';input_name=prefix+'.input'
  with (a.output/input_name).open('wb') as f:
   f.write(b'S3DSSI01'+struct.pack('<9I4d',*args,*options));f.write(scale.astype('<f4').tobytes());f.write(shift.astype('<f4').tobytes());f.write(xyz.astype('<f4').tobytes());f.write(mask.astype('<f4').tobytes())
  def snapshot(out,back):return {'pointmap':out.pointmap.detach().cpu().numpy().copy(),'scale':out.scale.detach().cpu().numpy().copy(),'shift':out.shift.detach().cpu().numpy().copy(),'denormalized':back.detach().cpu().numpy().copy()}
  with torch.no_grad():
   baseline=net.normalize(x,m,s,t);back=net.denormalize(baseline.pointmap,baseline.scale,baseline.shift);expected=snapshot(baseline,back)
   for _ in range(2):
    other=net.normalize(x,m,s,t);repeat=snapshot(other,net.denormalize(other.pointmap,other.scale,other.shift))
    if any(expected[k].tobytes()!=repeat[k].tobytes() for k in expected):raise ValueError('original baseline not repeatable')
   taps={};phase='norm'
   def emit(key,value):
    if key in taps:raise ValueError('duplicate original tap '+key)
    taps[key]=np.ascontiguousarray(value.detach().cpu().numpy(),dtype=np.float32)
   def profile(frame,event,value):
    if event!='return':return
    fn=frame.f_code.co_name;l=frame.f_locals;file=frame.f_code.co_filename
    if file.endswith('/pytorch3d/transforms/transform3d.py'):
     base=10 if phase=='norm' else 30
     if fn=='_broadcast_bmm' and frame.f_back.f_code.co_name=='transform_points':emit(f'{base+2}.transformed',value.squeeze(0))
     elif fn=='transform_points':emit(f'{base}.matrix',l['composed_matrix'].squeeze(0));emit(f'{base+1}.homogeneous',l['points_batch'].squeeze(0))
    elif file.endswith('/pose_target.py') and fn=='get_scale_and_shift':
     emit('02.centered',l['shifted_pointmap'].permute(2,0,1));emit('05.computed_scale',value[0]);emit('06.computed_shift',value[1])
    elif file.endswith('/img_and_mask_transforms.py'):
     if fn in ['_compute_scale_and_shift','_get_scale_and_shift']:
      if 'mask_resized' in l:emit('01.mask',l['mask_resized'])
      if 'points_centered' in l:emit('02.centered',l['points_centered'].reshape(3,h,w))
      if 'shifted_mask_points' in l:emit('02.centered',l['shifted_mask_points'])
      if 'max_dims' in l:emit('03.statistic',l['max_dims'])
      if 'norm' in l:emit('03.statistic',l['norm'])
      if 'quantiles' in l:emit('04.quantiles',l['quantiles'])
      emit('05.computed_scale',value[0]);emit('06.computed_shift',value[1])
     elif fn=='_apply_metric_to_ssi':emit('13.unclipped' if phase=='norm' else '33.metric_space',value)
     elif fn=='normalize':
      emit('00.remapped',l.get('disparity_space_pointmap',l['pointmap']));emit('20.normalized',value.pointmap);emit('21.scale',value.scale);emit('22.shift',value.shift)
   sys.setprofile(profile)
   try:
    actual=net.normalize(x,m,s,t);phase='denorm';actual_back=net.denormalize(actual.pointmap,actual.scale,actual.shift)
   finally:sys.setprofile(None)
   emit('40.denormalized',actual_back);observed=snapshot(actual,actual_back)
   if any(expected[k].tobytes()!=observed[k].tobytes() for k in expected):raise ValueError('observer changed original output')
   for k,v in taps.items():tensors[prefix+'.'+k]=v
   for k,v in expected.items():full[prefix+'.'+k]=v
  cases.append({'prefix':prefix,'input':input_name,'args':args,'options':options,'order':sorted(taps),'shapes':{k:list(v.shape) for k,v in taps.items()}})
  if index<15:
   reg.append(' '.join(map(str,args)));reg.append(' '.join(map(str,options)))
   for arr in [scale,shift,xyz,mask]:reg.append(' '.join(format(float(v),'.9g') for v in arr.reshape(-1)))
   reg.append(str(len(taps)))
   for key in sorted(taps):
    arr=taps[key].reshape(-1);reg.append(f'{key} {len(arr)}')
    for start in range(0,len(arr),8):reg.append(' '.join(format(float(v),'.9g') for v in arr[start:start+8]))
  print(json.dumps({'case':index,'mode':mode,'shape':[h,w],'taps':len(taps),'observer_exact':True}),flush=True)
 # Record the actual original empty-mask/all-invalid behavior, not an inferred fallback.
 for name,mask_value,xyz_value in [('empty_mask',0,1.),('all_invalid',1,float('nan'))]:
  net=ns['ObjectCentricSSI'](raise_on_no_valid_points=True)
  try:net.normalize(torch.full((3,5,7),xyz_value,device=a.device),torch.full((1,5,7),mask_value,dtype=torch.float32,device=a.device))
  except (ValueError,RuntimeError) as e:rejections.append({'case':name,'type':type(e).__name__,'message':str(e)})
  else:raise ValueError('expected original rejection did not occur')
 save_file(tensors,a.output/'upstream.safetensors');save_file(full,a.output/'full.safetensors');(a.output/'regression.txt').write_text('\n'.join(reg)+'\n')
 manifest={'scope':'original SSI normalization/own denormalization on supplied pointmaps; NOT MoGe or learned Objects parity',
  'device':a.device,'torch':torch.__version__,'threads':1,'tf32':False,'deterministic_warn_only':a.device=='cuda','repeats':3,'observer_exact':True,'source_hashes':HASHES,'pytorch3d_revision':P3D_REV,
  'max_abs':1e-4,'relative_l2':2e-5,'script_sha256':digest(Path(__file__)),'loader_sha256':digest(Path(__file__).with_name('original_objects_ssi.py')),
  'cases':cases,'rejections':rejections,'artifacts':{p.name:digest(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'}}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
