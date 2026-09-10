#!/usr/bin/env python3
"""Original PointPatch + Fuser consuming verified ORIGINAL preprocessing outputs.

The native comparison starts from that fixture's raw RGBA/XYZ, not these saved
intermediates. This is an explicit point-only configuration with synthetic state.
"""
import argparse,ast,json,logging,math,struct,sys
from pathlib import Path
from typing import Optional,Tuple,List,Literal,Dict
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.numpy import load_file,save_file
from original_pointpatch import load_original,digest
from capture_objects_fuser import HASHES,ObserveSilu

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['upstream','timm-wheel','preprocessing','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--full-size',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 source=json.loads((a.preprocessing/'manifest.json').read_text())
 if source['device']!='cpu' or not source['observer_exact'] or source['script_sha256']!=digest(Path(__file__).with_name('capture_objects_preprocess.py')):raise ValueError('unverified preprocessing fixture')
 Original,_=load_original(a.upstream,a.timm_wheel)
 ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'math':math,'logger':logging.getLogger('original-condition'),'Optional':Optional,'Tuple':Tuple,'List':List,'Literal':Literal,'Dict':Dict}
 for path,cls in [('model/layers/llama3/ff.py','FeedForward'),('model/backbone/dit/embedder/embedder_fuser.py','EmbedderFuser')]:
  file=a.upstream/'sam3d_objects'/path
  if digest(file)!=HASHES[path]:raise ValueError('unverified original fuser')
  nodes=[n for n in ast.parse(file.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls]
  if len(nodes)!=1:raise ValueError('missing original fuser class')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),ns)
 torch.set_num_threads(1);torch.manual_seed(61403);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rng=np.random.default_rng(61403);cases=[];invalid=[]
 specs=[(0,8,2,32,False),(2,8,4,64,True)] if not a.full_size else [(8,256,8,768,False)]
 for index,(ci,side,patch,dim,drop) in enumerate(specs):
  pc=source['cases'][ci];path=a.preprocessing/pc['full'];raw=a.preprocessing/pc['input']
  for f in [path,raw]:
   if digest(f)!=source['artifacts'][f.name]:raise ValueError('changed original preprocessing artifact')
  prepared=load_file(path);input_tensors={k:torch.from_numpy(v).to(a.device) for k,v in prepared.items()}
  encoder=Original(side,patch,dim,'exp').float().to(a.device).eval()
  net=ns['EmbedderFuser']([(encoder,[('pointmap','object'),('rgb_pointmap','scene')])],force_drop_modalities=['rgb_pointmap'] if drop else []).float().to(a.device).eval()
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   for name,value in net.state_dict().items():
    if name.endswith('weight') and value.ndim==2:arr=rng.normal(scale=1/np.sqrt(value.shape[1]),size=tuple(value.shape))
    elif name.endswith('weight'):arr=rng.uniform(.8,1.2,size=tuple(value.shape))
    else:arr=rng.normal(scale=.05,size=tuple(value.shape))
    value.copy_(torch.from_numpy(arr.astype(np.float32)).to(a.device))
   remap=2
   if ci==2:
    # Keep the incompatible original normalization/remapper combination as a
    # negative control. Forced dropout multiplies by zero; it cannot cure NaNs.
    remapped=[]
    hook=encoder.point_remapper.register_forward_hook(lambda _m,args,v:remapped.append({'safe_input_finite':bool(torch.isfinite(args[0]).all()),'remapped_nonfinite':int((~torch.isfinite(v)).sum())}))
    bad=net(**input_tensors);hook.remove()
    if torch.isfinite(bad).all() or not any(x['safe_input_finite'] and x['remapped_nonfinite'] for x in remapped):raise ValueError('expected original remapping-domain failure missing')
    invalid.append({'case':index,'first_nonfinite':'PointRemapper.forward exp/log1p','encoder_calls':remapped,'output_nonfinite':int((~torch.isfinite(bad)).sum()),'native_expected':'nonfinite PointPatch parameter/output'})
    encoder.point_remapper.remap_type='sinh';remap=1
   baseline=net(**input_tensors).clone()
   for _ in range(2):
    if not torch.equal(baseline,net(**input_tensors)):raise ValueError('nonrepeatable original point conditioner')
   prefix=f'case.{index:04d}';taps={};hooks=[];modality=[-1];active=[''];field_keys=['pointmap','rgb_pointmap']
   def emit(key,v):
    if key in taps:raise ValueError('duplicate original condition tap '+key)
    array=np.array(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v,dtype=np.float32,copy=True,order='C')
    file=prefix+'.'+key+'.safetensors';save_file({'value':array},a.output/file);taps[key]={'file':file,'shape':list(array.shape),'sha256':digest(a.output/file)}
   for key,v in prepared.items():emit('00.prepared.'+key,v)
   def start_encoder(_m,args):modality[0]+=1
   hooks.append(encoder.register_forward_pre_hook(start_encoder))
   def enc_emit(key,v):emit(f'10.encoder.{modality[0]}.'+key,v)
   # Composed stage/module boundaries. Standalone PointPatch tests retain its
   # finer SDPA/remap operation taps; this test validates own-input composition.
   block=encoder.blocks[0]
   for module,key in [(block.norm1,'11.norm1'),(block.attn.qkv,'12.qkv'),(block.attn.proj,'19.attn_projected'),(block.norm2,'21.norm2'),(block.mlp.fc1,'22.fc1'),(block.mlp.act,'23.gelu'),(block.mlp.fc2,'24.fc2'),(block,'30.block_output')]:
    hooks.append(module.register_forward_hook(lambda _m,_args,v,key=key:enc_emit(key,v)))
   hooks.append(encoder.register_forward_hook(lambda _m,_args,v:enc_emit('90.output',v)))
   hooks.append(encoder.register_forward_hook(lambda _m,_args,v:emit(f'20.fusion.10.modality.{modality[0]}.input',v)))
   seq=net.projection_nets[0]
   def start_projection(_m,args):active[0]=f'20.fusion.10.modality.{modality[0]}.projection';emit(active[0]+'.00.input',args[0])
   hooks.append(seq.register_forward_pre_hook(start_projection))
   for module,key in [(seq[0],'.01.norm'),(seq[1].w1,'.02.w1'),(seq[1].w3,'.04.w3'),(seq[1].w2,'.06.projected')]:
    hooks.append(module.register_forward_hook(lambda _m,_args,v,key=key:emit(active[0]+key,v)))
   def profile(frame,event,value):
    if frame.f_code is ns['FeedForward'].forward.__code__ and event=='return':emit(active[0]+'.05.gated',frame.f_locals['x'])
    elif frame.f_code is ns['EmbedderFuser']._apply_force_drop.__code__ and event=='call':
     for i,v in enumerate(frame.f_locals['tokens']):emit(f'20.fusion.10.modality.{i}.positioned',v)
    elif frame.f_code is ns['EmbedderFuser']._dropout_modalities.__code__ and event=='return':
     for i,v in enumerate(value):emit(f'20.fusion.10.modality.{i}.dropped',v)
   sys.setprofile(profile)
   try:
    with ObserveSilu(emit,active):observed=net(**input_tensors)
   finally:sys.setprofile(None)
   for hook in hooks:hook.remove()
   if not torch.equal(baseline,observed):raise ValueError('observation changed composed reference')
   emit('20.fusion.20.joined',observed);emit('20.fusion.90.output',observed)
   save_file({'conditioning':baseline.cpu().numpy()},a.output/(prefix+'.full.safetensors'))
   name=prefix+'.weights';state=net.state_dict()
   with (a.output/name).open('wb') as f:
    f.write(b'S3DPCW01'+struct.pack('<5I',side,patch,dim,int(drop),remap))
    for key in sorted(encoder.state_dict()):f.write(encoder.state_dict()[key].cpu().numpy().astype('<f4').tobytes())
    for key in sorted(k for k in state if not k.startswith('module_list.')):f.write(state[key].cpu().numpy().astype('<f4').tobytes())
   cases.append({'prefix':prefix,'preprocess_case':ci,'preprocess_input':pc['input'],'weights':name,'side':side,'patch':patch,'dim':dim,'remap':remap,'drop_scene':drop,'taps':taps,'full':prefix+'.full.safetensors','point_fields':field_keys})
   print(json.dumps({'case':index,'boundaries':len(taps),'observer_exact':True}),flush=True)
 meta={'scope':'native-own raw RGBA/supplied XYZ -> preprocessing -> shared PointPatch -> fuser; explicit point-only config and synthetic weights, NOT MoGe/image encoder/checkpoint-conditioned generation',
  'reference_preprocessing_device':'cpu','neural_device':a.device,'torch':torch.__version__,'threads':1,'tf32':False,'repeats':3,'observer_exact':True,'full_size':a.full_size,'script_sha256':digest(Path(__file__)),
  'preprocessing_manifest_sha256':digest(a.preprocessing/'manifest.json'),'pointpatch_loader_sha256':digest(Path(__file__).with_name('original_pointpatch.py')),'fuser_sources':HASHES,'cases':cases,'invalid_configurations':invalid,
  'artifacts':{p.name:digest(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'}}
 (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
