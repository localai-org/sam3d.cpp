#!/usr/bin/env python3
"""Unchanged original EmbedderFuser after its encoders; synthetic eval state.

Clone-only encoders expose the post-encoder boundary deliberately. This does not
claim raw image/pointmap conditioning or learned-model parity. Isolated use only.
"""
import argparse,ast,json,logging,math,struct,sys
from pathlib import Path
from typing import Optional,Tuple,List,Literal,Dict
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.numpy import save_file
from original_pointpatch import digest
HASHES={
 'model/backbone/dit/embedder/embedder_fuser.py':'a4d2ef27bddbff542550a76ad5a46facf0a653bc8dbc71fa6e92676c805d3c29',
 'model/layers/llama3/ff.py':'f3d6b9e6b60e71e00db34f9f3d6bbbf06e67e538e64a5ad796bf923af2c6e05d',
}
class ObserveSilu(TorchDispatchMode):
 def __init__(self,emit,prefix):super().__init__();self.emit=emit;self.prefix=prefix
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  result=func(*args,**(kwargs or {}))
  if func==torch.ops.aten.silu.default:self.emit(self.prefix[0]+'.03.silu',result)
  return result

class SuppliedEmbedding(torch.nn.Module):
 def __init__(self,dim):super().__init__();self.embed_dim=dim
 def forward(self,x):
  # Original fuser adds positions in place. Real encoder results are newly
  # computed on each invocation; preserve that ownership at this test boundary.
  return x.clone()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['upstream','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'math':math,'logger':logging.getLogger('original-fuser'),'Optional':Optional,'Tuple':Tuple,'List':List,'Literal':Literal,'Dict':Dict}
 for name,cls in [('model/layers/llama3/ff.py','FeedForward'),('model/backbone/dit/embedder/embedder_fuser.py','EmbedderFuser')]:
  path=a.upstream/'sam3d_objects'/name
  if digest(path)!=HASHES[name]:raise ValueError('unverified fuser source')
  nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls]
  if len(nodes)!=1:raise ValueError('missing original class')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 torch.set_num_threads(1);torch.manual_seed(61303);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rng=np.random.default_rng(61303)
 # batch, widths, (encoder,tokens,position,forced-drop), norm, random, projection, compression
 specs=[(2,[16,24],[(0,5,0,0),(0,3,1,1),(1,7,0,0)],1,0,4.,0.),
        (1,[32],[(0,3,-1,0),(0,5,0,0)],0,1,1.5,0.),
        (2,[16,16],[(0,5,0,1),(1,5,1,0)],1,0,2.,3.),
        (1,[16,24],[(0,5,-1,0),(1,5,-1,1)],0,1,0.,2.),
        (2,[16],[(0,3,-1,0),(0,2,-1,1)],0,0,0.,0.),
        (1,[32],[(0,5,0,0)],1,0,0.,0.),
        (1,[768,1024],[(0,1024,0,0),(1,1024,1,0)],1,0,4.,0.)]
 cases=[];reg=['S3D_FUSER_REGRESSION_V1','6']
 for index,(batch,dims,modalities,norm,random,projection,compression) in enumerate(specs):
  encoders=[]
  for ei,d in enumerate(dims):
   encoder=SuppliedEmbedding(d)
   encoders.append((encoder,[(f'input_{i}',None if pos<0 else f'group_{pos}') for i,(which,n,pos,drop) in enumerate(modalities) if which==ei]))
  net=ns['EmbedderFuser'](encoders,use_pos_embedding='random' if random else 'learned',projection_pre_norm=bool(norm),projection_net_hidden_dim_multiplier=projection,compression_projection_multiplier=compression,
      dropout_prob=.5,drop_modalities_weight=[(['input_0'],1.)],force_drop_modalities=[f'input_{i}' for i,x in enumerate(modalities) if x[3]]).float().to(a.device).eval()
  with torch.no_grad():
   for name,value in net.state_dict().items():
    if name.endswith('weight') and value.ndim==2:arr=rng.normal(scale=1/np.sqrt(value.shape[1]),size=tuple(value.shape))
    elif name.endswith('weight'):arr=rng.uniform(.8,1.2,size=tuple(value.shape))
    else:arr=rng.normal(scale=.05,size=tuple(value.shape))
    value.copy_(torch.from_numpy(arr.astype(np.float32)).to(a.device))
   inputs={f'input_{i}':torch.from_numpy(rng.normal(scale=.7,size=(batch,n,dims[ei])).astype(np.float32)).to(a.device) for i,(ei,n,pos,drop) in enumerate(modalities)}
   baseline=net(**inputs).clone()
   for _ in range(2):
    if not torch.equal(baseline,net(**inputs)):raise ValueError('nonrepeatable original fuser')
   prefix=f'case.{index:04d}';folder=a.output/prefix;folder.mkdir(exist_ok=True);taps={};active=[''];modality=[-1];hooks=[]
   def emit(key,v):
    if key in taps:raise ValueError('duplicate original boundary '+key)
    taps[key]=np.array(v.detach().cpu().numpy(),dtype=np.float32,copy=True,order='C')
   def encoder_end(_m,_args,v):
    modality[0]+=1;emit(f'10.modality.{modality[0]}.input',v)
   for encoder,_ in encoders:hooks.append(encoder.register_forward_hook(encoder_end))
   def instrument(seq,is_compression=False):
    def start(_m,args):
     active[0]='30.compression' if is_compression else f'10.modality.{modality[0]}.projection'
     emit(active[0]+'.00.input',args[0])
     if is_compression:emit('20.joined',args[0])
    hooks.append(seq.register_forward_pre_hook(start))
    for module,key in [(seq[0],'.01.norm'),(seq[1].w1,'.02.w1'),(seq[1].w3,'.04.w3'),(seq[1].w2,'.06.projected')]:
     hooks.append(module.register_forward_hook(lambda _m,_args,v,key=key:emit(active[0]+key,v)))
   if projection>0:
    for seq in net.projection_nets:instrument(seq)
   if compression>0:instrument(net.compression_projector,True)
   def profile(frame,event,value):
    if frame.f_code is ns['FeedForward'].forward.__code__ and event=='return':emit(active[0]+'.05.gated',frame.f_locals['x'])
    elif frame.f_code is ns['EmbedderFuser']._apply_force_drop.__code__ and event=='call':
     for i,v in enumerate(frame.f_locals['tokens']):emit(f'10.modality.{i}.positioned',v)
    elif frame.f_code is ns['EmbedderFuser']._dropout_modalities.__code__ and event=='return':
     for i,v in enumerate(value):emit(f'10.modality.{i}.dropped',v)
   sys.setprofile(profile)
   try:
    with ObserveSilu(emit,active):observed=net(**inputs)
   finally:sys.setprofile(None)
   for h in hooks:h.remove()
   if not torch.equal(baseline,observed):raise ValueError('observer changed original fuser output')
   if compression==0:emit('20.joined',observed)
   emit('90.output',observed);save_file(taps,folder/'upstream.safetensors');save_file({'output':baseline.cpu().numpy()},folder/'full.safetensors')
   state=net.state_dict();name=prefix+'.input'
   with (a.output/name).open('wb') as f:
    f.write(b'S3DFUS01'+struct.pack('<5I2d',batch,len(dims),len(modalities),norm,random,projection,compression));f.write(struct.pack('<'+'I'*len(dims),*dims))
    for m in modalities:f.write(struct.pack('<IIiI',*m))
    for v in inputs.values():f.write(v.cpu().numpy().astype('<f4').tobytes())
    for key in sorted(state):f.write(state[key].cpu().numpy().astype('<f4').tobytes())
   if index<6:
    reg+=[' '.join(map(str,[batch,len(dims),len(modalities),norm,random,projection,compression])),' '.join(map(str,dims))]+[' '.join(map(str,m)) for m in modalities]
    for v in inputs.values():reg.append(' '.join(format(float(x),'.9g') for x in v.cpu().numpy().reshape(-1)))
    for key in sorted(state):reg.append(' '.join(format(float(x),'.9g') for x in state[key].cpu().numpy().reshape(-1)))
    reg.append(str(len(taps)))
    for key in sorted(taps):
     v=taps[key].reshape(-1);reg.append(f'{key} {len(v)}')
     for start in range(0,len(v),8):reg.append(' '.join(format(float(x),'.9g') for x in v[start:start+8]))
   cases.append({'prefix':prefix,'input':name,'batch':batch,'dims':dims,'modalities':modalities,'pre_norm':bool(norm),'random_position':bool(random),'projection':projection,'compression':compression,'shapes':{k:list(v.shape) for k,v in taps.items()},'parameter_shapes':{k:list(v.shape) for k,v in state.items()},'taps_sha256':digest(folder/'upstream.safetensors'),'full_sha256':digest(folder/'full.safetensors')})
   print(json.dumps({'case':index,'taps':len(taps),'observer_exact':True}),flush=True)
 # Original rejects mixed projected widths with compression: constructor uses
 # the sum of UNprojected widths for compression, while projection equalizes them.
 (a.output/'regression.txt').write_text('\n'.join(reg)+'\n')
 manifest={'scope':'original EmbedderFuser post-encoder eval, supplied embeddings and synthetic state; NOT complete/learned conditioning','device':a.device,'torch':torch.__version__,'threads':1,'tf32':False,'repeats':3,'observer_exact':True,'max_abs':1e-4,'relative_l2':2e-5,'source_hashes':HASHES,'script_sha256':digest(Path(__file__)),'cases':cases,'artifacts':{p.name:digest(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'}}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
