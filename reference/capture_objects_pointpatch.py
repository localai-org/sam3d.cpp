#!/usr/bin/env python3
"""Original PointPatchEmbed eval + uninstrumented output, streaming tensor taps.

Synthetic state, never learned Objects parity. Full-size mode uses upstream's
default 256/8/768 shape. No SSI normalization or MoGe pointmap inference here.
"""
import argparse,ast,json,struct,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.numpy import save_file
from original_pointpatch import load_original,digest,WHEEL_SHA,TIMM_HASHES,OBJECT_HASHES

class ObserveAttention(TorchDispatchMode):
 def __init__(self,active,emit,windows,n):super().__init__();self.active=active;self.emit=emit;self.windows=windows;self.n=n;self.bmm=0
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  result=func(*args,**(kwargs or {}))
  if self.active['attention']:
   if func==torch.ops.aten.bmm.default:
    if self.bmm==0:self.emit('16.logits',result.reshape(self.windows,16,self.n,self.n))
    self.bmm+=1
   elif func==torch.ops.aten._safe_softmax.default:self.emit('17.probs',result.reshape(self.windows,16,self.n,self.n))
  return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['upstream','timm-wheel','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--full-size',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);Original,_=load_original(a.upstream,a.timm_wheel)
 torch.set_num_threads(1);torch.manual_seed(60321);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rng=np.random.default_rng(60321);modes=['linear','sinh','exp','sinh_exp','exp_disparity']
 shapes=[(2,9,13,8,2,32,2,1,0,0),(1,17,11,8,4,64,0,0,0,0),(1,19,13,16,4,64,1,0,0,0),
         (1,8,9,8,2,32,3,0,1,1),(1,11,12,8,4,32,4,0,1,0),(1,17,19,16,8,768,2,0,0,0)]
 if a.full_size:shapes=[(1,257,259,256,8,768,2,0,0,0)]
 cases=[];text=['S3D_POINTPATCH_REGRESSION_V1'];source=a.upstream/'sam3d_objects/model/backbone/dit/embedder/pointmap.py'
 tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PointPatchEmbed')
 inner=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='inner_forward')
 join_line=next(n.lineno for n in inner.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp) and isinstance(n.value.right,ast.Attribute) and n.value.right.attr=='pos_embed_window')
 for index,shape in enumerate(shapes):
  b,h,w,side,patch,d,remap,has_mask,drop,force=shape;g=side//patch;windows=b*g*g;pp=patch*patch;n=pp+1
  prefix=f'case.{index:04d}';folder=a.output/prefix;folder.mkdir(exist_ok=True)
  net=Original(side,patch,d,modes[remap],.3 if drop else 0.,bool(force)).float().to(a.device).eval()
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   for name,value in net.named_parameters():
    if name.endswith('weight') and value.ndim==2:arr=rng.normal(scale=1/np.sqrt(value.shape[1]),size=tuple(value.shape))
    elif name.endswith('weight'):arr=rng.uniform(.8,1.2,size=tuple(value.shape))
    else:arr=rng.normal(scale=.05,size=tuple(value.shape))
    value.copy_(torch.from_numpy(arr.astype(np.float32)))
   xyz=rng.normal(scale=.7,size=(b,3,h,w)).astype(np.float32);xyz[:,2]=rng.uniform(.2,3,size=(b,h,w))
   if index!=0 or a.full_size:xyz[:,0,::3,::4]=np.nan;xyz[:,2,1::7,2::5]=np.inf
   mask=np.ones((b,side,side),dtype=np.uint8) if has_mask else None
   if mask is not None:mask[:,::2,::3]=0
   x=torch.from_numpy(xyz).to(a.device);valid=None if mask is None else torch.from_numpy(mask.astype(bool)).to(a.device)
   state=net.state_dict();name=prefix+'.input'
   with (a.output/name).open('wb') as f:
    f.write(b'S3DPPT01'+struct.pack('<10I',*shape));f.write(xyz.astype('<f4').tobytes())
    if mask is not None:f.write(mask.tobytes())
    for key in sorted(state):f.write(state[key].cpu().numpy().astype('<f4').tobytes())
   print(json.dumps({'case':index,'phase':'baseline','shape':shape}),flush=True)
   baseline=net(x,valid).clone();repeat=net(x,valid)
   if not torch.equal(baseline,repeat):raise ValueError('nonrepeatable original PointPatch output')
   save_file({'output':baseline.cpu().numpy()},folder/'full.safetensors');taps={};hooks=[];active={'attention':False}
   def emit(key,value):
    if key in taps:raise ValueError('duplicate original tap '+key)
    array=value.detach().cpu().numpy().astype(np.float32,copy=False);array=np.ascontiguousarray(array)
    path=folder/(key+'.safetensors');save_file({'value':array},path)
    taps[key]={'shape':list(array.shape),'file':path.name,'sha256':digest(path)}
   def window(value):return value.reshape(b,g,patch,g,patch,-1).permute(0,1,3,2,4,5).reshape(windows,pp,-1)
   def pre(module,key):hooks.append(module.register_forward_pre_hook(lambda _m,args:emit(key,args[0])))
   def post(module,key):hooks.append(module.register_forward_hook(lambda _m,_args,v:emit(key,v)))
   hooks.append(net.point_proj.register_forward_pre_hook(lambda _m,args:emit('03.remapped',window(args[0]))))
   hooks.append(net.point_proj.register_forward_hook(lambda _m,_args,v:emit('04.projected',window(v))))
   blk=net.blocks[0];pre(blk.norm1,'10.window_tokens');post(blk.norm1,'11.norm1');post(blk.attn.qkv,'12.qkv')
   post(blk.attn.q_norm,'13.q');post(blk.attn.k_norm,'14.k')
   def attn_start(_m,_args):active['attention']=True
   def attn_end(_m,_args,v):active['attention']=False
   hooks.append(blk.attn.register_forward_pre_hook(attn_start));hooks.append(blk.attn.register_forward_hook(attn_end))
   pre(blk.attn.proj,'18.attention');post(blk.attn.proj,'19.attn_projected');pre(blk.norm2,'20.residual');post(blk.norm2,'21.norm2')
   post(blk.mlp.fc1,'22.fc1');post(blk.mlp.act,'23.gelu');post(blk.mlp.fc2,'24.fc2');post(blk,'30.block_output')
   def profile(frame,event,value):
    if frame.f_code.co_filename=='timm/models/vision_transformer.py' and frame.f_code.co_name=='forward' and isinstance(frame.f_locals.get('self'),type(blk.attn)) and event=='return':emit('15.v',frame.f_locals['v'])
    if frame.f_code.co_filename!=str(source):return
    fn=frame.f_code.co_name;l=frame.f_locals
    if fn=='apply_pointmap_dropout' and event=='call':emit('33.positioned',l['embeddings'].reshape(windows,d))
    if event!='return':return
    if fn=='resize_input':emit('00.resized',window(value))
    elif fn=='embed_pointmap_windows':emit('01.valid',window(l['valid_mask'][...,None]).squeeze(-1));emit('02.safe',window(l['xyz_safe']));emit('05.embedded',window(value[0]))
    elif fn=='inner_forward':
     emit('31.cls',l['window_embeddings'].reshape(windows,d));emit('32.position',l['pos_embed_patch'].expand(b,-1,-1).reshape(windows,d))
     if not force:emit('33.positioned',value.reshape(windows,d))
   def trace(frame,event,arg):
    if frame.f_code.co_filename==str(source) and frame.f_code.co_name=='inner_forward':
     if event=='line' and frame.f_lineno==join_line:emit('06.cls_joined',frame.f_locals['toks'])
     return trace
    return None
   print(json.dumps({'case':index,'phase':'observed'}),flush=True);observer=ObserveAttention(active,emit,windows,n)
   sys.setprofile(profile);sys.settrace(trace)
   try:
    with observer:actual=net(x,valid)
   finally:sys.setprofile(None);sys.settrace(None)
   for hook in hooks:hook.remove()
   if observer.bmm!=2 or not torch.equal(baseline,actual):raise ValueError('instrumentation changed original attention/output')
   emit('90.output',actual.reshape(windows,d))
  # Materialize no neural fixture arrays together for full-size captures.
  cases.append({'prefix':prefix,'input':name,'shape':shape,'parameter_shapes':{k:list(v.shape) for k,v in state.items()},'taps':taps,'order':sorted(taps),'full_sha256':digest(folder/'full.safetensors')})
  if index==0 and not a.full_size:
   text.append(' '.join(map(str,shape)));text.append(' '.join(format(float(v),'.9g') for v in xyz.reshape(-1)))
   if mask is not None:text.append(' '.join(map(str,mask.reshape(-1))))
   from safetensors.numpy import load_file
   for key in sorted(state):
    vals=state[key].cpu().numpy().reshape(-1);text.append(f'{key} {len(vals)}')
    for i in range(0,len(vals),8):text.append(' '.join(format(float(v),'.9g') for v in vals[i:i+8]))
   text.append(str(len(taps)))
   for key in sorted(taps):
    vals=load_file(folder/taps[key]['file'])['value'].reshape(-1);text.append(f'{key} {len(vals)}')
    for i in range(0,len(vals),8):text.append(' '.join(format(float(v),'.9g') for v in vals[i:i+8]))
  print(json.dumps({'case':index,'taps':len(taps),'observer_exact':True}),flush=True)
 if not a.full_size:(a.output/'regression.txt').write_text('\n'.join(text)+'\n')
 manifest={'scope':'original PointPatchEmbed eval with synthetic weights and supplied pointmaps; NOT SSI, MoGe or learned Objects parity',
  'device':a.device,'torch':torch.__version__,'threads':1,'sdpa_backend':'MATH','tf32':False,'observer_exact':True,'full_size':a.full_size,
  'timm_wheel_sha256':WHEEL_SHA,'timm_sources':TIMM_HASHES,'objects_sources':OBJECT_HASHES,'script_sha256':digest(Path(__file__)),'loader_sha256':digest(Path(__file__).with_name('original_pointpatch.py')),
  'max_abs_limit':1e-4,'relative_l2_limit':2e-5,'nonfinite_allowed':['00.resized','03.remapped','04.projected'],'cases':cases,'artifacts':{}}
 manifest['artifacts']={p.name:digest(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
