#!/usr/bin/env python3
"""Original complete MHRHead: synthetic head/mapping state, real released MHR.

Run only in the reviewed container. This is composition/contract parity, not
trained SAM image-to-body parity. No injected MHR geometry in the composed run.
"""
import argparse,hashlib,importlib,json,os,struct,sys,types
from pathlib import Path
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.numpy import save_file
from capture_mhr import MODEL_SHA,MODEL_BYTES,digest
from capture_body_pose import ROMA_SHA,MHR_SHA
from capture_camera_head import GEOMETRY_SHA256
from capture_camera_encoder import HASHES
from trace_mhr_repeat import error

class MappingTrace(TorchDispatchMode):
    def __init__(self,taps,batch):super().__init__();self.taps=taps;self.batch=batch;self.calls=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        value=func(*args,**(kwargs or {}))
        if str(func) in ['aten.mm.default','aten.matmul.default'] and isinstance(value,torch.Tensor) and tuple(value.shape)==(308,self.batch*3):
            self.calls+=1;self.taps['map.05.mapping_input']=args[1].detach().clone();self.taps['map.06.keypoints_linear']=value.detach().clone()
        return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','upstream','roma-wheel','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args();root=a.upstream/'sam_3d_body'
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA or digest(a.roma_wheel)!=ROMA_SHA:raise ValueError('unexpected MHR/RoMa asset')
    hashes={f'models/modules/{k}':v for k,v in HASHES.items()};hashes|={'models/heads/mhr_head.py':'03793d51ef82484c5d9906358c6314981c0bff9dbcc6f6e6b175bd977a251b35',
        'models/modules/mhr_utils.py':MHR_SHA,'models/modules/geometry_utils.py':GEOMETRY_SHA256,'models/modules/__init__.py':'ddad96a6a9bf09d36156ba82ebaa9ee6798045fe1fc715174cec4c0b4bd2ead1',
        'models/modules/misc.py':'f942d181ee8578c67a9ab4c0828d030aefdadd8891450ab1319b33a8237891eb'}
    for name,sha in hashes.items():
        if digest(root/name)!=sha:raise ValueError('source changed '+name)
    wheel=a.roma_wheel.resolve();sys.path.insert(0,str(wheel));import roma
    if roma.__version__!='1.6.1' or not str(roma.__file__).startswith(str(wheel)+'/'):raise ValueError('unexpected RoMa import')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.heads']:
        package=types.ModuleType(name);package.__path__=[str(a.upstream/Path(*name.split('.')))];sys.modules[name]=package
    os.environ['MOMENTUM_ENABLED']='0' # Original loader selects the verified TorchScript asset.
    Head=importlib.import_module('sam_3d_body.models.heads.mhr_head').MHRHead;utils=importlib.import_module('sam_3d_body.models.modules.mhr_utils')
    source=root/'models/heads/mhr_head.py'
    import ast
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MHRHead');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='mhr_forward')
    call=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='self' and n.value.func.attr=='mhr');after=method.body[call+1].lineno
    torch.set_num_threads(1);torch.manual_seed(8610);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(8610);a.output.mkdir(parents=True,exist_ok=True);cases=[];tensors={};baselines={}
    selected=np.linspace(0,18438,16,dtype=np.int32);n=18439+127
    for index,(b,d,h,depth,initial_flag) in enumerate([(2,8,4,1,0),(1,1024,128,2,1)]):
        head=Head(d,mlp_depth=depth,mhr_model_path=str(a.model),ffn_zero_bias=False,mlp_channel_div_factor=d//h).float().to(a.device).eval()
        def random(shape,scale):return torch.from_numpy(rng.normal(scale=scale,size=shape).astype(np.float32)).to(a.device)
        with torch.no_grad():
            base=head.get_zero_pose_init().to(a.device);initial=base.repeat(b,1) if initial_flag else None
            for name,v in head.proj.named_parameters():v.copy_(random(tuple(v.shape),.02/np.sqrt(v.shape[1]) if v.ndim==2 else .005))
            if not initial_flag:head.proj.layers[-2].bias.add_(base[0])
            head.scale_mean.copy_(random((68,),.01));head.scale_comps.copy_(random((28,68),.01));head.hand_pose_mean.copy_(utils.compact_model_params_to_cont_hand(torch.zeros(1,27))[0].to(a.device));head.hand_pose_comps.copy_(torch.eye(54,device=a.device)*.1+random((54,54),.001))
            indices=rng.permutation(np.arange(68,122,dtype=np.int32));head.hand_joint_idxs_left.copy_(torch.from_numpy(indices[:27].astype(np.int64)).to(a.device));head.hand_joint_idxs_right.copy_(torch.from_numpy(indices[27:].astype(np.int64)).to(a.device))
            head.faces.copy_(head.mhr.character_torch.mesh.faces)
            mapping=np.zeros((308,n),dtype=np.float32);pool=np.concatenate([selected,18439+np.array([0,4,12,36,57,83,126])])
            for row in range(308):
                cols=rng.choice(pool,5,replace=False);weight=rng.uniform(-.2,1.,size=5).astype(np.float32);weight/=np.sum(weight);mapping[row,cols]=weight
            head.keypoint_mapping.copy_(torch.from_numpy(mapping).to(a.device));token=random((b,d),.4)
            def clone(v):return {k:x.detach().clone() for k,x in v.items() if isinstance(x,torch.Tensor)}
            head(token,initial);baseline=clone(head(token,initial));taps={}
            def keep(name,v):taps[name]=v.detach().clone()
            def trace(frame,event,value):
                if frame.f_code.co_filename!=str(source) or frame.f_code.co_name!='mhr_forward':return None
                if event=='line' and frame.f_lineno==after:
                    keep('mhr.vertices_cm',frame.f_locals['curr_skinned_verts']);keep('mhr.skeleton',frame.f_locals['curr_skel_state'])
                return trace
            def profile(frame,event,value):
                if event!='return' or frame.f_code.co_filename!=str(source):return
                l=frame.f_locals
                if frame.f_code.co_name=='mhr_forward':
                    for name,key in [('00.vertices_m','curr_skinned_verts'),('01.joints_m','curr_joint_coords'),('02.quaternions','curr_joint_quats'),('03.joint_rotations','curr_joint_rots'),('04.vertex_joints','model_vert_joints'),('07.keypoints308','model_keypoints_pred')]:keep('map.'+name,l[key])
                    keep('map.08.first70',l['model_keypoints_pred'][:,:70]);keep('pose.90.model_params',l['model_params'])
                if frame.f_code.co_name=='forward':
                    keep('pose.10.pred',l['pred'])
                    for name,key in [('24.shape','shape'),('25.scale','scale'),('26.hand','hand'),('27.face','face')]:keep('pose.'+name,value[key])
                    for name,key in [('90.vertices','pred_vertices'),('91.joints','pred_joint_coords'),('92.keypoints','pred_keypoints_3d')]:keep('map.'+name,value[key])
                    for key in ['pred_pose_raw','global_rot','body_pose']:keep(key,value[key])
            old_trace,old_profile=sys.gettrace(),sys.getprofile();observer=MappingTrace(taps,b)
            try:
                sys.settrace(trace);sys.setprofile(profile)
                with observer:observed=clone(head(token,initial))
            finally:sys.settrace(old_trace);sys.setprofile(old_profile)
            if observer.calls!=1 or len(taps)!=23:raise ValueError(f'incomplete output capture: {observer.calls}, {list(taps)}')
            equivalence={k:error(v.cpu().numpy(),observed[k].cpu().numpy()) for k,v in baseline.items()}
            if any(e['max_abs']>1e-4 or e['relative_l2']>2e-5 for e in equivalence.values()):raise ValueError('observer changed original head')
            # Upstream flag is accepted but not forwarded to the asset. Check
            # the original result explicitly rather than changing semantics.
            without_flag=clone(head(token,initial,do_pcblend=False));flag_error={k:error(v.cpu().numpy(),without_flag[k].cpu().numpy()) for k,v in baseline.items()}
            if any(e['max_abs']>1e-4 or e['relative_l2']>2e-5 for e in flag_error.values()):raise ValueError('unexpected do_pcblend semantics')
        state={'proj.'+k:v for k,v in head.proj.state_dict().items()}
        for key in ['scale_mean','scale_comps','hand_pose_mean','hand_pose_comps']:state[key]=getattr(head,key)
        filename=f'case.{index:04d}.input'
        with (a.output/filename).open('wb') as f:
            f.write(b'S3DPGO01'+struct.pack('<5I',b,d,h,depth,initial_flag)+indices.astype('<i4').tobytes())
            for v in [token,*([initial] if initial_flag else []),*[state[k] for k in sorted(state)],head.keypoint_mapping]:f.write(v.detach().cpu().numpy().astype('<f4').tobytes())
        prefix=f'case.{index:04d}'
        for name,v in taps.items():tensors[prefix+'.'+name]=v.cpu().numpy().copy()
        for name,v in baseline.items():baselines[prefix+'.'+name]=v.cpu().numpy().copy()
        cases.append({'prefix':prefix,'input':filename,'order':sorted(taps),'shapes':{k:list(v.shape) for k,v in taps.items()},'observed_vs_unobserved':equivalence,'do_pcblend_false_vs_default':flag_error})
        if index==0 and a.device=='cpu':
            # Compressed isolated mapping fixture. Synthetic mapping uses only
            # these 16 vertices (plus joints), so omitted columns are all zero.
            small_mapping=np.concatenate([mapping[:,selected],mapping[:,18439:]],axis=1)
            small={'vertices_cm':taps['mhr.vertices_cm'].cpu().numpy()[:,selected,:].copy(),'skeleton':taps['mhr.skeleton'].cpu().numpy().copy(),'mapping':small_mapping}
            expected={k.removeprefix('map.'):v.cpu().numpy().copy() for k,v in taps.items() if k.startswith('map.')}
            for key in ['00.vertices_m','90.vertices']:expected[key]=expected[key][:,selected,:].copy()
            columns=np.concatenate([selected,np.arange(18439,n)])
            expected['04.vertex_joints']=expected['04.vertex_joints'][:,columns,:].copy();expected['05.mapping_input']=expected['05.mapping_input'][columns,:].copy()
            with (a.output/'body-output.txt').open('w') as f:
                f.write('S3D_BODY_OUTPUT_V1 2 16\n')
                for group in [small,expected]:
                    f.write(str(len(group))+'\n')
                    for name,v in sorted(group.items()):f.write(f'{name} {v.size}\n'+' '.join(format(x,'.9g') for x in v.flat)+'\n')
        print(json.dumps({'case':index,'boundaries':len(taps),'observation_max_abs':max(e['max_abs'] for e in equivalence.values())}),flush=True)
        del head
    save_file(tensors,a.output/'upstream.safetensors');save_file(baselines,a.output/'full.safetensors')
    manifest={'schema_version':1,'scope':'original complete MHRHead composition with real released MHR and explicitly synthetic head/PCA/indices/keypoint mapping; NOT trained SAM Body inference',
        'model_sha256':MODEL_SHA,'source_sha256':hashes,'roma_sha256':ROMA_SHA,'script_sha256':digest(Path(__file__)),'device':a.device,'torch':torch.__version__,
        'learned_sam_checkpoint_loaded':False,'cases':cases,'artifacts':{f.name:digest(f) for f in sorted(a.output.iterdir()) if f.is_file() and f.name!='manifest.json'}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
