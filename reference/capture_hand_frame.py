#!/usr/bin/env python3
"""Original wrist-frame and hand masks with real buffers; not neural inference."""
import argparse,ast,json,struct,sys,types
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from safetensors.numpy import save_file
from capture_mhr import digest
from capture_body_pose import ROMA_SHA
STATE_SHA='4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3'
HEAD_SHA='03793d51ef82484c5d9906358c6314981c0bff9dbcc6f6e6b175bd977a251b35'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','safe-state','roma-wheel','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--synthetic-buffers',action='store_true');a=p.parse_args()
    if digest(a.safe_state)!=STATE_SHA or digest(a.roma_wheel)!=ROMA_SHA:raise ValueError('unverified reference input')
    sys.path.insert(0,str(a.roma_wheel.resolve()));import roma
    if not str(roma.__file__).startswith(str(a.roma_wheel.resolve())+'/'):raise ValueError('unexpected RoMa import')
    source=a.upstream/'sam_3d_body/models/heads/mhr_head.py'
    if digest(source)!=HEAD_SHA:raise ValueError('original head changed')
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MHRHead')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='mhr_forward')
    def hand_if(n):return isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr=='enable_hand_model'
    direct=[n for n in method.body if hand_if(n)]
    nested=[n for n in ast.walk(method) if hand_if(n) and n not in direct]
    if len(direct)!=2 or len(nested)!=1:raise ValueError('unrecognized original hand boundaries')
    blocks=[compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec') for node in [*direct,*nested]]
    holder=types.SimpleNamespace(enable_hand_model=True)
    with safe_open(a.safe_state,framework='pt') as safe:
        for name,shape,dtype in [('local_to_world_wrist',(3,3),torch.float32),('right_wrist_coords',(3,),torch.float32),('root_coords',(3,),torch.float32),('nonhand_param_idxs',(145,),torch.int64)]:
            value=safe.get_tensor('head_pose_hand.'+name)
            if value.shape!=shape or value.dtype!=dtype:raise ValueError('invalid hand buffer')
            setattr(holder,name,value.to(a.device))
    if a.synthetic_buffers:
        holder.local_to_world_wrist=torch.tensor([[0,-1,0],[1,0,0],[0,0,1]],device=a.device,dtype=torch.float32)
        holder.right_wrist_coords=torch.tensor([.4,.2,-.1],device=a.device);holder.root_coords=torch.tensor([-.1,.3,.2],device=a.device)
        holder.nonhand_param_idxs=torch.arange(6,151,device=a.device,dtype=torch.int64)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(935);inputs=[]
    inputs.append(np.array([[0,0,0],[-.4,.2,1.]],np.float32))
    inputs.append(rng.uniform(-3,3,(12,3)).astype(np.float32))
    inputs.append(np.array([[1e-8,-1e-8,0],[np.pi/2,0,-np.pi/2],[0,np.pi/2-1e-4,0],[0,-np.pi/2+1e-4,0]],np.float32))
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    arrays={};rules=[];cases=[]
    def tensor(value):return torch.from_numpy(np.asarray(value,np.float32)).to(a.device)
    for i,angles in enumerate(inputs):
        b=len(angles);rotation=tensor(angles);translation=tensor(rng.normal(0,.2,(b,3)));parameters=tensor(rng.normal(0,.3,(b,204)));keypoints=tensor(rng.normal(0,1,(b,308,3)))
        def execute(observe):
            taps={};calls=[0]
            def euler_matrix(*args,**kwargs):
                value=roma.euler_to_rotmat(*args,**kwargs)
                if observe:taps['00.input_matrix' if calls[0]==0 else '03.output_matrix']=value.detach().clone()
                calls[0]+=1;return value
            def matrix_euler(convention,matrix):
                value=roma.rotmat_to_euler(convention,matrix)
                if observe:taps['01.composed_matrix']=matrix.detach().clone();taps['02.euler']=value.detach().clone()
                return value
            proxy=types.SimpleNamespace(euler_to_rotmat=euler_matrix,rotmat_to_euler=matrix_euler)
            local=dict(self=holder,global_rot=rotation.clone(),global_trans=translation.clone(),model_params=parameters.clone(),model_keypoints_pred=keypoints.clone())
            for block in blocks:exec(block,dict(roma=proxy if observe else roma,torch=torch),local)
            taps['02.euler']=local['global_rot'].detach().clone();taps['04.translation']=local['global_trans'].detach().clone();taps['05.masked_parameters']=local['model_params'].detach().clone();taps['06.masked_keypoints']=local['model_keypoints_pred'].detach().clone()
            return taps
        with torch.inference_mode():plain=execute(False);observed=execute(True);repeat=execute(True)
        if any(not torch.equal(v,observed[k]) for k,v in plain.items()) or any(not torch.equal(v,repeat[k]) for k,v in observed.items()):raise ValueError('observation changed original result')
        prefix=f'case.{i:04d}';filename=prefix+'.input'
        with (a.output/filename).open('xb') as f:
            f.write(b'S3DHFR01'+struct.pack('<I',b))
            for v in [rotation,translation,holder.local_to_world_wrist,holder.right_wrist_coords,holder.root_coords,parameters,keypoints]:f.write(v.cpu().numpy().astype('<f4').tobytes())
            f.write(holder.nonhand_param_idxs.cpu().numpy().astype('<i4').tobytes())
        for name,value in observed.items():
            key=prefix+'.'+name;arrays[key]=value.cpu().numpy().copy()
            rules.append(dict(name=key,mode='exact') if name.startswith(('05.','06.')) else dict(name=key,mode='float',max_abs=1e-4,relative_l2=2e-5,zero_reference_floor=1e-12))
        cases.append(dict(prefix=prefix,input=filename,order=sorted(observed),shapes={k:list(v.shape) for k,v in observed.items()}))
        print(prefix,'captured',len(observed),'tensors',flush=True)
    save_file(arrays,a.output/'upstream.safetensors')
    (a.output/'rules.json').write_text(json.dumps(dict(schema_version=1,boundary=__doc__,tensors=rules),indent=2)+'\n')
    manifest=dict(scope=__doc__,oracle='unchanged original hand-specific if blocks; actual RoMa calls observed without replacement; synthetic rotations/translations/parameters/points, real checkpoint frame and mask buffers',head_sha256=HEAD_SHA,state_sha256=STATE_SHA,roma_sha256=ROMA_SHA,script_sha256=digest(Path(__file__)),device=a.device,torch=torch.__version__,repeat='all output bytes exact with/without observers',cases=cases)
    manifest['buffers']='explicit synthetic frame/geometry/indices' if a.synthetic_buffers else 'verified real checkpoint buffers'
    if a.synthetic_buffers:manifest['oracle']=manifest['oracle'].replace('real checkpoint frame and mask buffers','explicit synthetic frame and mask buffers; checkpoint buffers are overridden, not included in fixture')
    manifest['artifacts']={p.name:digest(p) for p in a.output.iterdir() if p.is_file()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
