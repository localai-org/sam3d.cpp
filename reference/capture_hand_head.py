#!/usr/bin/env python3
"""Original trained hand MHRHead, synthetic input tokens, real MHR; not image inference.

Run only in the reviewed offline container. Observe unchanged original methods;
never substitute intermediate poses, geometry, masks or mapping calculations.
"""
import argparse,ast,importlib,json,os,struct,sys,types
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from capture_body_output import MappingTrace
from capture_mhr import MODEL_SHA,MODEL_BYTES,digest
from capture_body_pose import ROMA_SHA,MHR_SHA
from capture_camera_head import GEOMETRY_SHA256
from capture_camera_encoder import HASHES
from trained_body_state import STATE_SHA

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','upstream','roma-wheel','safe-state','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA or digest(a.roma_wheel)!=ROMA_SHA or digest(a.safe_state)!=STATE_SHA:
        raise ValueError('unverified model/state/rotation asset')
    root=a.upstream/'sam_3d_body';source=root/'models/heads/mhr_head.py';wheel=a.roma_wheel.resolve()
    hashes={f'models/modules/{k}':v for k,v in HASHES.items()}
    hashes|={'models/heads/mhr_head.py':'03793d51ef82484c5d9906358c6314981c0bff9dbcc6f6e6b175bd977a251b35',
             'models/modules/mhr_utils.py':MHR_SHA,'models/modules/geometry_utils.py':GEOMETRY_SHA256,
             'models/modules/__init__.py':'ddad96a6a9bf09d36156ba82ebaa9ee6798045fe1fc715174cec4c0b4bd2ead1',
             'models/modules/misc.py':'f942d181ee8578c67a9ab4c0828d030aefdadd8891450ab1319b33a8237891eb'}
    for name,sha in hashes.items():
        if digest(root/name)!=sha:raise ValueError('source changed '+name)
    sys.path.insert(0,str(wheel));import roma
    if roma.__version__!='1.6.1' or not str(roma.__file__).startswith(str(wheel)+'/'):raise ValueError('unexpected RoMa')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.heads']:
        package=types.ModuleType(name);package.__path__=[str(a.upstream/Path(*name.split('.')))];sys.modules[name]=package
    os.environ['MOMENTUM_ENABLED']='0'
    Head=importlib.import_module('sam_3d_body.models.heads.mhr_head').MHRHead
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MHRHead')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='mhr_forward')
    asset=[i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='self' and n.value.func.attr=='mhr']
    hand_if=[n for n in method.body if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr=='enable_hand_model']
    if len(asset)!=1 or len(hand_if)!=2:raise ValueError('unrecognized original head boundaries')
    asset_line=method.body[asset[0]].lineno;after_asset=method.body[asset[0]+1].lineno
    torch.set_num_threads(1);torch.manual_seed(937);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    head=Head(1024,mlp_depth=2,mhr_model_path=str(a.model),mlp_channel_div_factor=1,enable_hand_model=True).float().to(a.device).eval()
    selected={k:v for k,v in head.state_dict().items() if not k.startswith('mhr.')}
    with safe_open(a.safe_state,framework='pt') as safe,torch.no_grad():
        expected={k.removeprefix('head_pose_hand.') for k in safe.keys() if k.startswith('head_pose_hand.')}
        if set(selected)!=expected:raise ValueError('incomplete hand head state')
        for name,target in selected.items():
            value=safe.get_tensor('head_pose_hand.'+name)
            if target.shape!=value.shape or target.dtype!=value.dtype:raise ValueError('hand state layout '+name)
            target.copy_(value)
            if not torch.equal(target.cpu().contiguous().view(torch.uint8),value.contiguous().view(torch.uint8)):raise ValueError('hand state copy changed bytes')
        initial=safe.get_tensor('init_pose_hand.weight').to(a.device)
    if initial.shape!=(1,519) or not torch.isfinite(initial).all():raise ValueError('invalid initial hand estimate')
    if not torch.equal(head.faces,head.mhr.character_torch.mesh.faces):raise ValueError('hand/MHR topology mismatch')
    indices=torch.cat([head.hand_joint_idxs_left,head.hand_joint_idxs_right]).cpu().numpy().astype('<i4')
    state={'proj.'+k:v for k,v in head.proj.state_dict().items()}
    for key in ['scale_mean','scale_comps','hand_pose_mean','hand_pose_comps']:state[key]=getattr(head,key)
    rng=np.random.default_rng(937);cases=[];arrays={};full={};rules=[]
    def clone(value):return {k:v.detach().clone() for k,v in value.items() if isinstance(v,torch.Tensor)}
    def equal(left,right):return set(left)==set(right) and all(torch.equal(v.contiguous().view(torch.uint8),right[k].contiguous().view(torch.uint8)) for k,v in left.items())
    for index,b in enumerate([1,2]):
        token=torch.from_numpy(rng.normal(0,1,(b,1024)).astype(np.float32)).to(a.device);init=initial.repeat(b,1)
        with torch.inference_mode():
            head(token,init);baseline=clone(head(token,init));repeat=clone(head(token,init))
            if not equal(baseline,repeat):raise ValueError('original hand head is not byte-repeatable')
            taps={};calls=[0];euler_calls=[0];hooks=[];seen_lines=set()
            def keep(name,value):
                if name in taps:raise ValueError('duplicate observation '+name)
                taps[name]=value.detach().clone()
            for i in range(2):
                module=head.proj.layers[i] if i==1 else head.proj.layers[i][0]
                hooks.append(module.register_forward_hook(lambda _m,_a,v,i=i:keep(f'pose.00.ffn.{i}.linear',v)))
                if i==0:hooks.append(head.proj.layers[i][1].register_forward_hook(lambda _m,_a,v:keep('pose.00.ffn.0.relu',v)))
            def trace(frame,event,value):
                if frame.f_code.co_filename!=str(source) or frame.f_code.co_name!='mhr_forward':return None
                l=frame.f_locals
                # CPython revisits a multiline call's first line while setting
                # up arguments. Capture each boundary once, before execution.
                if event=='line' and frame.f_lineno in [hand_if[1].lineno,asset_line,after_asset]:
                    if frame.f_lineno in seen_lines:return trace
                    seen_lines.add(frame.f_lineno)
                if event=='line' and frame.f_lineno==hand_if[1].lineno:
                    keep('pose.hand.05.unmasked_parameters',l['model_params'])
                if event=='line' and frame.f_lineno==asset_line:
                    for name,key in [('hand.04.translation','global_trans'),('31.scales','scales'),('42.full_pose_hands','full_pose_params'),('90.model_params','model_params')]:keep('pose.'+name,l[key])
                if event=='line' and frame.f_lineno==after_asset:
                    keep('mhr.vertices_cm',l['curr_skinned_verts']);keep('mhr.skeleton',l['curr_skel_state'])
                return trace
            def profile(frame,event,value):
                name=frame.f_code.co_name;l=frame.f_locals;filename=frame.f_code.co_filename
                caller=frame.f_back.f_code.co_name if frame.f_back else ''
                if event=='call' and filename==str(source):
                    if name=='mhr_forward':keep('pose.18.global.translation',l['global_trans'])
                    if name=='replace_hands_in_pose':keep('pose.30.full_pose',l['full_pose_params'])
                if event=='call' and name=='compact_cont_to_model_params_hand' and filename==str(root/'models/modules/mhr_utils.py'):
                    keep('pose.'+('32.left.continuous' if calls[0]==0 else '33.right.continuous'),l['hand_cont']);calls[0]+=1
                if event!='return' or value is None:return
                if name=='rot6d_to_rotmat' and filename==str(root/'models/modules/geometry_utils.py'):
                    for key in ['b1','b2','b3']:keep('pose.'+{'b1':'11.global.b1','b2':'12.global.b2','b3':'13.global.b3'}[key],l[key])
                    keep('pose.14.global.matrix',value)
                if filename.startswith(str(wheel)):
                    if name=='rotmat_to_unitquat' and frame.f_back and frame.f_back.f_back and frame.f_back.f_back.f_code.co_name=='forward':
                        keep('pose.15.global.choice',l['choices'].float());keep('pose.16.global.quaternion',value)
                    if name=='rotmat_to_euler' and caller=='forward':keep('pose.17.global.euler',value)
                    if name=='rotmat_to_euler' and caller=='mhr_forward':
                        keep('pose.hand.01.composed_matrix',l['rotmat']);keep('pose.hand.02.euler',value)
                    if name=='euler_to_rotmat' and caller=='mhr_forward':
                        keep('pose.hand.'+('00.input_matrix' if euler_calls[0]==0 else '03.output_matrix'),value);euler_calls[0]+=1
                if filename==str(root/'models/modules/mhr_utils.py'):
                    if name=='batchXYZfrom6D':
                        prefix='20.body' if caller=='compact_cont_to_model_params_body' else ('40.left' if calls[0]==1 else '41.right')
                        for key in ['matrix','sy','singular']:keep('pose.'+prefix+'.'+key,l[key])
                        keep('pose.'+prefix+'.euler',value)
                    if name=='compact_cont_to_model_params_body':keep('pose.21.body.angles1',l['body_params_1dofs']);keep('pose.22.body.unmasked',value)
                    if name=='compact_cont_to_model_params_hand':
                        prefix='40.left' if calls[0]==1 else '41.right';keep('pose.'+prefix+'.angles1',l['hand_model_params_onedofs']);keep('pose.'+prefix+'.parameters',value)
                if filename==str(source):
                    if name=='mhr_forward':
                        for tap,key in [('00.vertices_m','curr_skinned_verts'),('01.joints_m','curr_joint_coords'),('02.quaternions','curr_joint_quats'),('03.joint_rotations','curr_joint_rots'),('04.vertex_joints','model_vert_joints'),('07.keypoints308','model_keypoints_pred')]:keep('map.'+tap,l[key])
                        keep('map.08.first70',l['model_keypoints_pred'][:,:70])
                    if name=='forward':
                        keep('pose.10.pred',l['pred']);keep('pose.23.body.masked',l['pred_pose_euler'])
                        for tap,key in [('24.shape','shape'),('25.scale','scale'),('26.hand','hand'),('27.face','face')]:keep('pose.'+tap,value[key])
                        for tap,key in [('90.vertices','pred_vertices'),('91.joints','pred_joint_coords'),('92.keypoints','pred_keypoints_3d')]:keep('map.'+tap,value[key])
                        for key in ['pred_pose_raw','global_rot','body_pose']:keep(key,value[key])
            old_trace,old_profile=sys.gettrace(),sys.getprofile();observer=MappingTrace(taps,b)
            try:
                sys.settrace(trace);sys.setprofile(profile)
                with observer:observed=clone(head(token,init))
            finally:
                sys.settrace(old_trace);sys.setprofile(old_profile)
                for hook in hooks:hook.remove()
            if observer.calls!=1 or calls!=[2] or euler_calls!=[2] or not equal(baseline,observed):raise ValueError('incomplete/perturbing observer')
        prefix=f'case.{index:04d}';filename=prefix+'.input'
        with (a.output/filename).open('xb') as f:
            f.write(b'S3DPGH01'+struct.pack('<5I',b,1024,1024,2,1)+indices.tobytes())
            for v in [token,init,*[state[k] for k in sorted(state)],head.keypoint_mapping,head.local_to_world_wrist,head.right_wrist_coords,head.root_coords]:f.write(v.detach().cpu().numpy().astype('<f4').tobytes())
            f.write(head.nonhand_param_idxs.detach().cpu().numpy().astype('<i4').tobytes())
        for name,v in taps.items():
            key=prefix+'.'+name;arrays[key]=v.cpu().numpy().copy()
            exact=name.endswith('.singular') or name in ['pose.15.global.choice','pose.18.global.translation','pose.27.face']
            rules.append(dict(name=key,mode='exact') if exact else dict(name=key,mode='float',max_abs=1e-4,relative_l2=2e-5,zero_reference_floor=1e-12))
        for name,v in baseline.items():full[prefix+'.'+name]=v.cpu().numpy().copy()
        full[prefix+'.faces']=head.faces.cpu().numpy().copy()
        cases.append(dict(prefix=prefix,input=filename,order=sorted(taps),shapes={k:list(v.shape) for k,v in taps.items()},repeat='all final tensor bytes exact over repeated and observed executions'))
        print(prefix,'captured',len(taps),'operation/geometry taps and',len(baseline)+1,'final fields',flush=True)
    save_file(arrays,a.output/'upstream.safetensors');save_file(full,a.output/'full.safetensors')
    (a.output/'rules.json').write_text(json.dumps(dict(schema_version=1,boundary=__doc__,tensors=rules),indent=2)+'\n')
    manifest=dict(scope=__doc__,source_sha256=hashes,model_sha256=MODEL_SHA,state_sha256=STATE_SHA,roma_sha256=ROMA_SHA,script_sha256=digest(Path(__file__)),device=a.device,torch=torch.__version__,
                  learned_head_state_loaded=sorted(selected),assignment_readback='every hand tensor byte equal safe source',tokens='synthetic standard-normal tokens, real init_pose_hand; no hand decoder/backbone yet',cases=cases)
    manifest['artifacts']={f.name:digest(f) for f in a.output.iterdir() if f.is_file()};(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
