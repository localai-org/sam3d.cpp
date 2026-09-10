#!/usr/bin/env python3
"""Original MHRHead prefix through MHR inputs, stopping before unavailable geometry."""
import argparse,ast,hashlib,importlib,json,struct,sys,types,zipfile
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors import safe_open
from capture_camera_head import GEOMETRY_SHA256
from capture_camera_encoder import HASHES

ROMA_SHA='79c3a07ab94c0e784e8de2ecc878645a5cf0aff312f7b1e281ca1be7c5f0b7e8'
MHR_SHA='5d95cca73aead5c783b29962b4bb2d930bd84a8d90f78627b534723cb1277e0c'

class Boundary(Exception):
    def __init__(self,values):self.values=values

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--roma-wheel',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--small-regression',action='store_true')
    p.add_argument('--trained-state',type=Path);p.add_argument('--trained-reference',type=Path);p.add_argument('--trained-candidate',type=Path);args=p.parse_args()
    diagnostic=any([args.trained_state,args.trained_reference,args.trained_candidate]);diagnostic_inputs=None
    if diagnostic:
        if not all([args.trained_state,args.trained_reference,args.trained_candidate]) or args.small_regression or args.device!='cpu':raise ValueError('trained diagnostic requires all three paths, CPU and no synthetic regression')
        from trained_body_state import STATE_SHA
        def digest(path):
            with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
        if digest(args.trained_state)!=STATE_SHA:raise ValueError('unverified safe trained state')
        reference_manifest=json.loads((args.trained_reference/'manifest.json').read_text());candidate_report=json.loads((args.trained_candidate/'report.json').read_text())
        candidate_parity=json.loads((args.trained_candidate/'parity-trained-v1.json').read_text())
        if candidate_parity['native_capture_report_sha256']!=digest(args.trained_candidate/'report.json'):raise ValueError('candidate report identity mismatch')
        if reference_manifest['device']!='cpu' or not reference_manifest['learned_sam_checkpoint_loaded'] or candidate_report['reference_manifest_sha256']!=digest(args.trained_reference/'manifest.json'):raise ValueError('unmatched trained CPU captures')
        diagnostic_inputs={};provenance={}
        for label,directory,filename,expected in [('reference',args.trained_reference,'upstream.safetensors',reference_manifest['artifacts']['upstream.safetensors']),('candidate',args.trained_candidate,'native.safetensors',candidate_parity['native_safetensors_sha256'])]:
            path=directory/filename
            if digest(path)!=expected:raise ValueError('unverified trained '+label)
            provenance[label]=expected
            with safe_open(path,framework='pt') as safe:
                for i in range(6):
                    value=safe.get_tensor(f'case.0000.layer.{i}.02.normalized')
                    if value.shape!=(1,145,1024) or value.dtype!=torch.float32 or not torch.isfinite(value).all():raise ValueError('invalid normalized token')
                    diagnostic_inputs[label,i]=value[:,0,:].clone()
    root=args.upstream.resolve()/'sam_3d_body';wheel=args.roma_wheel.resolve()
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=ROMA_SHA:raise ValueError('unexpected RoMa wheel')
    sys.path.insert(0,str(wheel));import roma
    if roma.__version__!='1.6.1' or not str(roma.__file__).startswith(str(wheel)+'/'):raise ValueError('unexpected RoMa import')
    hashes={f'models/modules/{k}':v for k,v in HASHES.items()}
    hashes|={'models/heads/mhr_head.py':'03793d51ef82484c5d9906358c6314981c0bff9dbcc6f6e6b175bd977a251b35',
        'models/modules/mhr_utils.py':MHR_SHA,'models/modules/geometry_utils.py':GEOMETRY_SHA256,
        'models/modules/__init__.py':'ddad96a6a9bf09d36156ba82ebaa9ee6798045fe1fc715174cec4c0b4bd2ead1',
        'models/modules/misc.py':'f942d181ee8578c67a9ab4c0828d030aefdadd8891450ab1319b33a8237891eb'}
    for name,digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'source changed: {name}')
    for name in ['sam_3d_body','sam_3d_body.models']:
        package=types.ModuleType(name);package.__path__=[str(root.parent/Path(*name.split('.')))];sys.modules[name]=package
    modules=importlib.import_module('sam_3d_body.models.modules');utils=importlib.import_module('sam_3d_body.models.modules.mhr_utils')
    FFN=importlib.import_module('sam_3d_body.models.modules.transformer').FFN
    source=root/'models/heads/mhr_head.py';tree=ast.parse(source.read_text(),filename=str(source));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MHRHead')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['forward','mhr_forward','replace_hands_in_pose']]
    namespace={'torch':torch,'Optional':Optional,'roma':roma,'rot6d_to_rotmat':modules.rot6d_to_rotmat,
        'compact_cont_to_model_params_body':utils.compact_cont_to_model_params_body,'compact_cont_to_model_params_hand':utils.compact_cont_to_model_params_hand,'mhr_param_hand_mask':utils.mhr_param_hand_mask}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    mhr=next(n for n in nodes if n.name=='mhr_forward')
    stops=[n.lineno for n in ast.walk(mhr) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='self' and n.value.func.attr=='mhr']
    if len(stops)!=1 or len(nodes)!=3:raise ValueError('unrecognized MHR boundary')
    torch.set_num_threads(1);torch.manual_seed(8606);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(8606)
    # B,D,hidden,depth,initial,special rotations. All learnable state is synthetic.
    shapes=[(2,8,4,1,0,0),(1,8,4,2,1,0),(2,8,4,3,1,0),(2,4,2,1,0,1),(2,4,2,1,0,2),(2,4,2,1,0,3)]
    if not args.small_regression:shapes.extend([(1,1024,128,1,1,0),(1,1024,128,2,1,0)])
    if diagnostic:shapes=[(1,1024,1024,2,1,0)]*12
    args.output.mkdir(parents=True,exist_ok=True);cases,tensors,rules=[],{},[];lines=['S3D_POSE_REGRESSION_V1',str(len(shapes))]
    def capture(taps,name,v):taps[name]=v.detach().clone()
    for index,(b,d,h,depth,initial_flag,special) in enumerate(shapes):
        holder=torch.nn.Module();holder.body_cont_dim=260;holder.num_shape_comps=45;holder.num_scale_comps=28;holder.num_hand_comps=54;holder.num_face_comps=72;holder.num_hand_pose_comps=54;holder.enable_hand_model=False
        holder.proj=FFN(embed_dims=d,feedforward_channels=h,output_dims=519,num_fcs=depth,add_identity=False).float().to(args.device).eval()
        for node in nodes:setattr(holder,node.name,types.MethodType(namespace[node.name],holder))
        def random(shape,scale=1):return torch.from_numpy(rng.normal(scale=scale,size=shape).astype(np.float32)).to(args.device)
        with torch.no_grad():
            for name,v in holder.named_parameters():v.copy_(random(tuple(v.shape),.2/np.sqrt(v.shape[1]) if v.ndim==2 else .1))
            # Permuted explicit index buffers; true checkpoint mappings must be loaded.
            idx=rng.permutation(np.arange(68,122,dtype=np.int32));holder.hand_joint_idxs_left=torch.from_numpy(idx[:27].astype(np.int64)).to(args.device);holder.hand_joint_idxs_right=torch.from_numpy(idx[27:].astype(np.int64)).to(args.device)
            holder.scale_mean=random((68,),.1);holder.scale_comps=random((28,68),.1);holder.hand_pose_mean=random((54,),.1);holder.hand_pose_comps=random((54,54),.1)
            token=random((b,d));initial=random((b,519),.1) if initial_flag else None
            if diagnostic:
                with safe_open(args.trained_state,framework='pt') as safe:
                    for name,target in holder.proj.state_dict().items():
                        value=safe.get_tensor('head_pose.proj.'+name)
                        if value.shape!=target.shape or value.dtype!=target.dtype:raise ValueError('trained pose state mismatch')
                        target.copy_(value)
                        if not torch.equal(target.view(torch.uint8),value.view(torch.uint8)):raise ValueError('pose state readback differs')
                    for name in ['scale_mean','scale_comps','hand_pose_mean','hand_pose_comps','hand_joint_idxs_left','hand_joint_idxs_right']:
                        value=safe.get_tensor('head_pose.'+name)
                        if value.shape!=getattr(holder,name).shape or value.dtype!=getattr(holder,name).dtype:raise ValueError('trained pose buffer mismatch')
                        setattr(holder,name,value.clone())
                    initial=safe.get_tensor('init_pose.weight').clone()
                idx=torch.cat([holder.hand_joint_idxs_left,holder.hand_joint_idxs_right]).numpy().astype('<i4')
                token=diagnostic_inputs['reference' if index<6 else 'candidate',index%6]
            if special:
                for v in holder.proj.parameters():v.zero_()
                bias=holder.proj.layers[-2].bias
                rotations={1:[1,0,0,0,1,0],2:[0,0,-1,0,1,0],3:[-1,0,0,0,-1,0]}
                bias[:6]=torch.tensor(rotations[special],device=args.device)
                # Exact zero, identity, collinear, tiny and gimbal-lock 6D vectors.
                options=[[0,0,0,0,0,0],[1,0,0,0,1,0],[1,0,0,2,0,0],[1e-14,0,0,0,1e-14,0],[0,0,-1,0,1,0],[0,0,1,0,1,0]]
                for k in range(23):bias[6+k*6:12+k*6]=torch.tensor(options[k%len(options)],device=args.device)
                bias[6+138:6+254]=torch.tensor([0.,1.]*58,device=args.device)
                bias[266:]=random((253,),.1)
            def run(taps=None):
                hand_call=[0];hooks=[]
                if taps is not None:
                    for i in range(depth):
                        module=holder.proj.layers[i] if i+1==depth else holder.proj.layers[i][0]
                        hooks.append(module.register_forward_hook(lambda _m,_a,v,i=i:capture(taps,f'00.ffn.{i}.linear',v)))
                        if i+1<depth:hooks.append(holder.proj.layers[i][1].register_forward_hook(lambda _m,_a,v,i=i:capture(taps,f'00.ffn.{i}.relu',v)))
                def profile(frame,event,value):
                    name=frame.f_code.co_name;local=frame.f_locals;filename=frame.f_code.co_filename
                    if event=='call' and name=='replace_hands_in_pose' and filename==str(source):capture(taps,'30.full_pose',local['full_pose_params'])
                    if event=='call' and name=='compact_cont_to_model_params_hand' and filename==str(root/'models/modules/mhr_utils.py'):
                        prefix='32.left.continuous' if hand_call[0]==0 else '33.right.continuous';capture(taps,prefix,local['hand_cont']);hand_call[0]+=1
                    if event!='return':return
                    if name=='rot6d_to_rotmat' and filename==str(root/'models/modules/geometry_utils.py'):
                        for key in ['b1','b2','b3']:capture(taps,{'b1':'11.global.b1','b2':'12.global.b2','b3':'13.global.b3'}[key],local[key])
                        capture(taps,'14.global.matrix',value)
                    if name=='rotmat_to_unitquat' and filename.startswith(str(wheel)):
                        capture(taps,'15.global.choice',local['choices'].float());capture(taps,'16.global.quaternion',value)
                    if name=='rotmat_to_euler' and filename.startswith(str(wheel)):capture(taps,'17.global.euler',value)
                    if filename!=str(root/'models/modules/mhr_utils.py'):return
                    if name=='batchXYZfrom6D':
                        caller=frame.f_back.f_code.co_name;prefix='20.body' if caller=='compact_cont_to_model_params_body' else ('40.left' if hand_call[0]==1 else '41.right')
                        for key in ['matrix','sy','singular']:capture(taps,prefix+'.'+key,local[key])
                        capture(taps,prefix+'.euler',value)
                    if name=='compact_cont_to_model_params_body':
                        capture(taps,'21.body.angles1',local['body_params_1dofs']);capture(taps,'22.body.unmasked',value)
                    if name=='compact_cont_to_model_params_hand':
                        prefix='40.left' if hand_call[0]==1 else '41.right';capture(taps,prefix+'.angles1',local['hand_model_params_onedofs']);capture(taps,prefix+'.parameters',value)
                def trace(frame,event,value):
                    if frame.f_code.co_filename!=str(source) or frame.f_code.co_name!='mhr_forward':return None
                    if event=='line' and frame.f_lineno==stops[0]:
                        l=frame.f_locals;head=frame.f_back.f_locals
                        values={'10.pred':head['pred'],'18.global.translation':l['global_trans'],'23.body.masked':head['pred_pose_euler'],
                            '24.shape':l['shape_params'],'25.scale':l['scale_params'],'26.hand':l['hand_pose_params'],'27.face':l['expr_params'],
                            '31.scales':l['scales'],'42.full_pose_hands':l['full_pose_params'],'90.model_params':l['model_params']}
                        raise Boundary({k:v.detach().clone() for k,v in values.items()})
                    return trace
                old_trace,old_profile=sys.gettrace(),sys.getprofile()
                try:
                    sys.settrace(trace)
                    if taps is not None:sys.setprofile(profile)
                    holder.forward(token,initial)
                    raise ValueError('failed to stop before MHR geometry')
                except Boundary as boundary:return boundary.values
                finally:
                    sys.settrace(old_trace);sys.setprofile(old_profile)
                    for hook in hooks:hook.remove()
            baseline=run();taps={};observed=run(taps)
            if not all(torch.equal(v,observed[k]) for k,v in baseline.items()):raise ValueError('instrumentation changed MHR inputs')
            taps.update(observed)
        state={**holder.proj.state_dict()};state={'proj.'+k:v for k,v in state.items()}
        for name in ['scale_mean','scale_comps','hand_pose_mean','hand_pose_comps']:state[name]=getattr(holder,name)
        inputs={'token':token,'initial':initial if initial is not None else torch.empty(0,device=args.device)}
        prefix=f'case.{index:04d}';filename=prefix+'.input'
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DPSE01'+struct.pack('<5I',b,d,h,depth,initial_flag)+idx.astype('<i4').tobytes())
            for v in [inputs['token'],inputs['initial'],*[state[k] for k in sorted(state)]]:stream.write(v.cpu().numpy().astype('<f4').tobytes())
        if args.small_regression:
            lines.append(f'{b} {d} {h} {depth} {initial_flag}');lines.append(' '.join(map(str,idx)))
            for group in [inputs,state,taps]:
                lines.append(str(len(group)))
                for name,v in sorted(group.items()):
                    a=v.cpu().numpy().reshape(-1);lines.append(f'{name} {a.size}')
                    for start in range(0,len(a),8):lines.append(' '.join(format(float(v),'.9g') for v in a[start:start+8]))
        order=sorted(taps);cases.append({'prefix':prefix,'input':filename,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()}})
        for name in order:
            key=prefix+'.'+name;tensors[key]=taps[name].cpu().numpy().copy()
            exact=name.endswith('.singular') or name in ['15.global.choice','18.global.translation','27.face']
            rules.append({'name':key,'mode':'exact'} if exact else {'name':key,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {prefix}, B={b}, D={d}, depth={depth}, special={special}, taps={len(taps)}',flush=True)
    if args.small_regression:(args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='unchanged MHRHead method prefix, original compact rotations and RoMa; stops before self.mhr; synthetic state, not geometry or trained-model parity'
    if diagnostic:boundary='diagnostic intervention: original trained pose head on reference/native normalized tokens; NOT end-to-end acceptance'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    with zipfile.ZipFile(wheel) as z:roma_hashes={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith('.py')}
    manifest={'boundary':boundary,'revision':'b5c765a0d89d789985e186d396315e7590887b94','source_hashes':hashes,'roma_wheel_sha256':ROMA_SHA,'roma_source_hashes':roma_hashes,
        'selected_methods':[n.name for n in nodes],'method_changes':'none; original method ASTs, stop on line before self.mhr call; no geometry substitution',
        'holder':'synthetic source-defined scalar configuration, FFN and data buffers; no asset-loading constructor',
        'capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch':torch.__version__,'numpy':np.__version__,'roma':roma.__version__,
        'device':args.device,'threads':1,'tf32_matmul':False,'tf32_cudnn':False,'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'unobserved_vs_observed':'exact_equal_all_MHR_inputs','cases':cases,'artifacts':{}}
    if diagnostic:
        manifest['holder']='source-defined configuration with all pose head weights/buffers loaded and checked from verified safe state'
        manifest['diagnostic_inputs']=dict(provenance,state_sha256=STATE_SHA,case_order='reference layers 0..5, candidate layers 0..5')
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json':manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
