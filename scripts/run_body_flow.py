#!/usr/bin/env python3
"""Compare native full Body decoder composition with original own-intermediate runs."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from run_mhr_skeleton import digest
from rotation_parity import compare_euler, POLICY as ROTATION_POLICY

def has_box_outputs(manifest,case):
    # The standalone hand decoder returns detection tokens only. Both trained
    # image branches additionally execute the shared bbox/classification heads.
    return bool(manifest.get('learned_sam_checkpoint_loaded') and
                (not manifest.get('hand_branch') or 'image_pipeline' in case))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runner','module','gguf','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backbone-input',type=Path,help='required only for RGB pipeline captures; original synthetic backbone parameter stream')
    p.add_argument('--backbone-gguf',type=Path,help='trained RGB pipeline: verified real backbone GGUF, instead of synthetic stream')
    p.add_argument('--backend',choices=['CPU','Vulkan'],default='CPU');p.add_argument('--description',default='-');p.add_argument('--device',type=int,default=0);p.add_argument('--threads',type=int,default=1);a=p.parse_args()
    manifest=json.loads((a.reference/'manifest.json').read_text());a.output.mkdir(parents=True,exist_ok=True);env=os.environ.copy()
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    before={str(path):digest(path) for path in [a.runner,a.module,a.gguf,Path(__file__),Path(__file__).with_name('rotation_parity.py')]}
    for path in [a.backbone_input,a.backbone_gguf]:
        if path:before[str(path)]=digest(path)
    if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
    for name in ['upstream.safetensors','full.safetensors']:
        if digest(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('reference hash mismatch')
    ref=load_file(a.reference/'upstream.safetensors');full=load_file(a.reference/'full.safetensors');actual={};checks=[];executions=[]
    if manifest['device']!=('cpu' if a.backend=='CPU' else 'cuda'):raise ValueError('reference/backend pairing mismatch')
    def check(name,v,r):
        if v.shape!=r.shape or v.dtype!=np.float32 or r.dtype!=np.float32 or not np.isfinite(v).all() or not np.isfinite(r).all():raise ValueError('invalid tensor '+name)
        if name.endswith('.pose.global_rot') or name.endswith('.full.global_rot'):
            checks.append({'name':name,**compare_euler(r,v)});return
        delta=v.astype(np.float64)-r.astype(np.float64);maximum=float(np.max(np.abs(delta)));relative=float(np.linalg.norm(delta)/max(np.linalg.norm(r.astype(np.float64)),1e-12))
        # Pixel/box quantities have a different unit from meters and latents.
        pixel=any(name.endswith(x) for x in ['03.vertex_pixels','12.scaled_box','19.intrinsic_projection','20.pixels','pred_keypoints_2d','pred_keypoints_2d_verts'])
        limit=1e-3 if pixel else 1e-4
        checks.append({'name':name,'max_abs':maximum,'relative_l2':relative,'max_abs_limit':limit,'pass':maximum<=limit and relative<=2e-5})
    names={'pred_pose_raw':'pose.pred_pose_raw','global_rot':'pose.global_rot','body_pose':'pose.body_pose','shape':'pose.pose.24.shape','scale':'pose.pose.25.scale','hand':'pose.pose.26.hand','face':'pose.pose.27.face',
           'pred_keypoints_3d':'pose.map.92.keypoints','pred_vertices':'pose.map.90.vertices','pred_joint_coords':'pose.map.91.joints','joint_global_rots':'pose.map.03.joint_rotations','mhr_model_params':'pose.pose.90.model_params',
           'pred_cam':'camera.10.pred_cam','pred_cam_t':'camera.15.translation','focal_length':'camera.13.focal','pred_keypoints_2d':'camera.20.pixels','pred_keypoints_2d_depth':'camera.17.depth',
           'pred_keypoints_2d_verts':'03.vertex_pixels','pred_keypoints_2d_cropped':'04.crop_points'}
    expected_full=set();expected_taps=set()
    for case in manifest['cases']:
        prefix=case['prefix'];expected_full.add(prefix+'.tokens')
        expected_full.update(prefix+f'.layer.{i}.'+key for i in range(case['depth']) for key in names)
        if has_box_outputs(manifest,case):
            expected_full.update(prefix+f'.layer.{case["depth"]-1}.'+key for key in ['hand_box','hand_logits'])
        expected_taps.update(prefix+'.'+key for key in case['order'])
    if set(full)!=expected_full or set(ref)!=expected_taps:raise ValueError('missing or untested original outputs/taps')
    for case in manifest['cases']:
        prefix,name=case['prefix'],case['input']
        if Path(prefix).name!=prefix or Path(name).name!=name or digest(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('invalid case identity')
        extra=[]
        if 'image_pipeline' in case:
            if manifest.get('learned_sam_checkpoint_loaded'):
                if a.backbone_input or not a.backbone_gguf or digest(a.backbone_gguf)!='9228c12b5b34cdb3627fce731a3e3d5890c54dc78ca7eb357d66056893f5bf1f':raise ValueError('missing/unverified trained backbone GGUF')
                if manifest['trained_state']['state_sha256']!='4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3':raise ValueError('unverified trained state')
                extra=[str(a.backbone_gguf.resolve())]
            else:
                if a.backbone_gguf or not a.backbone_input or digest(a.backbone_input)!=case['image_pipeline']['backbone_input_sha256']:raise ValueError('missing/unverified backbone input')
                extra=[str(a.backbone_input.resolve())]
        elif a.backbone_input or a.backbone_gguf:raise ValueError('backbone input supplied to non-image case')
        target=a.output/(prefix+'.bin');started=time.monotonic()
        completed=subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str(a.gguf.resolve()),str(a.reference/name),str(target),str(a.threads),*extra],env=env,check=True)
        executions.append(dict(case=prefix,returncode=completed.returncode,elapsed_seconds=time.monotonic()-started,output_sha256=digest(target)))
        flat=np.fromfile(target,dtype='<f4');offset=0
        if case['order']!=sorted(set(case['order'])):raise ValueError('invalid tap order')
        for key in case['order']:
            shape=case['shapes'][key];count=int(np.prod(shape));v=flat[offset:offset+count].reshape(shape);offset+=count;actual[prefix+'.'+key]=v.copy();check(prefix+'.'+key,v,ref[prefix+'.'+key])
        if offset!=flat.size:raise ValueError('trailing native output')
        check(prefix+'.full.tokens',actual[prefix+'.90.output_tokens'],full[prefix+'.tokens'])
        for i in range(case['depth']):
            base=prefix+f'.layer.{i}.'
            for key,native in names.items():check(base+'full.'+key,actual[base+native],full[base+key])
        if has_box_outputs(manifest,case):
            for key in ['hand_box','hand_logits']:check(prefix+'.full.'+key,actual[prefix+'.branch.'+key],full[prefix+f'.layer.{case["depth"]-1}.'+key])
        print(json.dumps({'case':prefix,'checks':len(checks),'failed':sum(not x['pass'] for x in checks)}),flush=True)
    if set(actual)!=set(ref) or any(digest(Path(path))!=sha for path,sha in before.items()):raise ValueError('incomplete native outputs or changed inference binary/model')
    save_file(actual,a.output/'native.safetensors')
    report={'scope':manifest['scope'],'reference_manifest_sha256':digest(a.reference/'manifest.json'),'gguf_sha256':digest(a.gguf),'runner_sha256':digest(a.runner),'module_sha256':digest(a.module),
        'script_sha256':digest(Path(__file__)),'rotation_policy':ROTATION_POLICY,'rotation_policy_sha256':digest(Path(__file__).with_name('rotation_parity.py')),
        'backbone_gguf_sha256':digest(a.backbone_gguf) if a.backbone_gguf else None,
        'backend':a.backend,'reference_device':manifest['device'],'threads':a.threads,'native_executions':executions,'relative_l2_limit':2e-5,'native_sha256':digest(a.output/'native.safetensors'),'checks':checks,'passed':all(x['pass'] for x in checks)}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':report['passed'],'checks':len(checks),'failures':[x for x in checks if not x['pass']][:12]},indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
