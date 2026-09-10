#!/usr/bin/env python3
"""Run native pose->real MHR->Body mapping against original full-head captures."""
import argparse,json,os,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from run_mhr_skeleton import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runner','module','gguf','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backend',choices=['CPU','Vulkan'],default='CPU');p.add_argument('--description',default='-');p.add_argument('--device',type=int,default=0);p.add_argument('--threads',type=int,default=1);a=p.parse_args()
    manifest=json.loads((a.reference/'manifest.json').read_text());a.output.mkdir(parents=True,exist_ok=True);env=os.environ.copy()
    if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
    for name in ['upstream.safetensors','full.safetensors']:
        if digest(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('reference hash mismatch')
    ref=load_file(a.reference/'upstream.safetensors');full=load_file(a.reference/'full.safetensors');actual={};checks=[]
    def check(name,v,r):
        if v.shape!=r.shape or v.dtype!=np.float32 or r.dtype!=np.float32 or not np.isfinite(v).all() or not np.isfinite(r).all():raise ValueError('invalid tensor shape/dtype/value')
        delta=v.astype(np.float64)-r.astype(np.float64);maximum=float(np.max(np.abs(delta)));relative=float(np.linalg.norm(delta)/max(np.linalg.norm(r.astype(np.float64)),1e-12))
        checks.append({'name':name,'max_abs':maximum,'relative_l2':relative,'pass':maximum<=1e-4 and relative<=2e-5})
    output_names={'pred_pose_raw':'pred_pose_raw','global_rot':'global_rot','body_pose':'body_pose','shape':'pose.24.shape','scale':'pose.25.scale','hand':'pose.26.hand','face':'pose.27.face',
        'pred_keypoints_3d':'map.92.keypoints','pred_vertices':'map.90.vertices','pred_joint_coords':'map.91.joints','joint_global_rots':'map.03.joint_rotations','mhr_model_params':'pose.90.model_params'}
    for case in manifest['cases']:
        prefix,name=case['prefix'],case['input']
        if Path(prefix).name!=prefix or Path(name).name!=name or digest(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('invalid case identity')
        target=a.output/(prefix+'.bin');subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str(a.gguf.resolve()),str(a.reference/name),str(target),str(a.threads)],env=env,check=True)
        flat=np.fromfile(target,dtype='<f4');offset=0
        if case['order']!=sorted(case['order']) or len(case['order'])!=23 or len(set(case['order']))!=23:raise ValueError('invalid tap order')
        for key in case['order']:
            shape=case['shapes'][key];count=int(np.prod(shape));v=flat[offset:offset+count].reshape(shape);offset+=count;actual[prefix+'.'+key]=v.copy();check(prefix+'.'+key,v,ref[prefix+'.'+key])
        if offset!=flat.size:raise ValueError('trailing native output')
        for key,native_name in output_names.items():check(prefix+'.full.'+key,actual[prefix+'.'+native_name],full[prefix+'.'+key])
    save_file(actual,a.output/'native.safetensors')
    report={'scope':manifest['scope'],'reference_manifest_sha256':digest(a.reference/'manifest.json'),'gguf_sha256':digest(a.gguf),'runner_sha256':digest(a.runner),'module_sha256':digest(a.module),
        'script_sha256':digest(Path(__file__)),'backend':a.backend,'reference_device':manifest['device'],'threads':a.threads,'thresholds':{'max_abs':1e-4,'relative_l2':2e-5},'checks':checks,'passed':all(x['pass'] for x in checks)}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':report['passed'],'checks':len(checks),'failures':[x for x in checks if not x['pass']][:5]},indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
