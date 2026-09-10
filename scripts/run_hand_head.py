#!/usr/bin/env python3
"""Compare all native trained hand-head operations and independently captured final fields."""
import argparse,json,os,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from check_parity import sha256_file,compare_array,validate_rules

FIELDS={'pred_pose_raw':'pred_pose_raw','global_rot':'global_rot','body_pose':'body_pose',
        'shape':'pose.24.shape','scale':'pose.25.scale','hand':'pose.26.hand','face':'pose.27.face',
        'pred_keypoints_3d':'map.92.keypoints','pred_vertices':'map.90.vertices','pred_joint_coords':'map.91.joints',
        'joint_global_rots':'map.03.joint_rotations','mhr_model_params':'pose.90.model_params'}

def compare_final(prefix,values,full,faces):
    expected={prefix+'.'+k for k in [*FIELDS,'faces']}
    if {k for k in full if k.startswith(prefix+'.')}!=expected:raise ValueError('missing/unexpected original final fields')
    checks=[]
    for name,key in FIELDS.items():
        rule=dict(name=prefix+'.full.'+name,mode='float',max_abs=1e-4,relative_l2=2e-5,zero_reference_floor=1e-12)
        checks.append(compare_array(full[prefix+'.'+name],values[prefix+'.'+key],rule))
    checks.append(compare_array(full[prefix+'.faces'],faces.astype(np.int64),dict(name=prefix+'.full.faces',mode='exact')))
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runner','module','gguf','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backend',choices=['CPU','Vulkan'],required=True);p.add_argument('--threads',type=int,default=6)
    p.add_argument('--device',type=int,default=0);p.add_argument('--description',default='-');a=p.parse_args()
    m=json.loads((a.reference/'manifest.json').read_text())
    if m['device']!=('cpu' if a.backend=='CPU' else 'cuda') or len(m['cases'])!=2 or not m['learned_head_state_loaded']:
        raise ValueError('requires matched original trained hand head')
    for name in ['upstream.safetensors','full.safetensors','rules.json']:
        if sha256_file(a.reference/name)!=m['artifacts'][name]:raise ValueError('reference changed '+name)
    if sha256_file(a.gguf)!='d52ab772628fb6d851550381b428185a32da6398793f0ff70299bad1999ba8f8':raise ValueError('unverified MHR GGUF')
    rules=validate_rules(json.loads((a.reference/'rules.json').read_text()));rules={r['name']:r for r in rules}
    ref=load_file(a.reference/'upstream.safetensors');full=load_file(a.reference/'full.safetensors')
    if set(ref)!=set(rules):raise ValueError('incomplete operation policy')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    before={str(path):sha256_file(path) for path in [a.runner,a.module,a.gguf]}
    env=os.environ.copy()
    if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
    values={};checks=[];artifact_hashes={}
    for case in m['cases']:
        name,prefix=case['input'],case['prefix']
        if Path(name).name!=name or Path(prefix).name!=prefix or sha256_file(a.reference/name)!=m['artifacts'][name] or case['order']!=sorted(set(case['order'])):
            raise ValueError('invalid/changed original case')
        target=a.output/(prefix+'.bin')
        subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str(a.gguf.resolve()),str((a.reference/name).resolve()),str(target.resolve()),str(a.threads)],env=env,check=True)
        flat=np.fromfile(target,dtype='<f4');offset=0
        for key in case['order']:
            shape=case['shapes'][key];count=int(np.prod(shape));v=flat[offset:offset+count].reshape(shape);offset+=count
            field=prefix+'.'+key
            if field in values:raise ValueError('duplicate native field')
            values[field]=v.copy();checks.append(compare_array(ref[field],v,rules[field]))
        if offset!=flat.size:raise ValueError('native output size mismatch')
        faces_path=Path(str(target)+'.faces');faces=np.fromfile(faces_path,dtype='<i4').reshape(36874,3)
        checks+=compare_final(prefix,values,full,faces)
        # Structural mask checks remain exact, independently of float tolerances.
        params=values[prefix+'.pose.90.model_params']
        with (a.reference/name).open('rb') as f:f.seek(-145*4,2);indices=np.frombuffer(f.read(),dtype='<i4')
        points=values[prefix+'.map.07.keypoints308']
        checks.append(dict(name=prefix+'.parameter_mask_exact_zero',pass_exact=bool(np.all(params[:,indices]==0)),**{'pass':bool(np.all(params[:,indices]==0))}))
        zero=bool(np.all(points[:,:21]==0) and np.all(points[:,42:]==0))
        checks.append(dict(name=prefix+'.keypoint_mask_exact_zero',pass_exact=zero,**{'pass':zero}))
        artifact_hashes[target.name]=sha256_file(target);artifact_hashes[faces_path.name]=sha256_file(faces_path)
    if set(values)!=set(ref) or len(full)!=len(m['cases'])*(len(FIELDS)+1):raise ValueError('incomplete native/reference coverage')
    if any(sha256_file(Path(path))!=sha for path,sha in before.items()):raise ValueError('inference binaries/model changed')
    save_file(values,a.output/'native.safetensors')
    report=dict(scope=m['scope'],backend=a.backend,threads=a.threads,reference_manifest_sha256=sha256_file(a.reference/'manifest.json'),
                source_hashes=before,script_sha256=sha256_file(Path(__file__)),native_sha256=sha256_file(a.output/'native.safetensors'),artifacts=artifact_hashes,
                checks=checks,passed=all(v['pass'] for v in checks))
    with (a.output/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failures=[v for v in checks if not v['pass']]),indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
