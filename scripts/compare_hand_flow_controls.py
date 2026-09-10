#!/usr/bin/env python3
"""Measure original CPU/CUDA hand-flow stability; diagnostic, never a new tolerance policy."""
import argparse,json
from pathlib import Path
from safetensors.numpy import load_file
from check_parity import compare_array,sha256_file
from rotation_parity import compare_euler

def compare(name,reference,candidate):
    if name.endswith('.pose.global_rot') or name.endswith('.full.global_rot'):
        return dict(name=name,**compare_euler(reference,candidate))
    pixel=any(name.endswith(x) for x in ['03.vertex_pixels','12.scaled_box','19.intrinsic_projection','20.pixels','pred_keypoints_2d','pred_keypoints_2d_verts'])
    return compare_array(reference,candidate,dict(name=name,mode='float',max_abs=1e-3 if pixel else 1e-4,relative_l2=2e-5,zero_reference_floor=1e-12))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cpu','cuda','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();manifests=[];values=[]
    for path,device in [(a.cpu,'cpu'),(a.cuda,'cuda')]:
        m=json.loads((path/'manifest.json').read_text())
        if m['device']!=device or not m.get('hand_branch') or not m['learned_sam_checkpoint_loaded'] or len(m['cases'])!=1:
            raise ValueError('requires original trained hand-flow controls')
        groups=[]
        for file in ['upstream.safetensors','full.safetensors']:
            if sha256_file(path/file)!=m['artifacts'][file]:raise ValueError('original outputs changed')
            groups.append(load_file(path/file))
        for case in m['cases']:
            if Path(case['input']).name!=case['input'] or sha256_file(path/case['input'])!=m['artifacts'][case['input']]:raise ValueError('original input changed')
            if any(v['max_abs']!=0 for v in case['unobserved_vs_observed'].values()):raise ValueError('observer not neutral')
        manifests.append(m);values.append(groups)
    cpu,cuda=manifests
    for key in ['source_sha256','trained_state','reference_helper_sha256','model_sha256','roma_sha256']:
        if cpu[key]!=cuda[key]:raise ValueError('original control identity mismatch '+key)
    if cpu['cases'][0]['shape']!=cuda['cases'][0]['shape'] or cpu['artifacts'][cpu['cases'][0]['input']]!=cuda['artifacts'][cuda['cases'][0]['input']]:
        raise ValueError('original inputs/state not byte-identical')
    checks=[]
    for index in [0,1]:
        left,right=values[0][index],values[1][index]
        if set(left)!=set(right):raise ValueError('original field set mismatch')
        for name in sorted(left):
            key=name
            if index:
                key=name.rsplit('.',1)[0]+'.full.'+name.rsplit('.',1)[-1]
            checks.append(compare(key,right[name],left[name]))
    report=dict(scope=__doc__,limits='unchanged flow limits, including 1e-3 pixels and 1e-4 other F32; no limits adjusted',
                cpu_manifest_sha256=sha256_file(a.cpu/'manifest.json'),cuda_manifest_sha256=sha256_file(a.cuda/'manifest.json'),
                inputs_byte_identical=True,script_sha256=sha256_file(Path(__file__)),checks=checks,passed=all(c['pass'] for c in checks))
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(checks=len(checks),passed=sum(c['pass'] for c in checks),failed=sum(not c['pass'] for c in checks)),indent=2))

if __name__=='__main__':main()
