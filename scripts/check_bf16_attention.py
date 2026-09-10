#!/usr/bin/env python3
"""Run all isolated trained attention cases against pre-frozen original controls.

Run inside run_bounded.py. Deliberately not an own-intermediate model parity
claim. Failures stay failures; this runner never recalibrates a tolerance.
"""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from safetensors import safe_open
from check_parity import sha256_file,validate_rules,compare_array

POLICY_SHA='bda549c44a11c03d34d8c46b8cede36a4f9d4656ba0445fae09cfb867e4eec78'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','output','runner','module']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--description',required=True)
    a=p.parse_args();policy_path=a.reference/'policy.json'
    if sha256_file(policy_path)!=POLICY_SHA:raise ValueError('not the frozen original-only attention policy')
    policy=json.loads(policy_path.read_text());rules=validate_rules(policy)
    if policy['native_candidate_used'] is not False or len(rules)!=32 or sha256_file(a.reference/'upstream.safetensors')!=policy['upstream_sha256']:raise ValueError('original policy identity mismatch')
    a.output.mkdir(exist_ok=False,parents=True);checks=[]
    with safe_open(a.reference/'upstream.safetensors',framework='np') as ref:
        for rule,record in zip(rules,policy['records'],strict=True):
            name=rule['name'];source=a.reference/(name+'.input');target=a.output/(name+'.bin')
            if name!=record['name'] or sha256_file(source)!=record['input_sha256']:raise ValueError('original input identity mismatch')
            subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.description,str(source),str(target)],check=True)
            values=np.fromfile(target,dtype='<f4').reshape(record['shape'])
            result=compare_array(ref.get_tensor(name+'.cuda_math'),values,rule)
            result['native_sha256']=sha256_file(target);checks.append(result)
            print(name,result['pass'],result.get('max_abs'),result.get('relative_l2'),flush=True)
    report=dict(passed=all(c['pass'] for c in checks),scope=policy['boundary'],checks=checks,
        policy_sha256=POLICY_SHA,runner_sha256=sha256_file(a.runner),module_sha256=sha256_file(a.module),script_sha256=sha256_file(Path(__file__)))
    with (a.output/'report.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
