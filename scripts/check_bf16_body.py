#!/usr/bin/env python3
"""Check all native final Body fields against a frozen original-only BF16 gate."""
import argparse,json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from check_parity import compare_array,sha256_file,read_rules,validate_rules
from check_body_api import read_output
from compare_body_precision import FIELDS
from calibrate_bf16_body import HAND_FIELDS

def check(native,reference,policy):
    rules=validate_rules(policy)
    if policy.get('native_candidate_used') is not False or {r['name'] for r in rules}!=set(FIELDS):raise ValueError('invalid final BF16 rule coverage/provenance')
    if sha256_file(reference)!=policy['reference_sha256']:raise ValueError('original result identity mismatch')
    candidate=read_output(native)
    if set(candidate)!=set(FIELDS)|{'faces'}:raise ValueError('public field coverage mismatch')
    # Geometry topology must be valid even though upstream faces are a model
    # asset rather than a predicted tensor in this benchmark output archive.
    faces=candidate['faces'];vertices=candidate['vertices']
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or np.any(faces<0) or np.any(faces>=vertices.shape[-2]):raise ValueError('invalid native face topology')
    checks=[];blocking=[];reported=[]
    with safe_open(reference,framework='np') as f:
        for rule in rules:
            name=rule['name']
            if rule.get('reference_key')!=FIELDS[name] or rule.get('blocking')!=(name not in HAND_FIELDS):raise ValueError('invalid field identity/severity')
            r=f.get_tensor(FIELDS[name]);v=candidate[name]
            result=compare_array(r,v,rule)
            if name in {'vertices','joints','keypoints'} and r.shape==v.shape and np.isfinite(v).all():
                distance=np.linalg.norm(v.astype(np.float64)-r.astype(np.float64),axis=-1)
                result.update(mean_euclidean_metres=float(distance.mean()),max_euclidean_metres=float(distance.max()))
            if not result['pass']:
                # A nonfinite hand is never a harmless hand-only precision miss.
                (blocking if rule['blocking'] or 'error' in result else reported).append(name)
            checks.append(result)
    return dict(scope=policy['boundary'],passed=not blocking,blocking_failures=blocking,nonblocking_hand_failures=reported,fields=checks)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['native','reference','rules','report']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();report=check(a.native,a.reference,read_rules(a.rules))
    report.update(native_sha256=sha256_file(a.native),reference_sha256=sha256_file(a.reference),rules_sha256=sha256_file(a.rules),script_sha256=sha256_file(Path(__file__)))
    with a.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ['passed','blocking_failures','nonblocking_hand_failures']}))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
