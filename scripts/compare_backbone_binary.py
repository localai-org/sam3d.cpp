#!/usr/bin/env python3
"""Compare every captured backbone boundary, preserving original strict rules."""
import argparse,json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from check_parity import compare_array,sha256_file,read_rules,validate_rules

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','native','report']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rules',type=Path,help='explicit frozen alternate precision policy; never changes the original rules')
    a=p.parse_args();rules_path=a.rules or a.reference/'rules.json';rules=read_rules(rules_path);entries=validate_rules(rules)
    if a.rules:
        if rules.get('native_candidate_used') is not False or sha256_file(a.reference/'upstream.safetensors') not in rules.get('original_safetensors_sha256',[]):
            raise ValueError('alternate rules do not identify this original reference')
    x=np.memmap(a.native,dtype='<f4',mode='r');offset=0;checks=[]
    with safe_open(a.reference/'upstream.safetensors',framework='np') as f:
        if set(f.keys())!={r['name'] for r in entries}:raise ValueError('reference rule coverage mismatch')
        for rule in entries:
            r=f.get_tensor(rule['name']);v=x[offset:offset+r.size].reshape(r.shape);offset+=r.size
            checks.append(compare_array(r,v,rule))
    if offset!=len(x):raise ValueError('native binary extent mismatch')
    failed=[r['name'] for r in checks if not r['pass']]
    report=dict(scope=rules['boundary'],passed=not failed,failed=failed,tensors=checks,
        native_sha256=sha256_file(a.native),reference_sha256=sha256_file(a.reference/'upstream.safetensors'),
        rules_sha256=sha256_file(rules_path),script_sha256=sha256_file(Path(__file__)))
    with a.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=not failed,checked=len(checks),failed=len(failed),first_failure=failed[0] if failed else None)))
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
