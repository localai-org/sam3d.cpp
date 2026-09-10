#!/usr/bin/env python3
"""Compare complete native hand-component outputs with original captures."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from check_parity import sha256_file,compare_array
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runner','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();m=json.loads((a.reference/'manifest.json').read_text());rules=json.loads((a.reference/'rules.json').read_text())
    for name in ['upstream.safetensors','rules.json']:
        if sha256_file(a.reference/name)!=m['artifacts'][name]:raise ValueError('reference changed')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    upstream=load_file(a.reference/'upstream.safetensors');rule={v['name']:v for v in rules['tensors']};values={};checks=[]
    binary=sha256_file(a.runner)
    for case in m['cases']:
        name=case['input'];prefix=case['prefix']
        if Path(name).name!=name or Path(prefix).name!=prefix or sha256_file(a.reference/name)!=m['artifacts'][name] or case['order']!=sorted(set(case['order'])):raise ValueError('invalid original case')
        target=a.output/(prefix+'.bin');subprocess.run([str(a.runner.resolve()),str((a.reference/name).resolve()),str(target.resolve())],check=True)
        flat=np.fromfile(target,dtype='<f4');offset=0
        for key in case['order']:
            shape=case['shapes'][key];count=int(np.prod(shape));value=flat[offset:offset+count].reshape(shape);offset+=count
            name=prefix+'.'+key;values[name]=value.copy();checks.append(compare_array(upstream[name],value,rule[name]))
        if offset!=flat.size:raise ValueError('native output size mismatch')
    if sha256_file(a.runner)!=binary or set(values)!=set(upstream) or set(values)!=set(rule):raise ValueError('incomplete capture or changed binary')
    save_file(values,a.output/'native.safetensors')
    report=dict(scope=m['scope'],checks=checks,passed=all(v['pass'] for v in checks),runner_sha256=binary,reference_sha256=sha256_file(a.reference/'manifest.json'),
                script_sha256=sha256_file(Path(__file__)),native_sha256=sha256_file(a.output/'native.safetensors'))
    with (a.output/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(checks=len(checks),passed=report['passed'],failures=[v for v in checks if not v['pass']]),indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
