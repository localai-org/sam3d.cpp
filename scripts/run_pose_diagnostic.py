#!/usr/bin/env python3
"""Replay trained pose heads on fixed tokens; explicitly NOT end-to-end parity."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from check_parity import sha256_file,compare_array

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runner','module','reference','output','full-reference','full-candidate']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--threads',type=int,default=12);a=p.parse_args()
    manifest=json.loads((a.reference/'manifest.json').read_text())
    if 'diagnostic_inputs' not in manifest or len(manifest['cases'])!=12:raise ValueError('requires trained pose diagnostic')
    for name in ['upstream.safetensors','rules.json']:
        if sha256_file(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('diagnostic identity mismatch')
    for path,label in [(a.full_reference,'reference'),(a.full_candidate,'candidate')]:
        if sha256_file(path)!=manifest['diagnostic_inputs'][label]:raise ValueError('full capture identity mismatch')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    original=load_file(a.reference/'upstream.safetensors');fullr=load_file(a.full_reference);fulln=load_file(a.full_candidate)
    rules={v['name']:v for v in json.loads((a.reference/'rules.json').read_text())['tensors']}
    arrays={};checks=[];cohort=[];runner_hash=sha256_file(a.runner);module_hash=sha256_file(a.module)
    for i,case in enumerate(manifest['cases']):
        name=case['input'];prefix=case['prefix']
        if Path(name).name!=name or prefix!=f'case.{i:04d}' or sha256_file(a.reference/name)!=manifest['artifacts'][name] or case['order']!=sorted(set(case['order'])):raise ValueError('invalid diagnostic case')
        target=a.output/(prefix+'.bin')
        subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),'CPU','0','-',str((a.reference/name).resolve()),str(target.resolve()),str(a.threads)],check=True)
        flat=np.fromfile(target,dtype='<f4');offset=0
        for key in case['order']:
            shape=case['shapes'][key];count=int(np.prod(shape));value=flat[offset:offset+count].reshape(shape);offset+=count
            if not np.isfinite(value).all():raise ValueError('nonfinite result')
            arrays[prefix+'.'+key]=value.copy();checks.append(compare_array(original[prefix+'.'+key],value,rules[prefix+'.'+key]))
        if offset!=flat.size:raise ValueError('native output length mismatch')
        # Original-token replay must reconstruct the saved original pose inputs.
        # Candidate-token replay separates head arithmetic from incoming drift.
        selected=fullr if i<6 else fulln;fields={}
        for key in ['10.pred','24.shape','25.scale','26.hand','27.face','90.model_params']:
            saved=selected[f'case.0000.layer.{i%6}.pose.pose.{key}'];replayed=original[prefix+'.'+key]
            fields[key]=dict(exact=bool(np.array_equal(saved,replayed)),max_abs=float(np.max(np.abs(saved.astype(np.float64)-replayed.astype(np.float64)))))
            if i<6 and not fields[key]['exact']:raise ValueError('original-token replay does not reproduce full original pose '+key)
            if i>=6 and not np.array_equal(saved,arrays[prefix+'.'+key]):raise ValueError('native-token replay does not reproduce full native pose '+key)
        cohort.append(dict(case=i,source='reference' if i<6 else 'candidate',fields=fields))
    if sha256_file(a.runner)!=runner_hash or sha256_file(a.module)!=module_hash:raise ValueError('binary changed during diagnosis')
    save_file(arrays,a.output/'native.safetensors')
    report=dict(scope=__doc__,checks=checks,replay_vs_full=cohort,runner_sha256=runner_hash,module_sha256=module_hash,
                reference_manifest_sha256=sha256_file(a.reference/'manifest.json'),script_sha256=sha256_file(Path(__file__)),
                native_sha256=sha256_file(a.output/'native.safetensors'),threads=a.threads)
    with (a.output/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(checks=len(checks),failures=[v for v in checks if not v['pass']],replay_vs_full=cohort),indent=2))
if __name__=='__main__':main()
