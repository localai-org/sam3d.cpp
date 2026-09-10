#!/usr/bin/env python3
"""Freeze BF16 stage limits from original CPU/math/automatic CUDA controls only.

No native candidate argument or data is used. This policy covers the named
backbone/image boundary, not final geometry, isolated operations or other images.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from safetensors import safe_open
from check_parity import compare_array,sha256_file
from calibrate_trained_dino import WEIGHTS

def limits(controls,peak,*,clip_headroom=False):
    if not controls or not math.isfinite(peak) or peak<=0:raise ValueError('invalid reference signal')
    for c in controls:
        if c.get('reference_dtype')!='float32' or not c.get('pass') or any(not math.isfinite(c.get(k,float('nan'))) or c[k]<0 for k in ['max_abs','relative_l2']):
            raise ValueError('invalid original BF16 control')
    # Two times independent original-kernel variation, not a BF16 bit-exact
    # claim. Hard caps reject unstable calibration instead of accepting it.
    absolute=max(1e-4,2*max(c['max_abs'] for c in controls))
    relative=max(2e-5,2*max(c['relative_l2'] for c in controls))
    absolute_cap=max(1e-4,.1*peak)
    if clip_headroom:
        # A new, explicitly selected original-only policy may reserve less
        # than 2x margin, never exceed the same safety ceilings. Reject if even
        # an observed original control is outside those ceilings. This must
        # be frozen before reading a native candidate, not used to revise an
        # already-frozen image policy after a native failure.
        if any(c['max_abs']>absolute_cap or c['relative_l2']>.05 for c in controls):
            raise ValueError('observed upstream variation exceeds calibration safety cap')
        absolute=min(absolute,absolute_cap);relative=min(relative,.05)
    elif relative>.05 or absolute>absolute_cap:
        raise ValueError('upstream variation exceeds calibration safety cap')
    return dict(max_abs=absolute,relative_l2=relative,zero_reference_floor=1e-12)

def original(directory,device,sdpa):
    m=json.loads((directory/'manifest.json').read_text())
    if m.get('device')!=device or m.get('dtype')!='bfloat16' or m.get('sdpa')!=sdpa or m.get('synthetic_weights') is not False or m.get('unobserved_repeats')!=3 or m.get('unobserved_vs_observed')!='exact' or m.get('tf32') is not False or m.get('safetensors_sha256')!=WEIGHTS:
        raise ValueError('requires matching repeatable original trained BF16 controls')
    for name in ['upstream.safetensors',*[c['input'] for c in m['cases']]]:
        if Path(name).name!=name or sha256_file(directory/name)!=m['artifacts'][name]:raise ValueError('original artifact identity mismatch')
    return m

def calibrate(math_dir,cpu_dir,auto_dir,*,clip_headroom=False):
    paths=[math_dir,cpu_dir,auto_dir];m=[original(math_dir,'cuda','math'),original(cpu_dir,'cpu','math'),original(auto_dir,'cuda','auto')]
    for other in m[1:]:
        for key in ['cases','source_hashes','body_source_hashes','image_reference_sha256','safetensors_sha256']:
            if m[0][key]!=other[key]:raise ValueError('original scope/input/source mismatch: '+key)
        for c in m[0]['cases']:
            if m[0]['artifacts'][c['input']]!=other['artifacts'][c['input']]:raise ValueError('original input mismatch')
    names=[c['prefix']+'.'+k for c in m[0]['cases'] for k in c['order']]
    rules=[];observations=[]
    with safe_open(paths[0]/'upstream.safetensors',framework='np') as base,safe_open(paths[1]/'upstream.safetensors',framework='np') as cpu,safe_open(paths[2]/'upstream.safetensors',framework='np') as auto:
        if any(set(f.keys())!=set(names) for f in [base,cpu,auto]):raise ValueError('original tensor coverage mismatch')
        for name in names:
            r=base.get_tensor(name)
            diagnostic=dict(name=name,mode='float',max_abs=1e30,relative_l2=1e30,zero_reference_floor=1e-12)
            controls=[compare_array(r,f.get_tensor(name),diagnostic) for f in [cpu,auto]]
            peak=float(np.max(abs(r)));rule=dict(name=name,mode='float',**limits(controls,peak,clip_headroom=clip_headroom))
            rules.append(rule)
            observations.append(dict(name=name,reference_peak=peak,controls=[{k:v for k,v in c.items() if k!='limits'} for c in controls]))
    return dict(schema_version=1,boundary=m[0]['boundary']+('; BF16 bounded-headroom original-control stage policy v2' if clip_headroom else '; BF16 original-control stage policy v1'),
        policy=('min(unchanged safety cap, max(legacy floor, 2 * maximum original CPU/math-CUDA and auto/math-CUDA discrepancy)); every measured original control must fit; relative-L2 cap 5%, peak-relative absolute cap 10%' if clip_headroom else 'max(legacy floor, 2 * maximum original CPU/math-CUDA and auto/math-CUDA discrepancy); relative-L2 cap 5%, peak-relative absolute cap 10%'),
        native_candidate_used=False,scope_limit='named image/backbone stages only; requires separate held-out image, isolated-operation and final-body acceptance',
        reference_manifests=[dict(device=x['device'],sdpa=x['sdpa'],sha256=sha256_file(p/'manifest.json')) for p,x in zip(paths,m)],
        original_safetensors_sha256=[sha256_file(p/'upstream.safetensors') for p in paths],weights_sha256=WEIGHTS,
        script_sha256=sha256_file(Path(__file__)),controls=observations,tensors=rules)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cuda-math','cpu-math','cuda-auto','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--clip-headroom',action='store_true',help='new original-only v2 policy: reduce 2x margin to existing caps; never raise caps or replace a frozen policy')
    a=p.parse_args();report=calibrate(a.cuda_math,a.cpu_math,a.cuda_auto,clip_headroom=a.clip_headroom)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(tensors=len(report['tensors']),native_candidate_used=False,sha256=sha256_file(a.output))))
if __name__=='__main__':main()
