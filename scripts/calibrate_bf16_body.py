#!/usr/bin/env python3
"""Freeze final-body limits from original CUDA BF16 math/auto/compiled controls.

No native input. This single-image policy is a final-output gate, not evidence
for all layers/images. Hand-only numerical misses are reported, not blocking;
shape, coverage and finiteness remain mandatory for every field.
"""
import argparse,json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from check_parity import compare_array,sha256_file
from compare_body_precision import FIELDS

HAND_FIELDS={'hand','hand_boxes','hand_logits'}

def field_limits(name,controls,peak):
    if not controls or not np.isfinite(peak) or peak<0:raise ValueError('invalid original reference')
    for c in controls:
        if c.get('pass') is not True or any(not np.isfinite(c.get(k,np.nan)) or c[k]<0 for k in ['max_abs','relative_l2']):raise ValueError('invalid original control')
    absolute=max(1e-4,2*max(c['max_abs'] for c in controls))
    relative=max(2e-5,2*max(c['relative_l2'] for c in controls))
    # Hard safety ceilings do not grow when a native candidate misses a gate.
    cap=.01 if name in {'vertices','joints','keypoints','camera_translation'} else (2. if name.endswith('_pixels') else max(1e-4,.1*peak))
    if absolute>cap or relative>.05:
        raise ValueError(f'unstable original final-body control {name}: derived max_abs={absolute:g} (ceiling {cap:g}), relative_l2={relative:g} (ceiling .05)')
    return dict(max_abs=absolute,relative_l2=relative,zero_reference_floor=1e-12)

def calibrate(auto,math,compiled):
    dirs=[auto,math,compiled];records=[]
    for directory,attention,compile_mode in zip(dirs,['auto','math','auto'],['none','none','backbone'],strict=True):
        m=json.loads((directory/'benchmark.json').read_text())
        if any(m.get(k)!=v for k,v in dict(status='complete',precision='bf16',attention=attention,compile=compile_mode,device='cuda',tf32=False,gradients=False,resident_weights=True).items()):raise ValueError('requires original completed BF16 benchmark controls')
        if compile_mode!='none' and m.get('compiled_graphs',0)<1:raise ValueError('compiled control did not compile')
        if sha256_file(directory/'final-output.safetensors')!=m['final_output_sha256']:raise ValueError('original output hash mismatch')
        # The compiled harness corrected a stale residency *description*;
        # both benchmark paths already remove all loading hooks before timing.
        # Keep both harness identities in the record; all neural source hashes,
        # pixels, boxes, intrinsics and weights still have to match exactly.
        def neural_source(record):
            return {k:v for k,v in record['source'].items() if k not in {'script_sha256','reference_weight_residency'}}
        if records and (neural_source(m)!=neural_source(records[0]) or any(m[k]!=records[0][k] for k in ['trained_state','torch','cuda'])):raise ValueError('original scope/weight/source mismatch')
        records.append(m)
    files=[safe_open(d/'final-output.safetensors',framework='np') for d in dirs]
    rules=[];observations=[]
    for name,key in FIELDS.items():
        r=files[0].get_tensor(key)
        rule=dict(name=name,mode='float',max_abs=1e30,relative_l2=1e30,zero_reference_floor=1e-12)
        controls=[compare_array(r,f.get_tensor(key),rule) for f in files[1:]]
        peak=float(abs(r).max())
        rules.append(dict(name=name,reference_key=key,mode='float',blocking=name not in HAND_FIELDS,**field_limits(name,controls,peak)))
        observations.append(dict(name=name,reference_peak=peak,controls=[{k:v for k,v in c.items() if k!='limits'} for c in controls]))
    return dict(schema_version=1,boundary='single official photograph '+records[0]['source']['photo_sha256']+': complete own-intermediate BF16 Body final outputs v1',
        native_candidate_used=False,policy='two times maximum original math/auto or compiled/auto variation; strict floors and independent safety ceilings',
        scope_limit='one named image, no detector or hand refinement; separate layer, held-out image and visual acceptance still required',
        reference_sha256=records[0]['final_output_sha256'],original_manifests_sha256=[sha256_file(d/'benchmark.json') for d in dirs],
        original_output_sha256=[m['final_output_sha256'] for m in records],source=records[0]['source'],
        original_harnesses=[dict(script_sha256=m['source']['script_sha256'],residency_description=m['source']['reference_weight_residency']) for m in records],
        script_sha256=sha256_file(Path(__file__)),controls=observations,tensors=rules)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['auto','math','compiled','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();report=calibrate(a.auto,a.math,a.compiled)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(fields=len(report['tensors']),sha256=sha256_file(a.output),native_candidate_used=False)))
if __name__=='__main__':main()
