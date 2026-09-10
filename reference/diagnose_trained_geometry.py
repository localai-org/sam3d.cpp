#!/usr/bin/env python3
"""Separate native pose-input drift from MHR arithmetic; NOT an acceptance run."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from capture_mhr import MODEL_SHA,MODEL_BYTES,digest
from trace_mhr_repeat import error

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','reference','candidate','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA:raise ValueError('unverified MHR')
    m=json.loads((a.reference/'manifest.json').read_text())
    if m['device']!='cpu' or not m['learned_sam_checkpoint_loaded']:raise ValueError('requires trained CPU reference')
    ref=a.reference/'upstream.safetensors';candidate=a.candidate/'native.safetensors'
    if digest(ref)!=m['artifacts']['upstream.safetensors']:raise ValueError('unverified reference')
    run=json.loads((a.candidate/'report.json').read_text())
    if run['reference_manifest_sha256']!=digest(a.reference/'manifest.json'):raise ValueError('native reference identity mismatch')
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
    model=torch.jit.load(str(a.model),map_location='cpu').eval()
    arrays={};checks=[]
    with torch.inference_mode(),safe_open(ref,framework='numpy') as r,safe_open(candidate,framework='numpy') as n:
        for i in range(6):
            prefix=f'case.0000.layer.{i}.pose.'
            def inputs(source):
                values=[source.get_tensor(prefix+key) for key in ['pose.24.shape','pose.90.model_params','pose.27.face']]
                if any(v.shape!=(1,size) or v.dtype!=np.float32 or not np.isfinite(v).all() for v,size in zip(values,[45,204,72])):raise ValueError('invalid MHR inputs')
                return [torch.from_numpy(v.copy()) for v in values]
            original=inputs(r);native=inputs(n)
            model(*original,True);control=model(*original,True)[0].numpy().copy()
            replay=model(*native,True)[0].numpy().copy()
            reference=r.get_tensor(prefix+'mhr.vertices_cm');actual=n.get_tensor(prefix+'mhr.vertices_cm')
            entry=dict(layer=i,original_replay_vs_reference=error(reference,control),
                       original_using_native_inputs_vs_reference=error(reference,replay),
                       native_vs_original_using_same_native_inputs=error(replay,actual),
                       full_native_vs_reference=error(reference,actual))
            checks.append(entry);arrays[f'layer.{i}.original_on_native_inputs']=replay
            arrays[f'layer.{i}.original_on_original_inputs']=control
            print(json.dumps(entry),flush=True)
    save_file(arrays,a.output/'replayed.safetensors')
    report=dict(scope='diagnostic intervention at MHR inputs; not end-to-end parity',checks=checks,
                model_sha256=MODEL_SHA,reference_sha256=digest(ref),native_sha256=digest(candidate),
                script_sha256=digest(Path(__file__)),torch=torch.__version__,threads=6)
    with (a.output/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
if __name__=='__main__':main()
