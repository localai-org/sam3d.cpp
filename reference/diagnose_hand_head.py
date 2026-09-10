#!/usr/bin/env python3
"""Diagnostic interventions in original MHR; never full hand-model acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file,save_file
from capture_mhr import digest,MODEL_SHA,MODEL_BYTES
from trace_mhr_repeat import error

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','reference','candidate','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();m=json.loads((a.reference/'manifest.json').read_text());report=json.loads((a.candidate/'report.json').read_text())
    if m['device']!='cpu' or report['backend']!='CPU' or report['reference_manifest_sha256']!=digest(a.reference/'manifest.json'):
        raise ValueError('requires matched original/native CPU cases')
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA:raise ValueError('unverified original MHR')
    if digest(a.reference/'upstream.safetensors')!=m['artifacts']['upstream.safetensors'] or digest(a.candidate/'native.safetensors')!=report['native_sha256']:
        raise ValueError('unverified diagnostic inputs')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=torch.jit.load(str(a.model),map_location='cpu').eval()
    original=load_file(a.reference/'upstream.safetensors');native=load_file(a.candidate/'native.safetensors');results={};checks=[]
    for case in m['cases']:
        prefix=case['prefix'];keys=[prefix+'.pose.'+k for k in ['24.shape','90.model_params','27.face']]
        a0=[torch.from_numpy(original[k].copy()) for k in keys];a1=[torch.from_numpy(native[k].copy()) for k in keys]
        variants={'original':a0,'native_inputs':a1}
        # Change one actual input group at a time in the original asset.
        for name,start,end in [('translation',0,3),('rotation',3,6),('pose',6,136),('scale',136,204)]:
            values=[x.clone() for x in a0];values[1][:,start:end]=a1[1][:,start:end];variants['native_'+name]=values
        variants['native_shape']=[a1[0],a0[1],a0[2]]
        for name,args in variants.items():
            with torch.inference_mode():verts,skel=model(*args)
            for label,value in [('vertices_cm',verts),('skeleton',skel)]:
                value=value.numpy().copy();key=prefix+'.mhr.'+label
                if not np.isfinite(value).all():raise ValueError('nonfinite original intervention')
                results[prefix+'.'+name+'.'+label]=value
                exact=value.tobytes()==original[key].tobytes()
                if name=='original' and not exact:raise ValueError('original-on-original replay differs')
                checks.append(dict(case=prefix,intervention=name,field=label,vs_original=error(original[key],value),vs_native=error(native[key],value),original_exact=exact))
        print(prefix,'original replay exact; input interventions captured',flush=True)
    save_file(results,a.output/'diagnostic.safetensors')
    metadata=dict(scope=__doc__,reference_sha256=digest(a.reference/'manifest.json'),candidate_sha256=digest(a.candidate/'report.json'),
                  model_sha256=MODEL_SHA,script_sha256=digest(Path(__file__)),checks=checks,outputs_sha256=digest(a.output/'diagnostic.safetensors'))
    with (a.output/'report.json').open('x') as f:json.dump(metadata,f,indent=2);f.write('\n')

if __name__=='__main__':main()
