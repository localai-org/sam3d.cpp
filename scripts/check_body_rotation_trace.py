#!/usr/bin/env python3
"""Report exact arithmetic differences in the isolated rotation trace, not a model gate."""
import argparse,json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from run_mhr_skeleton import digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--compare-reference',type=Path);a=p.parse_args();root=a.directory
    manifest=json.loads((root/'manifest.json').read_text());reference=load_file(root/'upstream.safetensors');offset=0;stages=[]
    other=None
    if a.compare_reference:
        if (root/'input.bin').read_bytes()!=(a.compare_reference/'input.bin').read_bytes():raise ValueError('reference inputs are not identical')
        other=load_file(a.compare_reference/'upstream.safetensors')
        if set(other)!=set(reference):raise ValueError('reference tensor sets differ')
        raw=np.array([],dtype=np.float32)
    else:raw=np.fromfile(root/'native.bin',dtype='<f4')
    for name in manifest['keys']:
        expected=np.concatenate([reference[c['prefix']+'.'+name].reshape(1,-1) for c in manifest['cases']])
        if other is not None:v=np.concatenate([other[c['prefix']+'.'+name].reshape(1,-1) for c in manifest['cases']])
        else:v=raw[offset:offset+expected.size].reshape(expected.shape);offset+=expected.size
        if not np.isfinite(v).all() or not np.isfinite(expected).all():raise ValueError('non-finite trace')
        error=v.astype(np.float64)-expected.astype(np.float64)
        cases=[{'name':c['prefix'],'max_abs':float(np.max(np.abs(error[i]))),'relative_l2':float(np.linalg.norm(error[i])/max(np.linalg.norm(expected[i].astype(np.float64)),1e-12))} for i,c in enumerate(manifest['cases'])]
        stages.append({'name':name,'max_abs':float(np.max(np.abs(error))),'different_cases':[manifest['cases'][i]['prefix'] for i in np.flatnonzero(np.any(v!=expected,axis=1))],'cases':cases})
    if offset!=raw.size:raise ValueError('trailing native trace')
    # Order by actual dependency, not the lexical dump order.
    order=['11.global.b1','12.global.b2','13.global.b3','14.global.matrix','15.global.choice','16.global.quaternion','diagnostic.abcd','diagnostic.hypot','diagnostic.middle','17.global.euler']
    if set(order)!={x['name'] for x in stages}:raise ValueError('trace stage set mismatch')
    stages=sorted(stages,key=lambda x:order.index(x['name']));first=next((x['name'] for x in stages if x['max_abs']!=0),None)
    report={'scope':'isolated scalar arithmetic on identical rotation vectors; NOT model parity acceptance','comparison':'upstream_vs_upstream' if other is not None else 'native_vs_upstream','reference_device':manifest['device'],
            'manifest_sha256':digest(root/'manifest.json'),'upstream_sha256':digest(root/'upstream.safetensors'),'compared_sha256':digest(a.compare_reference/'upstream.safetensors') if other is not None else digest(root/'native.bin'),'script_sha256':digest(Path(__file__)),
            'first_nonidentical_stage':first,'all_exact':first is None,'stages':stages}
    (root/('upstream-comparison.json' if other is not None else 'comparison.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'comparison':report['comparison'],'first_nonidentical_stage':first,'max_euler_error':next(x['max_abs'] for x in stages if x['name']=='17.global.euler')},indent=2))
if __name__=='__main__':main()
