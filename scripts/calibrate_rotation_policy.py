#!/usr/bin/env python3
"""Freeze angular-policy evidence from original CPU/CUDA controls only."""
import argparse,json
from pathlib import Path
from safetensors.numpy import load_file
from rotation_parity import POLICY,compare_euler
from run_mhr_skeleton import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cpu','cuda','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if (a.cpu/'input.bin').read_bytes()!=(a.cuda/'input.bin').read_bytes():raise ValueError('different control inputs')
    manifests=[json.loads((p/'manifest.json').read_text()) for p in [a.cpu,a.cuda]]
    if [m['device'] for m in manifests]!=['cpu','cuda']:raise ValueError('wrong reference devices')
    cpu=load_file(a.cpu/'upstream.safetensors');cuda=load_file(a.cuda/'upstream.safetensors');cases=[]
    # These six vectors are taken ONLY from the original full-decoder output.
    # The diagnostic's additional native-produced vectors are deliberately excluded.
    for i in range(6):
        key=f'original.{i}.17.global.euler';r,v=cpu[key],cuda[key]
        cases.append({'key':key,'cpu':r.tolist(),'cuda':v.tolist(),**compare_euler(r,v)})
    if not all(c['pass'] for c in cases):raise ValueError('angular policy rejects original cross-backend control')
    result={'schema_version':1,'policy':POLICY,'scope':'global Euler radians only; independent original CPU/CUDA elementary-function control, not trained model parity',
            'uses_native_output_for_calibration':False,'max_abs_radians':1e-4,'unit_circle_relative_l2':2e-5,
            'relative_representation':'per-coordinate sin/cos has fixed nonzero reference norm sqrt(number of coordinates), including identity; raw Euler absolute bound remains mandatory',
            'raw_euler_relative_error_retained':True,'native_inference_changed':False,
            'source_manifests_sha256':[digest(p/'manifest.json') for p in [a.cpu,a.cuda]],
            'original_capture_sha256':[digest(p/'upstream.safetensors') for p in [a.cpu,a.cuda]],
            'policy_code_sha256':digest(Path(__file__).with_name('rotation_parity.py')),'script_sha256':digest(Path(__file__)),
            'cases':cases}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'policy':POLICY,'controls':len(cases),'raw_relative_rejections':sum(c['raw_euler_relative_l2']>2e-5 for c in cases),'all_policy_checks_pass':True}))
if __name__=='__main__':main()
