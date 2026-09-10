#!/usr/bin/env python3
"""Check the actual browser job, wire/JSON/GLB geometry and ORIGINAL final mesh.
This is final-output demo QA, complementing (not replacing) layer-level parity.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from check_body_api import read_output
from check_parity import sha256_file

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['job','reference','report']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();job=json.loads((a.job/'job.json').read_text());ref=json.loads((a.reference/'manifest.json').read_text())
    if job['state']!='complete':raise ValueError('requires completed real job')
    if sha256_file(a.job/'image.input')!=ref['native_input_sha256'] or job['native_input_sha256']!=ref['native_input_sha256']:raise ValueError('different input/camera/box')
    for name in ['result.json','input.png']:
        if sha256_file(a.reference/name)!=ref['artifacts'][name]:raise ValueError('original comparison asset changed')
    if sha256_file(a.job/'result.bin')!=job['result_sha256'] or sha256_file(a.job/'body.glb')!=job['glb_sha256']:raise ValueError('job artifact changed')
    native=read_output(a.job/'result.bin');view=json.loads((a.job/'result.json').read_text());original=json.loads((a.reference/'result.json').read_text())
    wire=[]
    for name,v in native.items():
        js=np.asarray(view['faces'] if name=='faces' else view['tensors'][name],dtype=v.dtype).reshape(v.shape)
        wire.append(dict(name=name,exact=np.array_equal(v,js)))
    if not all(x['exact'] for x in wire):raise ValueError('JSON changed native outputs')
    if not np.array_equal(native['faces'].reshape(-1),original['faces']):raise ValueError('original faces differ')
    checks=[]
    # Same existing strict final-field limits used by the branch capture.
    # No alignment, rescaling, filtering or tolerance adjustment.
    for name in ['vertices','joints','keypoints','camera_translation','vertices_pixels','keypoints_pixels']:
        v=native[name].astype(np.float64);o=np.asarray(original['tensors'][name],np.float64).reshape(v.shape);d=v-o
        maximum=float(np.abs(d).max());relative=float(np.linalg.norm(d)/max(np.linalg.norm(o),1e-12));limit=1e-3 if name.endswith('_pixels') else 1e-4
        checks.append(dict(name=name,max_abs=maximum,relative_l2=relative,max_abs_limit=limit,relative_l2_limit=2e-5,passed=maximum<=limit and relative<=2e-5))
    glb=(a.job/'body.glb').read_bytes();magic,version,length,n,chunk=struct.unpack_from('<5I',glb)
    if (magic,version,length,chunk)!=(0x46546c67,2,len(glb),0x4e4f534a):raise ValueError('bad GLB')
    doc=json.loads(glb[20:20+n]);data=memoryview(glb)[28+n:];v=np.frombuffer(data,dtype='<f4',count=18439*3).reshape(1,18439,3)
    expected=native['vertices']*np.array([1,-1,-1],dtype=np.float32)
    faces=np.frombuffer(data,dtype='<u4',count=36874*3,offset=doc['bufferViews'][1]['byteOffset']).reshape(36874,3)
    if not np.array_equal(v,expected) or not np.array_equal(faces,native['faces']):raise ValueError('export changed vertices/topology')
    report=dict(scope=__doc__,job_id=job['id'],job_sha256=sha256_file(a.job/'job.json'),native_result_sha256=job['result_sha256'],glb_sha256=job['glb_sha256'],
        original_manifest_sha256=ref['original_manifest_sha256'],prepared_reference_sha256=sha256_file(a.reference/'manifest.json'),
        script_sha256=sha256_file(Path(__file__)),wire_checks=wire,original_checks=checks,glb_all_vertices_and_indices_exact=True,passed=all(c['passed'] for c in checks))
    with a.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
