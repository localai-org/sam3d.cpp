#!/usr/bin/env python3
"""Compact original hand-preparation regressions; no learned weights included."""
import argparse,json,struct
from pathlib import Path
from safetensors.numpy import load_file
from check_parity import sha256_file
def fingerprint(data):
    value=14695981039346656037
    for byte in data:value=((value^byte)*1099511628211)&0xffffffffffffffff
    return value
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.reference/'manifest.json').read_text());path=a.reference/'upstream.safetensors'
    if sha256_file(path)!=m['artifacts'][path.name]:raise ValueError('original reference changed')
    tensors=load_file(path);lines=['S3D_HAND_CROP_REGRESSION_V1','2']
    for case in m['cases'][1:]:
        path=a.reference/case['input']
        if sha256_file(path)!=m['artifacts'][path.name]:raise ValueError('original input changed')
        raw=path.read_bytes()
        if raw[:8]!=b'S3DHCP01':raise ValueError('wrong input protocol')
        dims=struct.unpack_from('<4I',raw,8);values=struct.unpack_from('<18f',raw,24)
        if dims[3]!=32 or dims[0]*dims[1]>4096:raise ValueError('not a small regression')
        lines+=[' '.join(map(str,dims)),' '.join(format(v,'.9g') for v in values),str(len(raw)-96),raw[96:].hex(),str(len(case['order']))]
        for key in case['order']:
            value=tensors[case['prefix']+'.'+key];lines.append(f'{key} {value.size} {fingerprint(value.tobytes())}')
    with a.output.open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(bytes=a.output.stat().st_size,sha256=sha256_file(a.output),source_sha256=sha256_file(a.reference/'manifest.json'))))
if __name__=='__main__':main()
