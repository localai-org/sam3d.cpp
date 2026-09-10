#!/usr/bin/env python3
"""Execute native crop/intrinsics -> rays/CLIFF C API, no injected geometry."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from safetensors.numpy import save_file
from run_patch_capture import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.reference/'manifest.json').read_text());tensors={};commands=[]
    for case in manifest['cases']:
        name=case['input'];prefix=case['prefix']
        if Path(name).name!=name or Path(prefix).name!=prefix:raise ValueError('invalid fixture path')
        source=args.reference/name;target=args.output/(prefix+'.output')
        if sha256_file(source)!=manifest['artifacts'][name]:raise ValueError('camera fixture hash mismatch')
        command=[str(args.binary.resolve()),str(source.resolve()),str(target.resolve())]
        result=subprocess.run(command,capture_output=True,text=True,check=False)
        commands.append({'argv':command,'returncode':result.returncode,'stderr':result.stderr})
        (args.output/'run.json').write_text(json.dumps({'commands':commands},indent=2)+'\n');result.check_returncode()
        values=np.fromfile(target,dtype='<f4');offset=0
        for key in case['order']:
            shape=case['shapes'][key];count=int(np.prod(shape))
            tensors[prefix+'.'+key]=values[offset:offset+count].reshape(shape);offset+=count
        if offset!=len(values):raise ValueError('camera native output size mismatch')
    save_file(tensors,args.output/'native.safetensors')
    (args.output/'run.json').write_text(json.dumps({'commands':commands,'binary_sha256':sha256_file(args.binary)},indent=2)+'\n')


if __name__=='__main__':main()
