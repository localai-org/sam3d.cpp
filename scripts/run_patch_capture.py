#!/usr/bin/env python3
"""Run native diagnostic on reference inputs, pack outputs (no PyTorch/model loading)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from safetensors.numpy import save_file


def sha256_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--module',type=Path,required=True)
    parser.add_argument('--backend',choices=['CPU','Vulkan'],required=True)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--threads',type=int,help='explicit native CPU worker count (default: binary default)')
    parser.add_argument('--gguf',type=Path,help='backbone runner only: checked learned GGUF instead of inline fixture weights')
    parser.add_argument('--block-traces',action='store_true',help='learned backbone only: stream all per-block operation outputs')
    parser.add_argument('--expect-device',default='-',help='require exact device description; - skips this guard')
    parser.add_argument('--vulkan-math',choices=['f32','ggml-default'],default='f32',
                        help='strict F32 baseline, or explicit reduced-precision diagnostic')
    args = parser.parse_args()
    if args.block_traces and not args.gguf: raise ValueError('block tracing requires a learned GGUF')
    if args.threads is not None and not 1<=args.threads<=1024:
        raise ValueError('threads must be in [1,1024]')
    manifest = json.loads((args.reference/'manifest.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    tensors, commands = {}, []
    environment = os.environ.copy()
    if args.backend == 'Vulkan' and args.vulkan_math == 'f32':
        # GGML's F32 tensor storage/accumulation flags do not prevent its Vulkan
        # fp16/coopmat paths from rounding operands. Configure before module load
        # in the child only; do not mutate the host process's global environment.
        for key in ['GGML_VK_DISABLE_F16','GGML_VK_DISABLE_COOPMAT','GGML_VK_DISABLE_COOPMAT2']:
            environment[key] = '1'
    recorded_environment = {key:value for key,value in environment.items()
        if key.startswith('GGML_VK_') or key in ['VK_DRIVER_FILES','VK_ICD_FILENAMES']}
    for case in manifest['cases']:
        name = case['input']
        if Path(name).name != name or Path(case['prefix']).name != case['prefix']:
            raise ValueError('invalid fixture path')
        path = args.reference/name
        if sha256_file(path) != manifest['artifacts'][name]:
            raise ValueError('input hash mismatch')
        output = args.output/(case['prefix']+'.output')
        command = [str(args.binary.resolve()),str(args.module.resolve()),args.backend,
                   str(args.device),args.expect_device,str(path.resolve()),str(output.resolve())]
        if args.threads is not None or args.gguf is not None: command.append(str(args.threads or 1))
        if args.gguf is not None: command.append(str(args.gguf.resolve()))
        if args.block_traces: command.append(str((args.output/(case['prefix']+'.blocks')).resolve()))
        diagnostic=[]
        # Forward stage progress while a large component runs, not only at exit.
        with subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE,text=True,env=environment) as process:
            for line in process.stderr:
                diagnostic.append(line); print(line,end='',flush=True)
            returncode=process.wait()
        commands.append({'argv':command,'stderr':''.join(diagnostic),'returncode':returncode,
                         'backend_environment':recorded_environment})
        (args.output/'run.json').write_text(json.dumps({'commands':commands},indent=2)+'\n')
        if returncode: raise subprocess.CalledProcessError(returncode,command)
        data = np.fromfile(output,dtype='<f4')
        offset = 0
        for tap in case.get('order',['patches','projection','tokens']):
            shape = case['shapes'][tap]
            count = int(np.prod(shape))
            tensors[case['prefix']+'.'+tap] = data[offset:offset+count].reshape(shape)
            offset += count
        if offset != data.size: raise ValueError('unexpected output length')
    save_file(tensors,args.output/'native.safetensors')
    (args.output/'run.json').write_text(json.dumps({'commands':commands,
        'gguf_sha256':sha256_file(args.gguf) if args.gguf else None,
        'binary_sha256':sha256_file(args.binary),
        'module_sha256':sha256_file(args.module)},indent=2)+'\n')


if __name__ == '__main__': main()
