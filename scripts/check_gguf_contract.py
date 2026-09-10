#!/usr/bin/env python3
"""Check both native/converter schemas against upstream and test format interop.

Only tiny synthetic data are converted. This does not load learned weights or
establish numerical model parity. All temporary artifacts are deleted on exit.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from safetensors.numpy import save_file
import convert_gguf as convert
import gguf_schema as schema


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-schema',type=Path,required=True)
    parser.add_argument('--verifier',type=Path,required=True)
    parser.add_argument('--archive-test',type=Path,required=True)
    args=parser.parse_args()
    original=json.loads(args.reference_schema.read_text())
    if original['revision']!=schema.DINO_REVISION:
        raise ValueError('reference revision mismatch')
    expected=original['shapes']
    native={}
    output=subprocess.check_output([str(args.verifier.resolve()),'--schema'],text=True)
    for line in output.splitlines():
        name,*dimensions=line.split()
        if name in native: raise ValueError('duplicate native schema tensor')
        native[name]=[int(d) for d in dimensions]
    python={k:list(v) for k,v in schema.body_dino_shapes().items()}
    for label,candidate in [('native',native),('converter',python)]:
        if candidate!=expected:
            differences=sorted(k for k in candidate.keys()|expected.keys() if candidate.get(k)!=expected.get(k))
            raise ValueError(f'{label} differs from upstream: {differences}')
    print(f'Both schemas match all {len(expected)} original-factory tensor shapes',flush=True)
    with tempfile.TemporaryDirectory(prefix='sam3d-gguf-contract-') as directory:
        root=Path(directory)
        values={'linear.weight':np.arange(6,dtype=np.float32).reshape(2,3),
                'linear.bias':np.array([.25,-.5],dtype=np.float32)}
        source=root/'synthetic.safetensors'; target=root/'synthetic.gguf'
        save_file(values,source)
        # Private small-shape contract only. Full verifier must reject this file.
        manifest={'body_revision':schema.BODY_REVISION,'dinov3_revision':schema.DINO_REVISION,
            'checkpoint_sha256':schema.CHECKPOINT_SHA256,'config_sha256':schema.CONFIG_SHA256,
            'safetensors_sha256':convert.sha256(source),'source_precision':['F32']}
        convert.convert(source,target,convert.metadata(manifest),
                        {k:v.shape for k,v in values.items()},manifest['safetensors_sha256'])
        subprocess.run([str(args.archive_test.resolve()),str(root),str(target)],check=True)
        rejected=subprocess.run([str(args.verifier.resolve()),str(target)],capture_output=True,text=True)
        if rejected.returncode!=1 or 'tensor count mismatch' not in rejected.stderr:
            raise ValueError('full verifier did not cleanly reject small synthetic component')
    print('Synthetic converter/native format interoperability passed; not model parity')


if __name__=='__main__': main()
