#!/usr/bin/env python3
"""Package small original synthetic wrist-frame cases, without checkpoint buffers."""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from check_parity import sha256_file
from make_hand_crop_regression import fingerprint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m = json.loads((a.reference / 'manifest.json').read_text())
    if 'explicit synthetic frame and mask buffers' not in m['oracle']:
        raise ValueError('synthetic-only original capture required')
    tensors = a.reference / 'upstream.safetensors'
    if sha256_file(tensors) != m['artifacts'][tensors.name]:
        raise ValueError('changed original outputs')
    values = load_file(tensors)
    lines = ['S3D_HAND_FRAME_REGRESSION_V1', '2']
    for case in [m['cases'][0], m['cases'][2]]:
        path = a.reference / case['input']
        if path.name != case['input'] or sha256_file(path) != m['artifacts'][path.name]:
            raise ValueError('changed original input')
        raw = path.read_bytes()
        batch, = struct.unpack_from('<I', raw, 8)
        count = batch * (6 + 204 + 924) + 15
        if raw[:8] != b'S3DHFR01' or batch not in (2, 4) or len(raw) != 12 + count * 4 + 145 * 4:
            raise ValueError('not a small frame regression')
        inputs = np.frombuffer(raw, dtype='<f4', count=count, offset=12)
        indices = np.frombuffer(raw, dtype='<i4', count=145, offset=12 + count * 4)
        # Verify actual bytes, not just the provenance claim.
        frame = inputs[batch * 6:batch * 6 + 15]
        expected = np.array([0,-1,0,1,0,0,0,0,1,.4,.2,-.1,-.1,.3,.2], dtype=np.float32)
        if not np.array_equal(frame, expected) or not np.array_equal(indices, np.arange(6,151)):
            raise ValueError('non-synthetic frame or mask')
        lines += [str(batch), ' '.join(format(float(x), '.9g') for x in inputs),
                  ' '.join(str(int(x)) for x in indices), str(len(case['order']))]
        for key in case['order']:
            value = values[case['prefix'] + '.' + key]
            lines.append(f'{key} {value.size}')
            lines.append(str(fingerprint(value.tobytes())) if key.startswith(('05.', '06.'))
                         else ' '.join(format(float(x), '.9g') for x in value.ravel()))
    with a.output.open('x') as f:
        f.write('\n'.join(lines) + '\n')
    provenance = dict(fixture_sha256=sha256_file(a.output), bytes=a.output.stat().st_size,
                      original_manifest_sha256=sha256_file(a.reference/'manifest.json'),
                      original_head_sha256=m['head_sha256'], roma_sha256=m['roma_sha256'],
                      scope='Synthetic inputs/buffers and original component outputs only; no trained parameters.',
                      geometry_max_abs=1e-4, geometry_relative_l2=2e-5, masks='byte-exact FNV-1a regression')
    with a.output.with_suffix('.json').open('x') as f:
        json.dump(provenance, f, indent=2); f.write('\n')
    print(json.dumps(provenance))

if __name__ == '__main__':
    main()
