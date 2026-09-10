#!/usr/bin/env python3
"""Freeze trained DINO activation limits from original CPU/CUDA controls only.

This deliberately does not accept a native candidate. Legacy synthetic limits
and failed reports remain unchanged. Not applicable to images, geometry, poses,
other models or performance/quality acceptance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from check_parity import compare_array, sha256_file

WEIGHTS = '98afde8ac5c13c68f3b7c7baf7cc8fa525df901fe1cec44b57d018fba11d53cc'


def limits(control):
    if not control['pass'] or control['reference_dtype'] != 'float32':
        raise ValueError('invalid F32 original control')
    # Four times the original cross-kernel discrepancy allows independent F32
    # reduction orders; it is not an exact-arithmetic or statistical guarantee.
    # Retain the original floors and cap the permitted relative-L2 budget.
    absolute = max(1e-4, 4 * control['max_abs'])
    relative = max(2e-5, 4 * control['relative_l2'])
    if relative > 1e-3:
        raise ValueError('original control unstable; cannot calibrate')
    return dict(max_abs=absolute, relative_l2=relative, zero_reference_floor=1e-12)


def reference(directory, device):
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('device') != device or manifest.get('synthetic_weights') is not False or \
            manifest.get('dtype') != 'float32' or manifest.get('unobserved_repeats') != 3 or \
            manifest.get('unobserved_vs_observed') != 'exact' or manifest.get('sdpa') != 'math' or \
            manifest.get('tf32') is not False:
        raise ValueError('requires repeatable original trained F32 CPU/CUDA controls')
    if manifest.get('weights_sha256', manifest.get('safetensors_sha256')) != WEIGHTS:
        raise ValueError('unsupported trained weights')
    path = directory / 'upstream.safetensors'
    if sha256_file(path) != manifest['artifacts']['upstream.safetensors']:
        raise ValueError('original capture hash mismatch')
    for case in manifest['cases']:
        if Path(case['input']).name != case['input'] or sha256_file(directory / case['input']) != manifest['artifacts'][case['input']]:
            raise ValueError('original input hash mismatch')
    return manifest, path


def calibrate(cpu_dir, cuda_dir):
    cpu, cpu_path = reference(cpu_dir, 'cpu')
    cuda, cuda_path = reference(cuda_dir, 'cuda')
    if cpu['boundary'] != cuda['boundary'] or cpu['cases'] != cuda['cases'] or cpu['source_hashes'] != cuda['source_hashes']:
        raise ValueError('original control scope/source mismatch')
    for case in cpu['cases']:
        if cpu['artifacts'][case['input']] != cuda['artifacts'][case['input']]:
            raise ValueError('original controls must have identical inputs and state')
    rules, controls = [], []
    with safe_open(cpu_path, framework='numpy') as first, safe_open(cuda_path, framework='numpy') as second:
        names = [case['prefix'] + '.' + key for case in cpu['cases'] for key in case['order']]
        if set(first.keys()) != set(names) or set(second.keys()) != set(names):
            raise ValueError('original control tensor-set mismatch')
        for name in names:
            a, b = first.get_tensor(name), second.get_tensor(name)
            diagnostic = dict(name=name, mode='float', max_abs=1e30, relative_l2=1e30, zero_reference_floor=1e-12)
            control = compare_array(b, a, diagnostic)
            rule = dict(name=name, mode='float', **limits(control))
            # A huge absolute allowance relative to signal magnitude is not a
            # rounding budget. Require separate investigation instead.
            peak = float(np.max(np.abs(b)))
            if rule['max_abs'] > max(1e-4, peak * 1e-4):
                raise ValueError(f'original absolute discrepancy too large: {name}')
            controls.append({key:value for key,value in control.items() if key != 'limits'} | dict(reference_peak=peak))
            rules.append(rule)
    return dict(schema_version=1, boundary=cpu['boundary'] + '; trained F32 original-control policy v1',
                policy='max(legacy floor, 4 * original CPU/CUDA discrepancy), per boundary; relative cap 1e-3 and peak-relative absolute cap 1e-4',
                native_candidate_used=False, weights_sha256=WEIGHTS,
                cpu_manifest_sha256=sha256_file(cpu_dir / 'manifest.json'),
                cuda_manifest_sha256=sha256_file(cuda_dir / 'manifest.json'),
                script_sha256=sha256_file(Path(__file__)), controls=controls, tensors=rules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-reference', type=Path, required=True)
    parser.add_argument('--cuda-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(args.cpu_reference, args.cuda_reference)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(boundary=report['boundary'], tensors=len(report['tensors']),
                          sha256=sha256_file(args.output), native_candidate_used=False), indent=2))


if __name__ == '__main__':
    main()
