#!/usr/bin/env python3
"""Diagnose trained QKV/LayerScale rounding; no tolerance changes or parity claim."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from capture_dino_backbone import digest


def metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('invalid diagnostic arrays')
    error = candidate - reference
    index = int(np.abs(error).argmax())
    return dict(max_abs=float(np.abs(error).max()),
                relative_l2=float(np.linalg.norm(error.reshape(-1)) / max(np.linalg.norm(reference.reshape(-1)), 1e-12)),
                worst_flat_index=index, reference_at_worst=float(reference.reshape(-1)[index]),
                candidate_at_worst=float(candidate.reshape(-1)[index]),
                reference_max_abs=float(np.abs(reference).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['weights', 'cpu', 'cuda', 'vulkan', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(6)
    hashes = {name:digest(getattr(args, name)) for name in ['weights', 'cpu', 'cuda', 'vulkan']}
    if hashes['weights'] != '98afde8ac5c13c68f3b7c7baf7cc8fa525df901fe1cec44b57d018fba11d53cc':
        raise ValueError('wrong trained weights')
    report = dict(scope='diagnostic high-precision linear and exact elementwise controls; not a parity gate',
                  inputs=hashes, torch=torch.__version__, qkv={}, layer_scale={})
    with safe_open(args.weights, framework='pt') as weights, safe_open(args.cpu, framework='numpy') as cpu, \
            safe_open(args.cuda, framework='numpy') as cuda, safe_open(args.vulkan, framework='numpy') as vulkan:
        weight = weights.get_tensor('blocks.0.attn.qkv.weight').double()
        for backend, values in [('cpu', cpu), ('cuda', cuda), ('vulkan', vulkan)]:
            norm = values.get_tensor('case.0000.02.norm1')
            actual = values.get_tensor('case.0000.03.qkv')
            # All stored QKV biases are masked to zero. F64 is a diagnostic
            # arithmetic reference, NOT a replacement for the upstream forward.
            reference = torch.nn.functional.linear(torch.from_numpy(norm).double(), weight).numpy()
            report['qkv'][backend] = metrics(reference, actual)
            report['qkv'][backend]['against_correctly_rounded_f32'] = metrics(reference.astype(np.float32), actual)
        gamma = weights.get_tensor('blocks.22.ls2.gamma').numpy()
        for backend, values in [('cpu', cpu), ('cuda', cuda), ('vulkan', vulkan)]:
            w3 = values.get_tensor('case.0002.19.w3')
            ls2 = values.get_tensor('case.0002.20.ls2')
            calculated = w3 * gamma
            report['layer_scale'][backend] = dict(
                exact_f32_multiply=np.array_equal(calculated, ls2),
                gamma_max_abs=float(np.abs(gamma).max()),
                output_max_abs=float(np.abs(ls2).max()))
        reference = cuda.get_tensor('case.0002.19.w3')
        for backend, values in [('cpu', cpu), ('vulkan', vulkan)]:
            candidate = values.get_tensor('case.0002.19.w3')
            report['layer_scale'][backend]['w3_error'] = metrics(reference, candidate)
            report['layer_scale'][backend]['ls2_error'] = metrics(
                cuda.get_tensor('case.0002.20.ls2'), values.get_tensor('case.0002.20.ls2'))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
