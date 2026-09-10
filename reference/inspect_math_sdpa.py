#!/usr/bin/env python3
"""Observe real PyTorch math-SDPA dispatch on trained captured Q/K/V.

Does not replace attention with a hand-written reference. The separate formula
controls are labelled auxiliary and checked against the unmodified original call.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors import safe_open
from safetensors.numpy import save_file
from capture_dino_backbone import digest


class ObserveMathSDPA(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations, self.taps = [], {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        self.operations.append(dict(op=name,
            inputs=[dict(shape=list(x.shape), dtype=str(x.dtype)) if isinstance(x, torch.Tensor)
                    else str(x) for x in args]))
        if name == 'aten._safe_softmax.default':
            self.taps['logits'] = args[0].detach().clone()
            self.taps['probabilities'] = result.detach().clone()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = {'cpu':'032e1db812a8df6c439233359cc4896ef2ecc7070bfbd82e903f8c7f3f8b57b7',
                'cuda':'25f27966816ddf4fb91a46f09fd6420cb38db463e20140552df6a6091884df0f'}
    if digest(args.capture) != expected[args.device]:
        raise ValueError('original trained capture identity mismatch')
    if any(args.output.iterdir()):
        raise FileExistsError('use an empty output directory')
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(device=args.device, torch=torch.__version__, source_sha256=digest(args.capture),
                  scope='real math-SDPA dispatch on original trained Q/K/V, not full backbone', cases=[])
    tensors = {}
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), safe_open(args.capture, framework='pt') as safe:
        for index in range(3):
            prefix = f'case.{index:04d}.'
            q, k, v = [safe.get_tensor(prefix + key).to(args.device) for key in ['07.q_rope', '08.k_rope', '06.v']]
            original = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            observer = ObserveMathSDPA()
            with observer:
                observed = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            if not torch.equal(original, observed):
                raise ValueError('dispatch observation changed original math SDPA')
            upstream = safe.get_tensor(prefix + '11.attention').to(args.device)
            if not torch.equal(original.transpose(1, 2).reshape(1, 1029, 1280), upstream):
                raise ValueError('standalone SDPA differs from original trained block')
            if set(observer.taps) != {'logits', 'probabilities'}:
                raise ValueError('actual softmax dispatch was not observed')
            scale = (q.shape[-1] ** -.5) ** .5
            pre = (q * scale) @ (k * scale).transpose(-1, -2)
            post = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -.5)
            comparisons = {}
            for label, logits in [('scaled_operands', pre), ('scaled_product', post)]:
                comparisons[label] = dict(
                    logits_exact=torch.equal(logits, observer.taps['logits']),
                    logits_max_abs=float((logits-observer.taps['logits']).abs().max()),
                    probabilities_max_abs=float((logits.softmax(-1)-observer.taps['probabilities']).abs().max()))
            report['cases'].append(dict(prefix=prefix, observation_exact=True,
                original_block_attention_exact=True, auxiliary_formulas=comparisons, operations=observer.operations))
            for name, value in observer.taps.items():
                tensors[prefix+name] = value.cpu().numpy().copy()
            print(json.dumps(dict(case=index, formulas=comparisons)), flush=True)
    save_file(tensors, args.output / 'actual-sdpa.safetensors')
    report['tensors_sha256'] = digest(args.output / 'actual-sdpa.safetensors')
    report['script_sha256'] = digest(Path(__file__))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
