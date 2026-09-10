#!/usr/bin/env python3
"""Same-input trained DINO block diagnostics, not composed backbone acceptance.

Inputs are original CUDA backbone states. CPU, CUDA and native receive the same
bytes so this isolates a block's arithmetic from accumulated input differences.
"""
import argparse
from functools import partial
import importlib
import json
from pathlib import Path
import struct

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from capture_dino_schema import load_factory, HASHES
from capture_dino_block import PARAMETERS
from capture_dino_backbone import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--backbone-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = parser.parse_args()
    if digest(args.weights) != '98afde8ac5c13c68f3b7c7baf7cc8fa525df901fe1cec44b57d018fba11d53cc':
        raise ValueError('trained safe-weight identity mismatch')
    if digest(args.backbone_reference) != '1e61a8068a3aadfd7856938b290854192a131691b03dce242d78712920bd2a48':
        raise ValueError('original CUDA backbone input identity mismatch')
    if any(args.output.iterdir()):
        raise FileExistsError('use an empty capture directory')
    load_factory(args.upstream)  # Verifies original source and exposes unmodified classes.
    Block = importlib.import_module('dinov3.layers.block').SelfAttentionBlock
    SwiGLU = importlib.import_module('dinov3.layers.ffn_layers').SwiGLUFFN
    Rope = importlib.import_module('dinov3.layers.rope_position_encoding').RopePositionEmbedding
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tensors, cases, rules = {}, [], []
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), safe_open(args.weights, framework='pt') as weights, \
            safe_open(args.backbone_reference, framework='pt') as reference:
        for case_index, index in enumerate([0, 1, 22]):
            case = f'case.{case_index:04d}'
            with torch.device('meta'):
                net = Block(1280, 20, ffn_ratio=6, qkv_bias=True, proj_bias=True, ffn_bias=True,
                            init_values=1e-5, norm_layer=partial(torch.nn.LayerNorm, eps=1e-5),
                            ffn_layer=SwiGLU, mask_k_bias=True).eval()
            state = {name: weights.get_tensor(f'blocks.{index}.{name}').to(args.device)
                     for name in net.state_dict()}
            net.load_state_dict(state, strict=True, assign=True)
            rope_net = Rope(1280, num_heads=20, base=100, normalize_coords='separate',
                            rescale_coords=2, dtype=torch.float32, device=args.device).eval()
            rope_net.periods.copy_(weights.get_tensor('rope_embed.periods'))
            key = '01.tokens' if index == 0 else f'02.block.{index-1:02d}'
            x = reference.get_tensor('case.0000.' + key).to(args.device)
            rope = rope_net(H=32, W=32)
            baseline = net(x, rope).clone()
            for _ in range(2):
                if not torch.equal(baseline, net(x, rope)):
                    raise ValueError('block reference not repeatable')
            taps = {}
            def capture(name, value):
                taps[name] = value.detach().clone()
            capture('00.rope_sin', torch.nn.functional.pad(rope[0], (0, 0, 5, 0), value=0))
            capture('01.rope_cos', torch.nn.functional.pad(rope[1], (0, 0, 5, 0), value=1))
            hooks = []
            for attr, name in [('norm1','02.norm1'), ('attn.qkv','03.qkv'), ('attn.proj','12.attn_proj'),
                               ('ls1','13.ls1'), ('norm2','15.norm2'), ('mlp.w1','16.w1'),
                               ('mlp.w2','17.w2'), ('mlp.w3','19.w3'), ('ls2','20.ls2')]:
                hooks.append(net.get_submodule(attr).register_forward_hook(
                    lambda _m, _a, v, name=name: capture(name, v)))
            for attr, name in [('attn.proj','11.attention'), ('norm2','14.residual1'), ('mlp.w3','18.hidden')]:
                hooks.append(net.get_submodule(attr).register_forward_pre_hook(
                    lambda _m, a, name=name: capture(name, a[0])))
            original = net.attn.apply_rope
            def observed_apply(q, k, rope):
                capture('04.q', q); capture('05.k', k)
                result = original(q, k, rope)
                capture('07.q_rope', result[0]); capture('08.k_rope', result[1])
                return result
            net.attn.apply_rope = observed_apply
            observed = net(x, rope)
            if not torch.equal(baseline, observed):
                raise ValueError('observation changed block result')
            capture('21.output', observed)
            for hook in hooks:
                hook.remove()
            net.attn.apply_rope = original
            capture('06.v', taps['03.qkv'].reshape(1, 1029, 3, 20, 64)[:, :, 2].transpose(1, 2))
            capture('09.logits', (taps['07.q_rope'] @ taps['08.k_rope'].transpose(-1, -2)) / 8)
            capture('10.probs', taps['09.logits'].softmax(-1))
            state['periods'] = rope_net.periods
            filename = case + '.input'
            with (args.output / filename).open('xb') as stream:
                stream.write(b'S3DBLK01' + struct.pack('<7I', 1, 32, 32, 1280, 20, 5, 5120))
                stream.write(x.cpu().numpy().astype('<f4').tobytes())
                for name in PARAMETERS:
                    stream.write(state[name].cpu().numpy().astype('<f4').tobytes())
            order = sorted(taps)
            cases.append(dict(prefix=case, input=filename, block=index, input_boundary=key,
                              order=order, shapes={key:list(taps[key].shape) for key in order}))
            for name in order:
                tensors[case + '.' + name] = taps[name].cpu().numpy().copy()
                rules.append(dict(name=case + '.' + name, mode='float', max_abs=1e-4,
                                  relative_l2=2e-5, zero_reference_floor=1e-12))
            print(f'Captured block {index}, three repeats + observed exact', flush=True)
            del taps, state, net, baseline, observed, x, rope_net
    save_file(tensors, args.output / 'upstream.safetensors')
    boundary = 'trained blocks 0/1/22, identical original CUDA input states; isolated arithmetic, not own-intermediate chain'
    (args.output / 'rules.json').write_text(json.dumps(dict(schema_version=1, boundary=boundary, tensors=rules), indent=2)+'\n')
    manifest = dict(boundary=boundary, cases=cases, source_hashes=HASHES, device=args.device,
                    torch=torch.__version__, threads=6, dtype='float32', sdpa='math', tf32=False,
                    synthetic_weights=False, unobserved_repeats=3, unobserved_vs_observed='exact',
                    weights_sha256=digest(args.weights), input_backbone_sha256=digest(args.backbone_reference),
                    script_sha256=digest(Path(__file__)), artifacts={})
    for path in sorted(args.output.iterdir()):
        manifest['artifacts'][path.name] = digest(path)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
