#!/usr/bin/env python3
"""Trained Body backbone on the official dancing crop; NOT full Body inference.

Original pinned factory and Body wrapper, three unobserved repeats and observed
forward. F32 is explicit, distinct from the published BF16 execution setting.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import types

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from capture_dino_schema import HASHES, load_factory
from capture_dino_backbone import BODY_PATH, BODY_HASH, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--body-upstream', type=Path, required=True)
    parser.add_argument('--extraction', type=Path, required=True)
    parser.add_argument('--image-reference', type=Path, required=True)
    parser.add_argument('--image-reference-manifest',type=Path,help='completed original benchmark identifying a pinned alternate image and normalized input')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--operation-traces', action='store_true', help='stream actual operations from all 32 blocks')
    parser.add_argument('--bf16',action='store_true',help='original precision helper and BF16 backbone input; math SDPA')
    parser.add_argument('--attention',choices=['math','auto'],default='math')
    parser.add_argument('--compile', action='store_true', help='compile the actual original Body wrapper, fullgraph; require observer neutrality')
    args = parser.parse_args()
    if args.compile and args.operation_traces:
        raise ValueError('compiled controls support stage taps only, not host-side operation observers')
    if args.operation_traces and args.attention!='math':
        raise ValueError('internal softmax observation requires math attention; automatic attention supports stage observations only')
    if not 1 <= args.threads <= 24:
        raise ValueError('invalid CPU thread count')
    report = json.loads((args.extraction / 'extraction.json').read_text())
    artifact = report['artifacts']['backbone']
    weights = args.extraction / artifact['file']
    if digest(weights) != artifact['sha256'] or artifact['sha256'] != '98afde8ac5c13c68f3b7c7baf7cc8fa525df901fe1cec44b57d018fba11d53cc':
        raise ValueError('verified extraction identity mismatch')
    image_hash = 'cbac6e6842b9c7508f4fbdd3cbf5ce87b83a0a55c35eca94dd2b6a30f8d25c2c'
    image_label='dancing crop'
    if args.image_reference_manifest:
        from body_image_cases import CASES
        im=json.loads(args.image_reference_manifest.read_text())
        if im.get('status')!='complete' or im.get('source',{}).get('photo_sha256') not in {v[1] for v in CASES.values()}:
            raise ValueError('alternate input requires a completed original benchmark on a pinned official photo')
        image_hash=im['normalized_input_sha256'];image_label='official photograph '+im['source']['photo_sha256']
    if digest(args.image_reference) != image_hash:
        raise ValueError('original normalized dancing crop reference mismatch')
    path = args.body_upstream / BODY_PATH
    if digest(path) != BODY_HASH:
        raise ValueError('Body wrapper source mismatch')
    if any(args.output.iterdir()):
        raise FileExistsError('capture requires an empty output directory')
    factory = load_factory(args.upstream)
    spec = importlib.util.spec_from_file_location('original_trained_body_backbone', path)
    body = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(body)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.device('meta'):
        net = factory(pretrained=False).eval()
    with safe_open(weights, framework='pt') as safe:
        state = {name: safe.get_tensor(name).to(args.device) for name in safe.keys()}
    net.load_state_dict(state, strict=True, assign=True)
    del state
    with safe_open(args.image_reference, framework='numpy') as safe:
        image_np = safe.get_tensor('case.0000.prepare.normalized_rgb').copy()
    image = torch.from_numpy(image_np).to(args.device)
    if args.bf16:
        from functools import partial
        utility=args.body_upstream/'sam_3d_body/models/optim/fp16_utils.py'
        if digest(utility)!='90e25559a99fa341666c9c5da5b7e91d679ffdfab635aed9e9cf21d8f43c107d':raise ValueError('precision helper source mismatch')
        spec=importlib.util.spec_from_file_location('original_precision',utility)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        net.apply(partial(helper.convert_to_fp16_safe,dtype=torch.bfloat16));net.rope_embed.to(torch.bfloat16)
        image=image.to(torch.bfloat16)
    wrapper = types.SimpleNamespace(encoder=net)
    compiled_graphs = {}
    def original_forward(value):
        return body.Dinov3Backbone.forward(wrapper, value)
    def make_forward(label):
        if not args.compile:
            return original_forward
        from torch._dynamo.utils import counters
        counters.clear()
        # This is the wrapper's real get_intermediate_layers entry point, not
        # encoder.forward (which Body never invokes). No eager fallback.
        forward = torch.compile(original_forward, mode='reduce-overhead', fullgraph=True)
        def invoke(value):
            torch.compiler.cudagraph_mark_step_begin()
            result = forward(value)
            compiled_graphs[label] = int(counters['stats']['unique_graphs'])
            if not compiled_graphs[label]:
                raise RuntimeError('no full compiled graph executed')
            return result
        return invoke
    print('Loaded trained '+('BF16' if args.bf16 else 'F32')+' backbone', flush=True)
    with torch.no_grad():
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backends=[SDPBackend.MATH] if args.attention=='math' else [SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.MATH]
        with sdpa_kernel(backends):
            forward = make_forward('unobserved')
            baseline = forward(image).clone()
            for repeat in range(2):
                if not torch.equal(baseline, forward(image)):
                    raise ValueError('uninstrumented trained reference is not repeatable')
                print(f'Unobserved repeat {repeat + 2} exact', flush=True)
            taps = {}
            operation_observer = None
            if args.operation_traces:
                from observe_trained_dino import TrainedDinoObserver
                operation_observer = TrainedDinoObserver(net, args.output/'operations')
            def capture(name, value):
                # Keep taps on device until the complete original forward has
                # finished. Host copies inside hooks cause graph breaks and
                # would not be a valid compiled whole-encoder control.
                taps[name] = value if args.compile else value.detach().float().cpu().numpy().copy()
            hooks = [net.patch_embed.register_forward_hook(lambda _m, _a, v: capture('00.patch', v.flatten(1, 2))),
                     net.norm.register_forward_hook(lambda _m, _a, v: capture('03.norm', v))]
            for index, block in enumerate(net.blocks):
                key = f'02.block.{index:02d}'
                hooks.append(block.register_forward_hook(lambda _m, _a, v, key=key: capture(key, v)))
            original = net.prepare_tokens_with_masks
            def observed_tokens(*a, **kw):
                result = original(*a, **kw)
                capture('01.tokens', result[0])
                return result
            net.prepare_tokens_with_masks = observed_tokens
            if args.compile:
                # Discard the unobserved graph's guards before instrumenting;
                # both executions must independently compile in full.
                torch._dynamo.reset()
            forward = make_forward('observed')
            observed = forward(image)
            if not torch.equal(baseline, observed):
                # A compiled graph can fuse differently when its intermediate
                # tensors become outputs. Preserve the rejected experiment,
                # never publish those taps as an observer-neutral fixture.
                expected=baseline.float().cpu().numpy()
                actual=observed.float().cpu().numpy()
                delta=actual.astype(np.float64)-expected.astype(np.float64)
                save_file({'unobserved':expected,'observed':actual},args.output/'rejected-observation.safetensors')
                rejection=dict(status='rejected',reason='instrumentation changed output',
                    compile=args.compile,compiled_graphs=compiled_graphs,
                    max_abs=float(np.max(np.abs(delta))),
                    relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected.astype(np.float64)),1e-12)),
                    mismatched_elements=int(np.count_nonzero(expected!=actual)),
                    capture_script_sha256=digest(Path(__file__)),
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    artifacts={'rejected-observation.safetensors':digest(args.output/'rejected-observation.safetensors')})
                (args.output/'rejected.json').write_text(json.dumps(rejection,indent=2)+'\n')
                raise ValueError('trained backbone instrumentation changed output')
            if operation_observer is not None:
                operation_observer.close()
            capture('04.features', observed)
            for hook in hooks:
                hook.remove()
            net.prepare_tokens_with_masks = original
            if args.compile:
                taps = {name: value.detach().float().cpu().numpy().copy() for name, value in taps.items()}
    order = sorted(taps)
    with (args.output / 'case.0000.input').open('xb') as stream:
        stream.write(b'S3DBBN01' + struct.pack('<9I', 1, 512, 512, 16, 1280, 20, 5120, 32, 4))
        stream.write(image_np.astype('<f4').tobytes())
    save_file({'case.0000.' + k: v for k, v in taps.items()}, args.output / 'upstream.safetensors')
    save_file({'features': baseline.float().cpu().numpy()}, args.output / 'unobserved.safetensors')
    boundary = 'trained '+('BF16' if args.bf16 else 'F32')+' Body backbone on original normalized '+image_label+'; not full Body inference'
    rules = dict(schema_version=1, boundary=boundary, tensors=[dict(
        name='case.0000.' + k, mode='float', max_abs=1e-4, relative_l2=2e-5, zero_reference_floor=1e-12) for k in order])
    (args.output / 'rules.json').write_text(json.dumps(rules, indent=2) + '\n')
    manifest = dict(boundary=boundary, synthetic_weights=False, extraction_sha256=digest(args.extraction / 'extraction.json'),
                    safetensors_sha256=artifact['sha256'], source_hashes=HASHES, body_source_hashes={BODY_PATH: BODY_HASH},
                    device=args.device, dtype='bfloat16' if args.bf16 else 'float32', sdpa=args.attention, tf32=False, threads=args.threads,
                    torch=torch.__version__, unobserved_repeats=3, unobserved_vs_observed='exact',
                    compile='body-wrapper-fullgraph' if args.compile else 'none', compiled_graphs=compiled_graphs,
                    deterministic_algorithms=True,
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    image_reference_sha256=image_hash, capture_script_sha256=digest(Path(__file__)),
                    cases=[dict(prefix='case.0000', input='case.0000.input', order=order,
                                shapes={key: list(taps[key].shape) for key in order})], artifacts={})
    for path in sorted(args.output.iterdir()):
        if path.is_file(): manifest['artifacts'][path.name] = digest(path)
    if args.operation_traces:
        manifest['operation_manifest_sha256'] = digest(args.output/'operations/manifest.json')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Captured {len(taps)} trained backbone boundaries', flush=True)


if __name__ == '__main__':
    main()
