#!/usr/bin/env python3
"""Extract verified official Body state in the isolated offline Docker reference.

All legacy parsing uses weights_only=True; no unsafe fallback or model import.
Float state is exactly upcast to F32. Integer mapping state remains I64.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from gguf_schema import (ARCHITECTURE, BODY_REVISION, DINO_REVISION,
                         CHECKPOINT_SHA256, CONFIG_SHA256, body_dino_shapes)
from convert_gguf import sha256
from safe_tensor_stream import write_verified
from preflight import check_artifact, one_match, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists():
        raise RuntimeError('run only inside the isolated Docker reference environment')
    outputs = {key: args.output / name for key, name in dict(
        backbone='body-dinov3-f32.safetensors', rest='body-other-state.safetensors',
        manifest='body-dinov3-manifest.json', report='extraction.json').items()}
    if any(os.path.lexists(path) for path in outputs.values()):
        raise FileExistsError('extraction output already exists; use a fresh output directory')
    manifest = read_json(ROOT / 'reference/sources.json')
    repo = one_match(manifest['model_repositories'], 'repository', 'facebook/sam-3d-body-dinov3')
    entries = [one_match(repo['selected_files'], 'path', name) for name in ['model.ckpt', 'model_config.yaml']]
    evidence = [check_artifact(args.model_directory, entry) for entry in entries]
    if [item['sha256'] for item in evidence] != [CHECKPOINT_SHA256, CONFIG_SHA256]:
        raise ValueError('unsupported checkpoint/config pair')
    # Git cleanliness/revisions are checked by host preflight. The container has
    # no Git executable and imports no upstream model; verify the only source
    # whose initializer is reproduced below by its exact content hash instead.
    # The one absent checkpoint tensor has a deterministic original initializer.
    init_source = ROOT / 'reference/upstream/dinov3/dinov3/models/vision_transformer.py'
    init_hash = '1af05405e19bc42188f80bfcb204b87ed1ec72af998e0bbd4658a85d7fae93f2'
    if sha256(init_source) != init_hash:
        raise ValueError('original mask-token initializer source changed')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    state = torch.load(args.model_directory / 'model.ckpt', map_location='cpu',
                       weights_only=True, mmap=True)
    if not isinstance(state, dict) or len(state) != 1127:
        raise ValueError('unexpected published Body state dictionary')
    prefix = 'backbone.encoder.'
    shapes = body_dino_shapes()
    backbone = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
    if set(backbone) != set(shapes) - {'mask_token'}:
        raise ValueError('unexpected missing or additional backbone state')
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.dtype not in [
                torch.bfloat16, torch.float32, torch.int64]:
            raise ValueError(f'unsupported state: {name}')
    for name, value in backbone.items():
        if tuple(value.shape) != shapes[name] or value.dtype != torch.bfloat16:
            raise ValueError(f'published backbone schema mismatch: {name}')
    backbone['mask_token'] = torch.zeros(shapes['mask_token'], dtype=torch.float32)
    others = {name: value for name, value in state.items() if not name.startswith(prefix)}

    def extract(path, tensors):
        specs = {name: ('I64' if value.dtype == torch.int64 else 'F32', tuple(value.shape))
                 for name, value in tensors.items()}
        def get(name):
            value = tensors[name].detach()
            return value.numpy() if value.dtype == torch.int64 else value.float().numpy()
        write_verified(path, specs, get)
        return dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size,
                    tensor_count=len(specs), readback='all tensor bytes exactly match source upcast')

    report = dict(schema_version=1, source=evidence, torch_version=torch.__version__,
                  loader='weights_only=True,mmap=True,map_location=cpu',
                  source_tensor_count=len(state), body_revision=BODY_REVISION,
                  dinov3_revision=DINO_REVISION, script_sha256=sha256(Path(__file__)),
                  initialized_tensors={'backbone.encoder.mask_token': dict(
                      shape=[1, 1280], dtype='F32', value=0,
                      reason='absent from checkpoint; original DinoVisionTransformer.init_weights uses nn.init.zeros_',
                      initializer_source_sha256=init_hash)}, artifacts={})
    report['artifacts']['backbone'] = extract(outputs['backbone'], backbone)
    print('Backbone extracted and byte-verified', flush=True)
    report['artifacts']['other_state'] = extract(outputs['rest'], others)
    if [check_artifact(args.model_directory, entry) for entry in entries] != evidence:
        raise ValueError('source artifact changed during extraction')
    conversion = dict(schema_version=1, architecture=ARCHITECTURE,
                      body_revision=BODY_REVISION, dinov3_revision=DINO_REVISION,
                      checkpoint_sha256=CHECKPOINT_SHA256, config_sha256=CONFIG_SHA256,
                      safetensors_sha256=report['artifacts']['backbone']['sha256'],
                      source_precision=['BF16'])
    for key, data in [('manifest', conversion), ('report', report)]:
        with outputs[key].open('x') as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
