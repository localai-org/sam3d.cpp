#!/usr/bin/env python3
"""Inventory pinned official checkpoints inside the isolated reference container.

Never run on the host. Uses the restricted weights-only loader, never an unsafe
pickle fallback. Artifact hashes are checked before and after loading. No model
or upstream Python module is imported or instantiated.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preflight import check_artifact, read_json, one_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('sources.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists():
        raise RuntimeError('checkpoint loading requires the isolated Docker reference environment')
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = one_match(read_json(args.manifest)['model_repositories'], 'repository', args.repository)
    entry = one_match(repo['selected_files'], 'path', args.checkpoint)
    before = check_artifact(args.model_directory, entry)
    import torch
    torch.set_num_threads(1)
    # mmap plus FakeTensorMode examines metadata without materializing weights.
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        loaded = torch.load(args.model_directory / args.checkpoint,
                            map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(loaded, dict):
        raise ValueError('expected checkpoint dictionary')
    state = loaded.get('state_dict', loaded)
    if not isinstance(state, dict) or not 0 < len(state) <= 100000:
        raise ValueError('invalid tensor state dictionary')
    tensors, dtypes, metadata = {}, collections.Counter(), {}
    for name, value in state.items():
        if name == '__mixhavior__' and isinstance(value, dict):
            metadata[name] = dict(type='dict', keys=sorted(value))
            continue
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f'non-tensor state entry: {name!r}')
        dtype = str(value.dtype).removeprefix('torch.')
        count = value.numel()
        tensors[name] = dict(shape=list(value.shape), dtype=dtype,
                             layout=str(value.layout), elements=count)
        dtypes[dtype] += count
    if check_artifact(args.model_directory, entry) != before:
        raise ValueError('source artifact changed during inventory')
    result = dict(schema_version=1, repository=args.repository, revision=repo['revision'],
                  artifact=before, torch_version=torch.__version__,
                  method='weights_only=True,mmap=True,FakeTensorMode; no model execution',
                  checkpoint_keys=sorted(loaded), state_dict_key='state_dict' if 'state_dict' in loaded else None,
                  tensor_count=len(tensors), elements_by_dtype=dict(dtypes), tensors=tensors,
                  non_tensor_metadata=metadata)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ['tensors', 'checkpoint_keys']}, indent=2))


if __name__ == '__main__':
    main()
