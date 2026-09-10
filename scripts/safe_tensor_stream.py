"""Bounded-memory safe tensor publication with independent per-tensor readback."""
import json
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
from safetensors import safe_open

TYPES = {'F32': np.dtype('<f4'), 'I64': np.dtype('<i8')}


def write_verified(destination, specs, tensor):
    """tensor(name) returns a fresh array; at most one tensor is copied at once.

    specs maps names to (safe dtype, shape). Output is never overwritten, and
    safetensors itself must accept every tensor and reproduce its exact bytes.
    """
    destination = Path(destination)
    if destination.suffix != '.safetensors':
        raise ValueError('expected safetensors destination')
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if not 0 < len(specs) <= 100000:
        raise ValueError('invalid tensor count')
    header, offset = {}, 0
    for name, (dtype, shape) in sorted(specs.items()):
        if not isinstance(name, str) or not name or name == '__metadata__' or len(name) > 512:
            raise ValueError('invalid tensor name')
        if dtype not in TYPES or len(shape) > 8 or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError('invalid tensor dtype or shape')
        size = math.prod(shape) * TYPES[dtype].itemsize
        if size > 512 * 1024**2:
            raise ValueError('per-tensor memory limit')
        header[name] = dict(dtype=dtype, shape=list(shape), data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    if len(encoded) > 4 * 1024**2:
        raise ValueError('header too large')

    def get(name):
        dtype, shape = specs[name]
        value = tensor(name)
        if value.dtype != TYPES[dtype] or value.shape != tuple(shape):
            raise ValueError(f'tensor dtype/shape mismatch: {name}')
        if dtype == 'F32' and not np.isfinite(value).all():
            raise ValueError(f'nonfinite tensor: {name}')
        return value if value.flags.c_contiguous else np.ascontiguousarray(value)

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.extract-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(struct.pack('<Q', len(encoded)))
            stream.write(encoded)
            for name in sorted(specs):
                value = get(name)
                stream.write(memoryview(value).cast('B'))
                del value
            if stream.tell() != 8 + len(encoded) + offset:
                raise ValueError('output length mismatch')
            stream.flush()
            os.fsync(stream.fileno())
        with safe_open(temporary, framework='numpy') as safe:
            if set(safe.keys()) != set(specs):
                raise ValueError('readback tensor set mismatch')
            for name in sorted(specs):
                expected, actual = get(name), safe.get_tensor(name)
                if expected.dtype != actual.dtype or expected.shape != actual.shape or not np.array_equal(
                        expected.reshape(-1).view('u1'), actual.reshape(-1).view('u1')):
                    raise ValueError(f'readback byte mismatch: {name}')
                del expected, actual
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
