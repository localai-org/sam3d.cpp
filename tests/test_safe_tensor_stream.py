from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from safe_tensor_stream import write_verified


class SafeStreamTests(unittest.TestCase):
    def test_mixed_exact_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.safetensors'
            arrays = {'weight': np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
                      'index': np.array([2**54 + 1], dtype=np.int64),
                      'scalar': np.array(-0., dtype=np.float32)}
            specs = {k: ('I64' if v.dtype == np.int64 else 'F32', v.shape) for k, v in arrays.items()}
            write_verified(path, specs, arrays.__getitem__)
            with safe_open(path, framework='numpy') as stream:
                for name, value in arrays.items():
                    self.assertEqual(stream.get_tensor(name).tobytes(), value.tobytes())
            with self.assertRaises(FileExistsError):
                write_verified(path, specs, arrays.__getitem__)
            self.assertFalse(list(Path(directory).glob('.extract-*')))

    def test_reject_invalid_data_and_changed_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.safetensors'
            specs = {'x': ('F32', (1,))}
            for value in [np.array([np.nan], dtype=np.float32), np.zeros(2, np.float32), np.zeros(1, np.float64)]:
                with self.assertRaises(ValueError):
                    write_verified(path, specs, lambda _: value)
                self.assertFalse(path.exists())
            calls = iter([np.zeros(1, np.float32), np.ones(1, np.float32)])
            with self.assertRaisesRegex(ValueError, 'readback byte mismatch'):
                write_verified(path, specs, lambda _: next(calls))
            self.assertFalse(path.exists())
            self.assertFalse(list(Path(directory).glob('.extract-*')))
            path.symlink_to(Path(directory) / 'missing')
            with self.assertRaises(FileExistsError):
                write_verified(path, specs, lambda _: np.zeros(1, np.float32))
