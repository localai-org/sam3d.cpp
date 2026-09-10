import sys
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_trained_operations import NAMES, native_views, calibrated_rule
from check_parity import compare_array


class TrainedOperationTests(unittest.TestCase):
    def test_frozen_full_operation_policy_and_negative_controls(self):
        path=Path(__file__).resolve().parents[1]/'reference/trained-dino-operations-policy-v1.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         'bded7ec65145f7d94fbdb4bf1ea3731ab4e8dfc4296bedbcf9f346e65609abb8')
        policy=json.loads(path.read_text())
        self.assertFalse(policy['native_candidate_used'])
        self.assertEqual([r['name'] for r in policy['tensors']],
                         [f'block.{i}.{name}' for i in range(32) for name in NAMES])
        for rule in policy['tensors']:
            self.assertLessEqual(rule['relative_l2'],1e-3)
            ref=np.full(4096,max(1.,rule['max_abs']*10),np.float32)
            self.assertTrue(compare_array(ref,ref,rule)['pass'])
            self.assertFalse(compare_array(ref,ref*1.01,rule)['pass'])
            spike=ref.copy();spike[0]+=2*rule['max_abs']
            self.assertFalse(compare_array(ref,spike,rule)['pass'])

    def test_bounded_native_trace_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.bin'
            values = np.arange(44, dtype='<f4')
            path.write_bytes(values.tobytes())
            record = {'shapes':{name:[1,2] for name in NAMES}}
            result = native_views(path, record)
            for index, name in enumerate(NAMES):
                np.testing.assert_array_equal(result[name], values[2*index:2*index+2].reshape(1,2))
            del result
            path.write_bytes(values.tobytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'length'):
                native_views(path, record)
            record['shapes'][NAMES[0]] = [2**32]
            with self.assertRaisesRegex(ValueError, 'shape'):
                native_views(path, record)

    def test_probability_ceiling_is_not_generic_activation_limit(self):
        control = {'pass':False, 'reference_dtype':'float32', 'max_abs':.0007, 'relative_l2':.00006}
        probability = calibrated_rule(control, 1., 'block.0.10.probs')
        self.assertAlmostEqual(probability['max_abs'], .0028)
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            calibrated_rule(control, 1., 'block.0.02.norm1')
        ref = np.full(1024, .5, np.float32)
        self.assertFalse(compare_array(ref, ref*1.001, probability)['pass'])
        spike=ref.copy(); spike[0]+=.01
        self.assertFalse(compare_array(ref, spike, probability)['pass'])
        self.assertEqual(calibrated_rule(control|{'max_abs':.003}, 1., 'block.0.10.probs')['max_abs'], .01)
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            calibrated_rule(control|{'max_abs':.011}, 1., 'block.0.10.probs')
        with self.assertRaisesRegex(ValueError, 'unstable'):
            calibrated_rule(control|{'relative_l2':.001}, 1., 'block.0.10.probs')
