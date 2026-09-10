import sys
import json
import hashlib
from pathlib import Path
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from calibrate_trained_dino import limits
from check_parity import compare_array


class TrainedDinoPolicyTests(unittest.TestCase):
    def test_frozen_policy_and_each_boundary_negative_controls(self):
        path = Path(__file__).resolve().parents[1] / 'reference/trained-dino-backbone-policy-v1.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         'cdc4bd14042edb87e43c9f0655e469e572874856c29a5776cec3d6c9f4364299')
        policy = json.loads(path.read_text())
        self.assertFalse(policy['native_candidate_used'])
        self.assertEqual(len(policy['tensors']), 36)
        for rule, control in zip(policy['tensors'], policy['controls'], strict=True):
            self.assertEqual(rule['name'], control['name'])
            for key, value in limits(control).items():
                self.assertEqual(rule[key], value)
            ref = np.full(4096, max(control['reference_peak']/2, 1e-3), np.float32)
            self.assertTrue(compare_array(ref, ref, rule)['pass'])
            self.assertFalse(compare_array(ref, ref*1.001, rule)['pass'])
            spike = ref.copy(); spike[0] += 2*rule['max_abs']
            self.assertFalse(compare_array(ref, spike, rule)['pass'])

    def test_floors_and_reference_only_multiplier(self):
        control = {'pass':True, 'reference_dtype':'float32', 'max_abs':0., 'relative_l2':0.}
        self.assertEqual(limits(control), dict(max_abs=1e-4, relative_l2=2e-5, zero_reference_floor=1e-12))
        control.update(max_abs=.01, relative_l2=1e-4)
        self.assertEqual(limits(control)['max_abs'], .04)
        self.assertEqual(limits(control)['relative_l2'], .0004)
        control['relative_l2'] = .01
        with self.assertRaisesRegex(ValueError, 'unstable'):
            limits(control)

    def test_negative_controls_at_large_and_small_scales(self):
        for scale in [.1, 1., 100., 30000.]:
            reference = np.linspace(-scale, scale, 1024, dtype=np.float32)
            control = {'pass':True, 'reference_dtype':'float32', 'max_abs':scale*2e-6, 'relative_l2':2e-6}
            rule = dict(name='test', mode='float', **limits(control))
            self.assertTrue(compare_array(reference, reference.copy(), rule)['pass'])
            self.assertFalse(compare_array(reference, reference*1.001, rule)['pass'])
            spike = reference.copy()
            spike[0] += rule['max_abs']*2
            self.assertFalse(compare_array(reference, spike, rule)['pass'])
            self.assertFalse(compare_array(reference, reference[::-1].copy(), rule)['pass'])
            self.assertFalse(compare_array(reference, np.zeros_like(reference), rule)['pass'])
