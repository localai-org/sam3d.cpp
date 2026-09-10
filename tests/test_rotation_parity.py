import sys
import unittest
import json
import hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rotation_parity import compare_euler

class EulerParityTests(unittest.TestCase):
    def test_frozen_original_controls(self):
        root=Path(__file__).resolve().parents[1]
        fixture=json.loads((root/'reference/rotation-policy-v1.json').read_text())
        self.assertEqual(fixture['policy_code_sha256'],hashlib.sha256((root/'scripts/rotation_parity.py').read_bytes()).hexdigest())
        self.assertFalse(fixture['uses_native_output_for_calibration'])
        self.assertEqual(len(fixture['cases']),6)
        failed_raw=0
        for case in fixture['cases']:
            result=compare_euler(np.array(case['cpu'],dtype=np.float32),np.array(case['cuda'],dtype=np.float32))
            self.assertTrue(result['pass']);self.assertEqual(result['max_abs'],case['max_abs'])
            failed_raw+=result['raw_euler_relative_l2']>2e-5
        self.assertEqual(failed_raw,5)

    def test_upstream_rounding_control(self):
        a=np.array([[.00383416,.0023403168,.00049818]],dtype=np.float32)
        b=a.copy();b[0,1]=np.float32(.0023401976)
        result=compare_euler(a,b)
        self.assertGreater(result['raw_euler_relative_l2'],2e-5)
        self.assertTrue(result['pass'])
        self.assertLess(result['relative_l2'],1e-7)

    def test_zero_and_small_coordinates(self):
        zero=np.zeros((2,3),dtype=np.float32)
        self.assertEqual(compare_euler(zero,zero)['relative_l2'],0)
        self.assertTrue(compare_euler(zero,zero+np.float32(1e-7))['pass'])

    def test_material_errors_fail_each_gate(self):
        zero=np.zeros((1,3),dtype=np.float32)
        self.assertFalse(compare_euler(zero,zero+np.float32(.001))['pass'])
        r=compare_euler(zero,zero+np.float32(5e-5))
        self.assertLess(r['max_abs'],r['max_abs_limit'])
        self.assertGreater(r['relative_l2'],r['relative_l2_limit'])
        self.assertFalse(r['pass'])

    def test_wrapping_axes_and_units_not_hidden(self):
        a=np.array([[.1,.3,.5]],dtype=np.float32)
        for b in (a[:,::-1].copy(),-a,a*180/np.float32(np.pi),a+np.float32(2*np.pi)):
            self.assertFalse(compare_euler(a,b)['pass'])

    def test_reject_malformed(self):
        a=np.zeros((1,3),dtype=np.float32)
        for b in (a.astype(np.float64),np.zeros((0,3),dtype=np.float32),a.reshape(3),np.ones((1,3),dtype=np.float32)*np.inf):
            with self.assertRaises(ValueError):compare_euler(a,b)

if __name__=='__main__':unittest.main()
