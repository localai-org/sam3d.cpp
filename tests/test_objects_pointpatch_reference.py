import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_objects_pointpatch import check

class PointPatchTests(unittest.TestCase):
    def test_fixture_and_capture_identity(self):
        meta = json.loads((ROOT/'tests/fixtures/objects-pointpatch.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures/objects-pointpatch.txt').read_bytes()).hexdigest(), meta['artifacts']['regression.txt'])
        for key, path in [('script_sha256','capture_objects_pointpatch.py'), ('loader_sha256','original_pointpatch.py')]:
            self.assertEqual(hashlib.sha256((ROOT/'reference'/path).read_bytes()).hexdigest(), meta[key])
        self.assertTrue(meta['observer_exact'])
        self.assertEqual(len(meta['cases']), 6)
        self.assertTrue(all(len(c['order']) == 27 for c in meta['cases']))
        self.assertEqual(meta['max_abs_limit'], 1e-4)
        self.assertEqual(meta['relative_l2_limit'], 2e-5)

    def test_matching_invalid_categories(self):
        x=np.array([np.nan,np.inf,-np.inf,1],dtype=np.float32)
        self.assertTrue(check('invalid',x,x.copy(),True)['pass'])

    def test_invalid_categories_cannot_change(self):
        x=np.array([np.nan,np.inf,-np.inf,1],dtype=np.float32)
        for index,replacement in [(0,0),(0,np.inf),(1,-np.inf),(2,np.nan),(3,np.nan)]:
            y=x.copy();y[index]=replacement
            self.assertFalse(check('invalid',x,y,True)['pass'])

    def test_finite_error_not_hidden_by_invalid_mask(self):
        x=np.array([np.nan,np.inf,-np.inf,1],dtype=np.float32);y=x.copy();y[-1]=1.01
        self.assertFalse(check('invalid',x,y,True)['pass'])

    def test_forbidden_boundary_rejects_even_matching_nan(self):
        for value in [np.nan,np.inf,-np.inf]:
            x=np.array([value,1],dtype=np.float32)
            self.assertFalse(check('final',x,x.copy(),False)['pass'])

    def test_exact_and_shape_contract(self):
        x=np.ones(3,dtype=np.float32);y=x.copy();y[-1]+=1e-6
        self.assertFalse(check('exact',x,y,False,True)['pass'])
        with self.assertRaises(ValueError):check('shape',x,x.reshape(1,3),False)

if __name__=='__main__':unittest.main()
