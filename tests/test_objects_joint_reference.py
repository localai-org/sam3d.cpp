import hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_objects_joint import joint_check

class JointReferenceTests(unittest.TestCase):
    def test_original_identity_and_scope(self):
        m=json.loads((ROOT/'tests/fixtures/objects-joint.json').read_text())
        expected='b58ab38c14c500371687f77200deb449b80d6ed645bf9921eeeeb1ca9e1fdc79'
        self.assertEqual(m['artifacts']['regression.txt'],expected)
        self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures/objects-joint.txt').read_bytes()).hexdigest(),expected)
        self.assertEqual(len(m['cases']),7);self.assertTrue(m['observer_exact']);self.assertEqual(m['repeats'],3)
        self.assertEqual(m['cases'][-1]['args'][:2],[4096,2160]);self.assertTrue(m['cases'][-1]['official_image'])
        self.assertTrue(all(c['pointmap']=='synthetic supplied XYZ' for c in m['cases']))
        for key,name in [('script_sha256','capture_objects_joint.py'),('loader_sha256','capture_objects_image.py')]:
            self.assertEqual(m[key],hashlib.sha256((ROOT/'reference'/name).read_bytes()).hexdigest())
    def test_strict_nonfinite_and_finite_limits(self):
        x=np.array([np.nan,np.inf,-np.inf,1.],dtype=np.float32)
        self.assertTrue(joint_check('point',x,x.copy(),True)['pass'])
        for at,value in [(0,0),(1,0),(1,-np.inf),(2,np.nan),(3,1.001)]:
            y=x.copy();y[at]=value;self.assertFalse(joint_check('point',x,y,True)['pass'])
        self.assertFalse(joint_check('rgb',x,x,False)['pass'])
    def test_soft_alpha_must_match_exactly(self):
        x=np.array([1/255,127/255,128/255,1],dtype=np.float32)
        self.assertFalse(joint_check('alpha',x,np.ones_like(x))['pass'])
        y=x.copy();y[0]+=1e-7;self.assertFalse(joint_check('alpha',x,y)['pass'])
if __name__=='__main__':unittest.main()
