import hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_objects_preprocess import check

class PreprocessReferenceTests(unittest.TestCase):
    def test_original_identity_and_scope(self):
        m=json.loads((ROOT/'tests/fixtures/objects-preprocess.json').read_text())
        sha='345842d6f6e32664bd9be0d7c31279df5d0938121985b3d0b4b487bdb54e4940'
        self.assertEqual(m['artifacts']['regression.txt'],sha)
        self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures/objects-preprocess.txt').read_bytes()).hexdigest(),sha)
        self.assertEqual(len(m['cases']),9);self.assertTrue(m['observer_exact']);self.assertEqual(m['repeats'],3)
        self.assertEqual(m['device'],'cpu');self.assertIn('supplied synthetic XYZ',m['scope'])
        self.assertEqual(m['cases'][-1]['args'][:2],[4096,2160]);self.assertTrue(m['cases'][-1]['official_image'])
        self.assertEqual(m['script_sha256'],hashlib.sha256((ROOT/'reference/capture_objects_preprocess.py').read_bytes()).hexdigest())
        for name,sha in m['loaders'].items():self.assertEqual(sha,hashlib.sha256((ROOT/'reference'/name).read_bytes()).hexdigest())
        self.assertEqual({o[0] for c in m['cases'] for o in c['normalizers']},set(range(8)))
        self.assertEqual({c['args'][7] for c in m['cases']},{0,1})
        self.assertEqual({c['args'][8] for c in m['cases']},{0,1})
        self.assertEqual(m['pipeline_sha256'],'55b69917ff0bb8b5ca6e918f516d75e9e7561a6c9396120abde863bf5757aa72')
    def test_nonfinite_categories_and_limits(self):
        x=np.array([np.nan,np.inf,-np.inf,1],dtype=np.float32)
        for name in ['01.object.pointmap','04.apply.2.pad','04.apply.5.output','90.final.rgb_pointmap']:
            self.assertTrue(check(name,x,x.copy())['pass'])
            for i,value in [(0,0),(1,-np.inf),(2,np.nan),(3,1.001)]:
                y=x.copy();y[i]=value;self.assertFalse(check(name,x,y)['pass'])
        with self.assertRaises(ValueError):check('90.final.image',x,x)
        for name in ['01.object.scale','05.full.shift','90.final.pointmap_scale']:
            with self.assertRaises(ValueError):check(name,x,x)
        with self.assertRaises(ValueError):check('90.final.pointmap',x,x[:3])
    def test_exact_masks_and_independent_rgb_policy(self):
        x=np.array([.5,1.],dtype=np.float32);y=x.copy();y[0]+=1e-7
        for name in ['00.mask','04.apply.1.output','04.apply.4.pad','90.final.rgb_image_mask']:
            self.assertFalse(check(name,x,y)['pass'])
        y=x+2e-6;self.assertFalse(check('90.final.image',x,y)['pass'])
        y=x+2e-4;self.assertFalse(check('90.final.pointmap',x,y)['pass'])
if __name__=='__main__':unittest.main()
