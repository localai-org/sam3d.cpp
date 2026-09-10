import hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_objects_point_condition import check
class PointConditionReferenceTests(unittest.TestCase):
    def test_original_identity_and_scope(self):
        m=json.loads((ROOT/'tests/fixtures/point-condition.json').read_text());r=m['reference_manifest']
        for name,sha in m['artifacts'].items():self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures'/name).read_bytes()).hexdigest(),sha)
        self.assertEqual(m['packager_sha256'],hashlib.sha256((ROOT/'scripts/make_point_condition_regression.py').read_bytes()).hexdigest())
        self.assertEqual(r['script_sha256'],hashlib.sha256((ROOT/'reference/capture_objects_point_condition.py').read_bytes()).hexdigest())
        self.assertEqual(r['pointpatch_loader_sha256'],hashlib.sha256((ROOT/'reference/original_pointpatch.py').read_bytes()).hexdigest())
        self.assertTrue(r['observer_exact']);self.assertEqual(r['repeats'],3);self.assertEqual(r['neural_device'],'cpu')
        self.assertIn('explicit point-only config and synthetic weights',r['scope'])
        self.assertEqual(len(r['cases']),2);self.assertEqual([c['preprocess_case'] for c in r['cases']],[0,2])
    def test_original_domain_failure_is_preserved(self):
        r=json.loads((ROOT/'tests/fixtures/point-condition.json').read_text())['reference_manifest'];bad=r['invalid_configurations']
        self.assertEqual(len(bad),1);self.assertEqual(bad[0]['case'],1);self.assertGreater(bad[0]['output_nonfinite'],0)
        self.assertTrue(any(c['safe_input_finite'] and c['remapped_nonfinite']>0 for c in bad[0]['encoder_calls']))
        self.assertEqual(r['cases'][1]['remap'],1);self.assertTrue(r['cases'][1]['drop_scene'])
    def test_final_neural_output_requires_finite_values(self):
        a=np.array([.25,1.],dtype=np.float32)
        self.assertTrue(check('90.result',a,a.copy())['pass'])
        for bad in [np.nan,np.inf,-np.inf,1.001]:
            b=a.copy();b[1]=bad;self.assertFalse(check('90.result',a,b)['pass'])
        x=np.array([np.nan,1],dtype=np.float32)
        self.assertTrue(check('00.prepared.pointmap',x,x.copy())['pass'])
        self.assertFalse(check('90.result',x,x.copy())['pass'])
if __name__=='__main__':unittest.main()
