import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_trained_body_branch import boundary_rule,same_bytes,POLICY_SHA,FULL_NAMES,DECODER_NAMES,validate_operation_coverage
from check_parity import sha256_file,compare_array

class TrainedBranchTests(unittest.TestCase):
    def test_complete_real_decoder_operation_coverage(self):
        names=[f'layer.{i}.decoder.{name}' for i in range(6) for name in DECODER_NAMES]+[f'other.{i}' for i in range(325)]
        validate_operation_coverage(dict(order=names),True)
        with self.assertRaises(ValueError):validate_operation_coverage(dict(order=names[1:]),True)
        with self.assertRaises(ValueError):validate_operation_coverage(dict(order=names),False)

    def test_policy_only_applies_to_backbone(self):
        path=Path(__file__).resolve().parents[1]/'reference/trained-dino-backbone-policy-v1.json'
        self.assertEqual(sha256_file(path),POLICY_SHA)
        rules={r['name'].removeprefix('case.0000.'):r for r in json.loads(path.read_text())['tensors']}
        for key,rule in rules.items():
            mapped=boundary_rule('case.0000.backbone.'+key,rules)
            self.assertEqual(mapped['max_abs'],rule['max_abs'])
            self.assertEqual(mapped['relative_l2'],rule['relative_l2'])
        for key in FULL_NAMES:
            rule=boundary_rule('case.0000.layer.5.full.'+key,rules)
            self.assertEqual(rule['relative_l2'],2e-5)
            self.assertLessEqual(rule['max_abs'],1e-3 if key in ['pred_keypoints_2d','pred_keypoints_2d_verts'] else 1e-4)
        ref=np.ones(10,np.float32)
        rule=boundary_rule('case.0000.layer.5.full.pred_vertices',rules)
        self.assertFalse(compare_array(ref,ref+.001,rule)['pass'])

    def test_original_cohort_requires_exact_dtype_shape_and_bits(self):
        a=np.array([0.,1.],np.float32)
        self.assertTrue(same_bytes(a,a.copy()))
        self.assertFalse(same_bytes(a,a.reshape(1,2)))
        self.assertFalse(same_bytes(a,a.astype(np.float64)))
        b=a.copy();b[0]=-0.
        self.assertFalse(same_bytes(a,b))
        b=a.copy();b[1]=np.nextafter(b[1],np.float32(2))
        self.assertFalse(same_bytes(a,b))
