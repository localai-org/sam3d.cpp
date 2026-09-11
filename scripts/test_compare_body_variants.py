"""Regression checks for reported distances, rotations and topology validation."""
import unittest
import numpy as np
from compare_body_variants import compare


class ComparisonTest(unittest.TestCase):
    def fixture(self):
        transforms = np.zeros((127, 8), dtype=np.float32)
        transforms[:, 6:8] = 1
        return {
            'faces': np.array([[0, 1, 2]], dtype=np.int32),
            'vertices': np.zeros((3, 3), dtype=np.float32),
            'joints': np.zeros((127, 3), dtype=np.float32),
            'keypoints': np.zeros((70, 3), dtype=np.float32),
            'camera_translation': np.zeros(3, dtype=np.float32),
            'keypoints_pixels': np.zeros((70, 2), dtype=np.float32),
            'joint_transforms': transforms,
        }

    def test_identity_and_translation(self):
        base = self.fixture()
        self.assertTrue(compare(base, base)['exact'])
        candidate = {k: v.copy() for k, v in base.items()}
        candidate['joints'][1:, 0] = 0.125
        candidate['joints'][0] = 900  # Artificial world joint must be excluded.
        candidate['camera_translation'][0] = -0.125
        result = compare(base, candidate)
        self.assertEqual(result['joints_distance_mm']['mean'], 125)
        self.assertEqual(result['pelvis_aligned_joints_mm']['max'], 0)
        self.assertEqual(result['camera_relative_joints_mm']['max'], 0)
        self.assertEqual(result['joint_rotation_degrees']['max'], 0)

    def test_rotation_sign_and_angle(self):
        base = self.fixture()
        candidate = {k: v.copy() for k, v in base.items()}
        candidate['joint_transforms'][:, 6] = -1
        self.assertEqual(compare(base, candidate)['joint_rotation_degrees']['max'], 0)
        candidate['joint_transforms'][1:, 3] = 1
        candidate['joint_transforms'][1:, 6] = 0
        self.assertAlmostEqual(compare(base, candidate)['joint_rotation_degrees']['mean'], 180)

    def test_invalid_results(self):
        base = self.fixture()
        for key, value in [('faces', 2), ('vertices', np.nan)]:
            candidate = {k: v.copy() for k, v in base.items()}
            candidate[key].flat[0] = value
            with self.assertRaises(ValueError):
                compare(base, candidate)


if __name__ == '__main__':
    unittest.main()
