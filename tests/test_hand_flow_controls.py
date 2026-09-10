import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_hand_flow_controls import compare
from run_body_flow import has_box_outputs

class HandFlowControlTests(unittest.TestCase):
    def test_hand_image_final_heads_are_required(self):
        manifest={'learned_sam_checkpoint_loaded':True,'hand_branch':True}
        self.assertFalse(has_box_outputs(manifest,{}))
        self.assertTrue(has_box_outputs(manifest,{'image_pipeline':{}}))
        manifest['hand_branch']=False
        self.assertTrue(has_box_outputs(manifest,{'image_pipeline':{}}))
        manifest['learned_sam_checkpoint_loaded']=False
        self.assertFalse(has_box_outputs(manifest,{'image_pipeline':{}}))
    def test_units_keep_existing_limits(self):
        ref=np.ones((1,3),np.float32);candidate=ref+np.float32(.0005)
        self.assertTrue(compare('case.layer.0.20.pixels',ref*100,candidate*100)['pass'] is False)
        # Same relative error bound still applies to pixels; a large reference
        # isolates the absolute unit distinction without changing that bound.
        ref=np.full((1,3),100,np.float32);candidate=ref+np.float32(.0005)
        self.assertTrue(compare('case.layer.0.20.pixels',ref,candidate)['pass'])
        self.assertFalse(compare('case.layer.0.pose.mhr.vertices_cm',ref,candidate)['pass'])
    def test_bad_shape_or_nonfinite_fails(self):
        ref=np.ones((1,3),np.float32)
        for candidate in [ref.T,np.full_like(ref,np.nan)]:
            self.assertFalse(compare('case.layer.0.00.tokens',ref,candidate)['pass'])
    def test_original_failures_remain_failures(self):
        ref=np.full((1,3),1000,np.float32);candidate=ref+np.float32(.005)
        self.assertFalse(compare('case.layer.0.03.vertex_pixels',ref,candidate)['pass'])
