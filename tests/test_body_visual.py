import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qa_body_captures import image_source

class VisualProvenanceTests(unittest.TestCase):
    def record(self):return dict(status='complete',torch='torch',cuda='cuda',device='cuda',precision='bf16',attention='auto',trained_state={'weights':'pinned'},source={'photo_sha256':'photo','bbox_xyxy':[1,2,3,4],'intrinsics_fx_fy_cx_cy':[3,3,1,1],'source_hashes':{'encoder':'pinned'},'script_sha256':'old','reference_weight_residency':'old description'},image_input_sha256='pixels')
    def test_same_neural_sources_new_capture_description(self):
        original=self.record();new=copy.deepcopy(original);new['source']['script_sha256']='new';new['source']['reference_weight_residency']='corrected description'
        self.assertEqual(image_source(original,new),'pixels')
    def test_no_different_photos_settings_weights_or_framework(self):
        original=self.record()
        for key in ['photo_sha256','bbox_xyxy','intrinsics_fx_fy_cx_cy','source_hashes']:
            new=copy.deepcopy(original);new['source'][key]='changed'
            with self.assertRaises(ValueError):image_source(original,new)
        for key in ['torch','cuda','device','precision','attention','trained_state','status']:
            new=copy.deepcopy(original);new[key]='changed'
            with self.assertRaises(ValueError):image_source(original,new)
        new=copy.deepcopy(original);del new['image_input_sha256']
        with self.assertRaises(ValueError):image_source(original,new)
