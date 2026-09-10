import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_hand_head import FIELDS,compare_final

class HandHeadComparisonTests(unittest.TestCase):
    def data(self):
        values={'case.'+k:np.ones((1,3),np.float32) for k in FIELDS.values()}
        full={'case.'+k:values['case.'+v].copy() for k,v in FIELDS.items()}
        faces=np.array([[0,1,2]],np.int32);full['case.faces']=faces.astype(np.int64)
        return values,full,faces
    def test_all_final_fields(self):
        values,full,faces=self.data();checks=compare_final('case',values,full,faces)
        self.assertEqual(len(checks),13);self.assertTrue(all(c['pass'] for c in checks))
    def test_missing_or_extra_field_rejected(self):
        for extra in [False,True]:
            values,full,faces=self.data()
            if extra:full['case.surprise']=np.zeros(1,np.float32)
            else:del full['case.pred_vertices']
            with self.assertRaises(ValueError):compare_final('case',values,full,faces)
    def test_deformed_vertex_and_topology_errors_fail(self):
        values,full,faces=self.data();values['case.map.90.vertices'][0,2]+=1;faces[0,1]=2
        checks=compare_final('case',values,full,faces)
        self.assertEqual({c['name'] for c in checks if not c['pass']},{'case.full.pred_vertices','case.full.faces'})
    def test_nonfinite_or_shape_error_fails(self):
        for value in [np.full((1,3),np.nan,np.float32),np.ones((3,1),np.float32)]:
            values,full,faces=self.data();values['case.map.91.joints']=value
            self.assertFalse(all(c['pass'] for c in compare_final('case',values,full,faces)))
