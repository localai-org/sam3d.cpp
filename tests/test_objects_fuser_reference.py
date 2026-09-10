import hashlib,json,struct,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_objects_fuser import check
from run_objects_ssi import read_capture
class FuserReferenceTests(unittest.TestCase):
    def test_original_identity_and_scope(self):
        m=json.loads((ROOT/'tests/fixtures/objects-fuser.json').read_text());sha='7325ac82d1f8fd9c6c3b4d4a57ce4d260450d0545a322cec5be43fa6a26406e4'
        self.assertEqual(m['artifacts']['regression.txt'],sha)
        self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures/objects-fuser.txt').read_bytes()).hexdigest(),sha)
        self.assertEqual(m['script_sha256'],hashlib.sha256((ROOT/'reference/capture_objects_fuser.py').read_bytes()).hexdigest())
        self.assertEqual(len(m['cases']),7);self.assertTrue(m['observer_exact']);self.assertEqual(m['repeats'],3)
        self.assertIn('supplied embeddings and synthetic state',m['scope'])
        self.assertEqual(m['cases'][-1]['dims'],[768,1024]);self.assertEqual([x[1] for x in m['cases'][-1]['modalities']],[1024,1024])
        self.assertEqual({c['pre_norm'] for c in m['cases']},{True,False});self.assertEqual({c['random_position'] for c in m['cases']},{True,False})
    def test_strict_finite_and_error_metrics(self):
        x=np.array([.5,1.],dtype=np.float32)
        self.assertTrue(check('90.output',x,x.copy())['pass'])
        for v in [np.nan,np.inf,-np.inf,1.001]:
            y=x.copy();y[1]=v;self.assertFalse(check('90.output',x,y)['pass'])
        y=x.copy();y[0]+=1e-7;self.assertFalse(check('10.modality.0.input',x,y)['pass'])
        y=x+2e-4;self.assertFalse(check('10.modality.0.positioned',x,y)['pass'])
    def test_bounded_extended_capture_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'capture.bin';data=b'S3DST001'+struct.pack('<I',33)
            for i in range(33):
                k=str(i).encode();data+=struct.pack('<I',len(k))+k+struct.pack('<Qf',1,float(i))
            p.write_bytes(data)
            with self.assertRaises(ValueError):read_capture(p)
            self.assertEqual(len(read_capture(p,max_tensors=33)),33)
            for n in [0,32,257]:
                with self.assertRaises(ValueError):read_capture(p,max_tensors=n)
            p.write_bytes(data[:-1])
            with self.assertRaises(ValueError):read_capture(p,max_tensors=33)
if __name__=='__main__':unittest.main()
