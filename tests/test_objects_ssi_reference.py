import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_objects_ssi import read_capture,NONFINITE
from run_objects_pointpatch import check

class SSIReferenceTests(unittest.TestCase):
    def test_original_fixture_identity(self):
        meta=json.loads((ROOT/'tests/fixtures/objects-ssi.json').read_text())
        expected='878f073704d1254ba48a32fcaf98f8090757c6f128601c7533db58f5bc7f62a2'
        self.assertEqual(meta['artifacts']['regression.txt'],expected)
        self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures/objects-ssi.txt').read_bytes()).hexdigest(),expected)
        self.assertEqual(meta['pytorch3d_revision'],'75ebeeaea0908c5527e7b1e305fbc7681382db47')
        self.assertEqual(len(meta['source_hashes']),5)
        self.assertEqual(len(meta['cases']),23)
        self.assertEqual([c['args'][4] for c in meta['cases'][-8:]],list(range(8)))
        self.assertTrue(meta['observer_exact']);self.assertEqual(meta['repeats'],3)
        self.assertFalse(meta['deterministic_warn_only'])
        self.assertEqual([r['case'] for r in meta['rejections']],['empty_mask','all_invalid'])

    def test_capture_source_identity(self):
        meta=json.loads((ROOT/'tests/fixtures/objects-ssi.json').read_text())
        for key,name in [('script_sha256','capture_objects_ssi.py'),('loader_sha256','original_objects_ssi.py')]:
            self.assertEqual(meta[key],hashlib.sha256((ROOT/'reference'/name).read_bytes()).hexdigest())

    def parse(self,data):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'capture.bin';path.write_bytes(data);return read_capture(path)

    def sample(self):
        key=b'20.normalized';return b'S3DST001'+struct.pack('<II',1,len(key))+key+struct.pack('<Qf',1,2.)

    def test_keyed_capture(self):
        tensors=self.parse(self.sample());self.assertEqual(list(tensors),['20.normalized'])
        np.testing.assert_array_equal(tensors['20.normalized'],np.array([2],dtype=np.float32))

    def test_truncation_trailing_and_counts(self):
        blob=self.sample()
        for n in range(len(blob)):
            with self.assertRaises(ValueError):self.parse(blob[:n])
        for data in [blob+b'x',b'BADTAG01'+blob[8:],b'S3DST001'+struct.pack('<I',0),b'S3DST001'+struct.pack('<I',33)]:
            with self.assertRaises(ValueError):self.parse(data)

    def test_duplicate_and_oversized_capture(self):
        blob=self.sample();one=blob[12:]
        for data in [b'S3DST001'+struct.pack('<I',2)+one+one,
                     b'S3DST001'+struct.pack('<II',1,65),
                     b'S3DST001'+struct.pack('<II',1,1)+b'x'+struct.pack('<Q',2**64-1)]:
            with self.assertRaises(ValueError):self.parse(data)

    def test_invalid_points_are_not_a_blanket_nan_waiver(self):
        for key in ['10.matrix','21.scale','22.shift','30.matrix','05.computed_scale']:
            self.assertNotIn(key,NONFINITE)
            x=np.array([np.nan],dtype=np.float32)
            self.assertFalse(check(key,x,x,False)['pass'])
        x=np.array([1,np.nan,2],dtype=np.float32);y=x.copy();y[0]=np.nan
        self.assertFalse(check('20.normalized',x,y,True)['pass'])

if __name__=='__main__':unittest.main()
