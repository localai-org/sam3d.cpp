import struct,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_body_api import FIELDS,read_output
class BodyAPIOutputTests(unittest.TestCase):
    def encoded(self):
        out=bytearray(b'S3DOUT01'+struct.pack('<I',len(FIELDS)))
        for name,(_,shape) in FIELDS.items():
            data=np.ones(shape,dtype='<i4' if name=='faces' else '<f4');encoded=name.encode('ascii')
            out+=struct.pack('<I',len(encoded))+encoded+struct.pack('<IIQ',2 if name=='faces' else 1,len(shape),data.size)
            out+=struct.pack('<'+'Q'*len(shape),*shape)+data.tobytes()
        return out
    def test_all_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'output';p.write_bytes(self.encoded());v=read_output(p)
            self.assertEqual(set(v),set(FIELDS))
            self.assertEqual(v['joint_rotations'].shape,(1,127,3,3))
            self.assertEqual(v['faces'].dtype,np.int32)
    def test_rejection(self):
        raw=self.encoded();bad_count=raw.copy();struct.pack_into('<I',bad_count,8,18)
        bad_name=raw.copy();struct.pack_into('<I',bad_name,12,100000)
        bad_type=raw.copy();struct.pack_into('<I',bad_type,16+len('vertices'),2)
        cases=[b'',raw[:7],raw[:-1],raw+b'trailing',bad_count,bad_name,bad_type]
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'output'
            for value in cases:
                p.write_bytes(value)
                with self.assertRaises(ValueError):read_output(p)
