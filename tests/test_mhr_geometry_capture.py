"""Small native geometry diagnostic parser checks; no GGUF loading or fuzzing."""
import struct,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mhr_geometry import read_output

class GeometryCaptureTests(unittest.TestCase):
    def sample(self,count=19):
        parts=[b'S3DMGO01'+struct.pack('<I',count)]
        for i in range(count):
            name=f'tap.{i}'.encode();parts.append(struct.pack('<I',len(name))+name+struct.pack('<Qf',1,.5))
        return b''.join(parts)

    def parse(self,data):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'capture';p.write_bytes(data);return read_output(p)

    def test_both_corrective_modes(self):
        for count in [19,25]:
            r=self.parse(self.sample(count));self.assertEqual(len(r),count);self.assertTrue(all(v.dtype.itemsize==4 and v[0]==.5 for v in r.values()))

    def test_truncated_and_trailing(self):
        data=self.sample()
        for bad in [b'',data[:8],data[:12],data[:-1],data+b'x']:
            with self.assertRaises(ValueError):self.parse(bad)

    def test_invalid_metadata(self):
        data=self.sample()
        for bad in [b'BADMAGIC'+data[8:],data[:8]+struct.pack('<I',20)+data[12:],data[:12]+struct.pack('<I',1000000)+data[16:],
                    data[:21]+struct.pack('<Q',0)+data[29:],data[:21]+struct.pack('<Q',1000001)+data[29:],data[:37]+b'tap.0'+data[42:]]:
            with self.assertRaises(ValueError):self.parse(bad)
if __name__=='__main__':unittest.main()
