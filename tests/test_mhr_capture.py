"""Deterministic diagnostic parser tests, not a GGUF parser fuzzer."""
import struct,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mhr_skeleton import read_output

class MHRCaptureTests(unittest.TestCase):
    def sample(self):
        parts=[b'S3DMHO01'+struct.pack('<I',46)]
        for i in range(46):
            name=f'tap.{i}'.encode();width=4 if i%2 else 8
            parts.append(struct.pack('<I',len(name))+name+struct.pack('<IQ',width,1)+struct.pack('<f' if width==4 else '<d',.25))
        return b''.join(parts)

    def parse(self,data):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'capture.bin';p.write_bytes(data);return read_output(p)

    def test_preserves_float_width(self):
        v=self.parse(self.sample());self.assertEqual(len(v),46)
        self.assertEqual(v['tap.0'].dtype.itemsize,8);self.assertEqual(v['tap.1'].dtype.itemsize,4)
        self.assertTrue(all(x[0]==.25 for x in v.values()))

    def test_truncation_and_trailing(self):
        data=self.sample()
        for bad in [b'',data[:8],data[:12],data[:-1],data+b'x']:
            with self.assertRaises(ValueError):self.parse(bad)

    def test_bad_header_shape_type(self):
        data=self.sample()
        cases=[b'BADMAGIC'+data[8:],data[:8]+struct.pack('<I',1000000)+data[12:],data[:12]+struct.pack('<I',1000000)+data[16:]]
        for width,count in [(3,1),(4,0),(8,1000000)]:cases.append(data[:21]+struct.pack('<IQ',width,count)+data[33:])
        # Equal-length duplicate name after the first record.
        cases.append(data[:45]+b'tap.0'+data[50:])
        for bad in cases:
            with self.assertRaises(ValueError):self.parse(bad)
if __name__=='__main__':unittest.main()
