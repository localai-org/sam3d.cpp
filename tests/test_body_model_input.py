"""The GGUF runner must extract pixels, never fixture weights or hidden states."""
import struct,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_body_model import extract_image

class BodyModelInputTests(unittest.TestCase):
    def payload(self,magic=b'S3DRGB03',width=2,height=3,stride=6):
        return magic+b'\x91'*(24*4+4+54*4)+struct.pack('<3I',width,height,stride)+struct.pack('<8f',0,0,2,3,10,10,1,1.5)+bytes(range(18))
    def test_only_rgb_geometry_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'original';target=Path(directory)/'image'
            for magic in [b'S3DRGB02',b'S3DRGB03']:
                data=self.payload(magic);source.write_bytes(data+b'weights_and_hidden_states')
                extract_image(source,target)
                self.assertEqual(target.read_bytes(),b'S3DIMG01'+data[8+24*4+4+54*4:])
                with self.assertRaises(FileExistsError):extract_image(source,target)
                target.unlink()
    def test_truncation_and_extents(self):
        data=self.payload()
        cases=[data[:i] for i in [0,7,8,100,len(data)-1]]
        cases += [self.payload(magic=b'S3DRGB01'),self.payload(width=0),self.payload(stride=5),self.payload(width=32767),self.payload(width=10000,height=10000,stride=30000)]
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'original';target=Path(directory)/'image'
            for value in cases:
                source.write_bytes(value)
                with self.assertRaises(ValueError):extract_image(source,target)
                self.assertFalse(target.exists())
