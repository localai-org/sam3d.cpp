import json
from pathlib import Path
import struct
import sys
import unittest
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_demo_reference import png_rgb, ORIGINAL
from check_body_api import FIELDS

class DemoReferenceTests(unittest.TestCase):
    def test_original_fields_complete(self):
        self.assertEqual(set(ORIGINAL),set(FIELDS)-{'faces'})
    def test_png_exact_rgb(self):
        rgb=bytes(range(18));data=png_rgb(3,2,rgb)
        self.assertEqual(data[:8],b'\x89PNG\r\n\x1a\n');offset=8;payload=b''
        while offset<len(data):
            n=struct.unpack_from('>I',data,offset)[0];name=data[offset+4:offset+8];chunk=data[offset+8:offset+8+n]
            self.assertEqual(zlib.crc32(name+chunk),struct.unpack_from('>I',data,offset+8+n)[0])
            if name==b'IDAT':payload+=chunk
            offset+=n+12
        self.assertEqual(zlib.decompress(payload),b'\0'+rgb[:9]+b'\0'+rgb[9:])
        with self.assertRaises(ValueError):png_rgb(3,2,b'bad')
    def test_viewer_topology_is_mhr(self):
        path=Path(__file__).resolve().parents[1]/'demo/web/skeleton.json';v=json.loads(path.read_text())
        self.assertEqual(len(v['parents']),127);self.assertEqual(len(set(v['names'])),127)
        self.assertEqual(v['names'][:2],['body_world','root']);self.assertEqual(v['parents'][0],-1)
        for i,p in enumerate(v['parents'][1:],1):self.assertTrue(0<=p<i)

if __name__=='__main__':unittest.main()
