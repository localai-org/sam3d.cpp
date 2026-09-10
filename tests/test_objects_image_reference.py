import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ObjectsImageProvenanceTests(unittest.TestCase):
    def test_original_fixture_identity(self):
        meta = json.loads((ROOT/'tests/fixtures/objects-image.json').read_text())
        expected = 'f59846fe4df9ebdf5685cccd15be7bf2628c3855907856da6bd6d71067f3c663'
        self.assertEqual(meta['artifacts']['regression.txt'], expected)
        self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures/objects-image.txt').read_bytes()).hexdigest(), expected)
        self.assertEqual(meta['revision'], 'f91db411c50efee93d8db7aeb323885650f6f722')
        self.assertEqual(meta['repeats'], 3)
        self.assertTrue(meta['observer_exact'])
        self.assertEqual(len(meta['cases']), 6)
        self.assertEqual([x['case'] for x in meta['rejections']], ['empty','single_pixel','one_row','one_column'])

    def test_capture_source_identity(self):
        meta = json.loads((ROOT/'tests/fixtures/objects-image.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'reference/capture_objects_image.py').read_bytes()).hexdigest(), meta['script_sha256'])
        self.assertEqual(len(meta['source_hashes']), 5)
        # The normal fixture is the five small complete cases; the official
        # photograph is a separate acceptance run, never silently elided.
        self.assertEqual([c['official_photo'] for c in meta['cases']], [False]*5+[True])
        self.assertEqual(meta['cases'][-1]['side'], 518)

if __name__ == '__main__':
    unittest.main()
