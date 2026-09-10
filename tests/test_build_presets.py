import json
from pathlib import Path
import unittest


class BuildPresetTests(unittest.TestCase):
    def test_optimized_verification_keeps_sanitizers_and_assertions(self):
        path = Path(__file__).resolve().parents[1] / 'CMakePresets.json'
        presets = json.loads(path.read_text())
        configs = {p['name']: p for p in presets['configurePresets']}
        preset = configs['optimized-sanitizers']
        self.assertEqual(preset['inherits'], 'debug')
        self.assertNotEqual(preset['binaryDir'], configs['debug']['binaryDir'])
        options = preset['cacheVariables']
        self.assertEqual(options['CMAKE_BUILD_TYPE'], 'RelWithDebInfo')
        for key in ['SAM3D_SANITIZERS', 'SAM3D_ADDRESS_SANITIZER']:
            self.assertEqual(options[key], 'ON')
        self.assertEqual(options['SAM3D_VULKAN'], 'OFF')
        for lang in ['C', 'CXX']:
            flags = options[f'CMAKE_{lang}_FLAGS_RELWITHDEBINFO'].split()
            self.assertIn('-O2', flags)
            self.assertIn('-g', flags)
            self.assertNotIn('-DNDEBUG', flags)
            self.assertNotIn('-ffast-math', flags)
        for group in ['buildPresets', 'testPresets']:
            entry = next(p for p in presets[group] if p['name'] == 'optimized-sanitizers')
            self.assertEqual(entry['configurePreset'], 'optimized-sanitizers')
