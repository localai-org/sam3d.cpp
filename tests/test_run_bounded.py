import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_bounded import available_bytes, require_headroom, verify_limits, high_watermark, MIB


class BoundedRunnerTests(unittest.TestCase):
    def test_available_not_free_memory(self):
        self.assertEqual(available_bytes('MemFree: 1 kB\nMemAvailable: 123 kB\n'), 123 * 1024)
        for text in ['', 'MemFree: 1 kB', 'MemAvailable: -1 kB', 'MemAvailable: 2 MB']:
            with self.assertRaises(ValueError): available_bytes(text)

    def test_reserve_is_additional_to_job_limit(self):
        require_headroom(14, 6, 8)
        for args in [(13, 6, 8), (20, 0, 8), (20, 6, 0)]:
            with self.assertRaises(ValueError): require_headroom(*args)

    def test_missing_or_incorrect_cgroup_limit_fails_closed(self):
        limit = 64 * MIB
        maximum, high = str(limit), str(high_watermark(limit))
        verify_limits(maximum + '\n', high, '0', '1', limit)
        for values in [('max', high, '0', '1'), (maximum, 'max', '0', '1'),
                       (maximum, high, 'max', '1'), (maximum, high, '0', '0')]:
            with self.assertRaises(ValueError): verify_limits(*values, limit)
