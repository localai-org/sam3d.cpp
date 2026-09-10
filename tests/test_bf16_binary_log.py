import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_bf16_binary_log import check


def complete_log():
    lines = []
    for op, width, broadcast, mode, fused in itertools.product(
            ('ADD', 'MUL'), (1, 7, 513, 1280), range(3), range(8), range(2)):
        expected = int(fused == 1 and mode == 0)
        lines.append(f'BF16_BINARY_CASE op={op} width={width} broadcast={broadcast} mode={mode} fused={fused} expected={expected}')
        if expected:
            lines.append(f'BF16_BINARY_ROUND {op}: 1 x 3 us = 3 us')
        lines.append('BF16_BINARY_CASE_END')
    return '\n'.join(lines)


class BinaryLogTests(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(check(complete_log()), dict(passed=True, graph_executions=384,
                                                    fused=24, refused_or_disabled=360))

    def test_disabled_or_missing_logger(self):
        with self.assertRaises(ValueError):
            check('')
        with self.assertRaises(ValueError):
            check('\n'.join(x for x in complete_log().splitlines() if not x.startswith('BF16_BINARY_ROUND')))

    def test_bad_guards_and_ops(self):
        for old, new in [('fused=0 expected=0', 'fused=0 expected=1'),
                         ('BF16_BINARY_ROUND ADD: 1', 'BF16_BINARY_ROUND MUL: 1'),
                         ('BF16_BINARY_ROUND ADD: 1', 'BF16_BINARY_ROUND ADD: 2')]:
            with self.assertRaises(ValueError):
                check(complete_log().replace(old, new, 1))

    def test_incomplete_and_duplicate(self):
        for text in [complete_log().rsplit('\n', 1)[0], complete_log() + '\n' + complete_log()]:
            with self.assertRaises(ValueError):
                check(text)


if __name__ == '__main__':
    unittest.main()
