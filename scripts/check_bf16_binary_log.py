#!/usr/bin/env python3
"""Verify that the model-free Vulkan binary-round tests exercised fusion guards."""
import argparse
import itertools
import json
from pathlib import Path
import re


def check(text):
    cases = {}
    current = None
    observed = 0
    for line in text.splitlines():
        if line.startswith('BF16_BINARY_CASE op='):
            if current is not None:
                raise ValueError('unterminated case')
            m = re.fullmatch(r'BF16_BINARY_CASE op=(ADD|MUL) width=(\d+) broadcast=(\d+) mode=(\d+) fused=([01]) expected=([01])', line)
            if not m:
                raise ValueError('invalid case marker')
            op, width, broadcast, mode, fused, expected = m.groups()
            current = (op, int(width), int(broadcast), int(mode), int(fused))
            if int(expected) != int(current[-1] == 1 and current[-2] == 0):
                raise ValueError('incorrect expected guard decision')
            observed = 0
        elif line == 'BF16_BINARY_CASE_END':
            if current is None or current in cases:
                raise ValueError('missing or duplicate case')
            expected = int(current[-1] == 1 and current[-2] == 0)
            if observed != expected:
                raise ValueError(f'fusion count {observed} != {expected} for {current}')
            cases[current] = observed
            current = None
        elif current is not None and 'BF16_BINARY_ROUND' in line:
            m = re.fullmatch(r'BF16_BINARY_ROUND (ADD|MUL): (\d+) x .*', line)
            if not m or m[1] != current[0]:
                raise ValueError('unexpected fusion log')
            observed += int(m[2])
    required = set(itertools.product(('ADD', 'MUL'), (1, 7, 513, 1280), range(3), range(8), range(2)))
    if current is not None or set(cases) != required:
        raise ValueError('incomplete binary test coverage')
    return dict(passed=True, graph_executions=len(cases), fused=sum(cases.values()),
                refused_or_disabled=len(cases)-sum(cases.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    result = check(a.log.read_text())
    with a.report.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
