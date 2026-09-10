#!/usr/bin/env python3
"""Serialize native verification and enforce a Linux cgroup RAM budget.

Fail closed when headroom, the lock, systemd user scopes or cgroup v2 limits
are unavailable. No rlimit on virtual memory: ASan needs a large shadow mapping.
Docker containers are daemon-owned, not descendants: cap them separately.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

MIB = 1024 * 1024


def available_bytes(text):
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'MemAvailable:':
            if len(fields) != 3 or fields[2] != 'kB' or not fields[1].isdigit():
                raise ValueError('invalid MemAvailable')
            return int(fields[1]) * 1024
    raise ValueError('MemAvailable unavailable')


def require_headroom(available, limit, reserve):
    if limit <= 0 or reserve <= 0 or available < limit + reserve:
        raise ValueError('insufficient RAM: need job budget plus reserved host headroom')


def verify_limits(maximum, high, swap, group, expected):
    if maximum.strip() != str(expected) or high.strip() != str(high_watermark(expected)) or swap.strip() != '0' or group.strip() != '1':
        raise ValueError('cgroup memory limits were not applied; refusing to run')


def high_watermark(limit):
    page = os.sysconf('SC_PAGE_SIZE')
    return limit * 5 // 6 // page * page


def inside(limit, metadata, command, lock_path, reserve):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r+') as lock:
        # Keep the lock in the capped job, not only its invoking agent. It must
        # survive loss of that parent while the inference process still runs.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_headroom(available_bytes(Path('/proc/meminfo').read_text()), limit, reserve)
        return execute_inside(limit, metadata, command)


def execute_inside(limit, metadata, command):
    entry = next((line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')), None)
    if entry is None or not entry.startswith('/') or '..' in Path(entry).parts:
        raise ValueError('cgroup v2 membership unavailable')
    cgroup = Path('/sys/fs/cgroup') / entry.lstrip('/')
    values = {name: (cgroup / name).read_text().strip() for name in ['memory.max', 'memory.high', 'memory.swap.max', 'memory.oom.group']}
    verify_limits(*values.values(), limit)
    record = dict(cgroup=str(cgroup), verified_limits=values, state='running')
    metadata.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(phase='bounded execution', **record)), flush=True)
    started = time.monotonic()
    result = subprocess.run(command, check=False)
    record.update(state='finished', returncode=result.returncode, elapsed_seconds=time.monotonic() - started,
                  memory_peak_bytes=int((cgroup / 'memory.peak').read_text()),
                  memory_events=(cgroup / 'memory.events').read_text())
    metadata.write_text(json.dumps(record, indent=2) + '\n')
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory-mib', type=int, default=6144)
    p.add_argument('--reserve-mib', type=int, default=10240)
    p.add_argument('--lock', type=Path, default=Path(__file__).resolve().parents[1] / 'generated/full-model.lock')
    p.add_argument('--report', type=Path)
    p.add_argument('--inside', type=int, help=argparse.SUPPRESS)
    p.add_argument('--inside-metadata', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--inside-lock', type=Path, help=argparse.SUPPRESS)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command: p.error('a command is required after --')
    if a.inside is not None:
        if not a.inside_metadata or not a.inside_lock: p.error('missing internal metadata/lock path')
        return inside(a.inside, a.inside_metadata, command, a.inside_lock, a.reserve_mib * MIB)
    if not a.report: p.error('--report is required (new file only)')
    if not 64 <= a.memory_mib <= 1048576 or not 64 <= a.reserve_mib <= 1048576:
        p.error('memory and reserve must be between 64 and 1048576 MiB')
    systemd = shutil.which('systemd-run')
    if not systemd: raise ValueError('systemd-run unavailable; no unbounded fallback')
    a.lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(a.lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        limit, reserve = a.memory_mib * MIB, a.reserve_mib * MIB
        available = available_bytes(Path('/proc/meminfo').read_text())
        require_headroom(available, limit, reserve)
        a.report.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the report before launching; never replace a previous run.
        with a.report.open('x') as report, tempfile.TemporaryDirectory(prefix='sam3d-budget-', dir=a.report.parent) as tmp:
            metadata = Path(tmp).resolve() / 'execution.json'
            unit = 'sam3d-verification-' + uuid.uuid4().hex
            record = dict(schema_version=1, unit=unit, command=command, available_bytes_before=available,
                          memory_max_bytes=limit, reserved_bytes=reserve, state='starting')
            report.write(json.dumps(record, indent=2) + '\n'); report.flush()
            argv = [systemd, '--user', '--scope', '--quiet', '--collect', '--unit=' + unit,
                    '-p', 'MemoryMax=' + str(limit), '-p', 'MemoryHigh=' + str(high_watermark(limit)),
                    '-p', 'MemorySwapMax=0', '-p', 'OOMPolicy=kill',
                    str(Path(sys.executable).resolve()), str(Path(__file__).resolve()),
                    '--inside', str(limit), '--inside-metadata', str(metadata),
                    '--inside-lock', str(a.lock.resolve()), '--reserve-mib', str(a.reserve_mib), '--', *command]
            # Transfer ownership to the inside monitor. Concurrent contenders
            # may create scopes, but only one can acquire the inside lock and
            # reach execution; the rest fail closed before launching a model.
            fcntl.flock(lock, fcntl.LOCK_UN)
            result = subprocess.run(argv, check=False)
            record.update(state='finished', scope_returncode=result.returncode)
            if metadata.exists(): record['execution'] = json.loads(metadata.read_text())
            report.seek(0); report.truncate(); report.write(json.dumps(record, indent=2) + '\n')
            return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print('bounded verification: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
