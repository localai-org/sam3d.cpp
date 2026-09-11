#!/usr/bin/env python3
"""Measure native inference variants against the existing implementation, not ground truth.
Run under an external memory limit. Uses the public CLI worker protocol; never loads Python models.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess

VARIANTS = {
    'baseline': [],
    'slim': ['--slim-body-intermediates'],
    'schedule': ['--body-intermediates=0,1,2'],
    'no-correctives': ['--no-body-correctives'],
    'crop448': ['--body-crop-size=448'],
    'crop384': ['--body-crop-size=384'],
    'fast512': ['--body-intermediates=0,1,2', '--no-body-correctives', '--slim-body-intermediates'],
    'fast448': ['--body-intermediates=0,1,2', '--no-body-correctives', '--slim-body-intermediates', '--body-crop-size=448'],
    'fast384': ['--body-intermediates=0,1,2', '--no-body-correctives', '--slim-body-intermediates', '--body-crop-size=384'],
    'final-only': ['--body-intermediates=none'],
}

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(args):
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    base = [str(args.runner.resolve()), '--worker', str(args.module.resolve()), args.backend, '0', args.device_name,
            str(args.backbone.resolve()), str(args.branch.resolve()), str(args.mhr.resolve()), str(args.threads)]
    env = os.environ.copy()
    if args.library:
        env['LD_LIBRARY_PATH'] = str(args.library.resolve().parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    report = {'scope': 'Native output divergence, no ground-truth accuracy claim',
              'git_head': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
              'runner_sha256': sha(args.runner), 'backend_sha256': sha(args.module),
              'library_sha256': sha(args.library) if args.library else None,
              'precision': args.precision, 'backend': args.backend, 'threads': args.threads,
              'warmup_requests': args.warmup, 'timed_repeats_per_input': args.repeats,
              'inputs': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in args.input],
              'models': {k: sha(getattr(args,k)) for k in ('backbone','branch','mhr')},
              'environment': {k:v for k,v in env.items() if k.startswith(('GGML_', 'SAM3D_', 'VK_'))}, 'variants': {}}
    for name in args.variant:
        directory = args.output / name
        directory.mkdir()
        cmd = base + (['--bf16'] if args.precision == 'bf16' else []) + VARIANTS[name]
        times, hashes = [], {}
        with (directory / 'stderr.log').open('w') as log:
            with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env) as proc:
                try:
                    if proc.stdout.readline() != b'READY\n':
                        raise RuntimeError(f'{name}: worker did not start; see {log.name}')
                    for step in range(args.warmup + args.repeats * len(args.input)):
                        idx = step % len(args.input)
                        target = directory / f'input-{idx}.bin'
                        for path in (args.input[idx].resolve(), target.resolve()):
                            raw = os.fsencode(path)
                            proc.stdin.write(struct.pack('<I', len(raw)) + raw)
                        proc.stdin.flush()
                        duration = None
                        while True:
                            line = proc.stdout.readline()
                            if not line:
                                raise RuntimeError(f'{name}: worker failed; see {log.name}')
                            if line.startswith(b'TIMING body_infer_ms '):
                                duration = float(line.split()[-1])
                            if line == b'DONE\n':
                                break
                        if duration is None:
                            raise RuntimeError('missing inference timing')
                        digest = sha(target)
                        hashes.setdefault(idx,set()).add(digest)
                        if step >= args.warmup:
                            times.append(duration)
                    proc.stdin.close()
                    if proc.wait(timeout=30):
                        raise RuntimeError('worker exited with error')
                finally:
                    if proc.poll() is None:
                        proc.kill(); proc.wait()
        v = {'command':cmd, 'median_ms':statistics.median(times), 'p95_ms':sorted(times)[math.ceil(.95*len(times))-1],
             'timings_ms':times, 'repeat_exact':all(len(v)==1 for v in hashes.values()),
             'output_sha256':{str(i):sorted(v) for i,v in hashes.items()}}
        report['variants'][name] = v
        (args.output / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'{name}: {v["median_ms"]:.2f} ms, repeat exact={v["repeat_exact"]}', flush=True)
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('runner','module','backbone','branch','mhr','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--library',type=Path)
    p.add_argument('--input',type=Path,action='append',required=True)
    p.add_argument('--variant',choices=VARIANTS,action='append',required=True)
    p.add_argument('--precision',choices=['f32','bf16'],default='bf16')
    p.add_argument('--backend',choices=['CPU','Vulkan'],default='Vulkan')
    p.add_argument('--device-name',default='-')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--repeats',type=int,default=10,help='timed repeats per input')
    args=p.parse_args()
    if args.warmup<1 or args.repeats<1 or not 1<=args.threads<=256:p.error('invalid repeat/thread counts')
    if len(set(args.variant)) != len(args.variant):p.error('duplicate variants')
    run(args)
