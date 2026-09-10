#!/usr/bin/env python3
"""Profile an unchanged native runner. Launch INSIDE run_bounded.py.

Captures timestamped stages and whole-device NVIDIA telemetry every 100 ms.
Optional perf and LD_PRELOAD shim collect CPU stacks and GGML host-call timings.
Those synchronous call durations include CPU dispatch/waits, not GPU-only time.
The output directory must not exist. No model download or server mutation.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import struct
import threading
import time

def warm_summary(durations,warmup):
    if warmup<1 or warmup>=len(durations) or any(not math.isfinite(x) or x<0 for x in durations):
        raise ValueError('invalid complete inference timings')
    warm=durations[warmup:]
    return dict(warmup=warmup,count=len(warm),median_seconds=statistics.median(warm),
        minimum_seconds=min(warm),maximum_seconds=max(warm),
        p95_seconds=sorted(warm)[math.ceil(.95*len(warm))-1],p95_method='nearest rank')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--perf', type=Path)
    p.add_argument('--perf-frequency',type=int,default=99,help='CPU samples/second (1–1000); diagnostic timing includes sampling overhead')
    p.add_argument('--perf-callgraph',choices=['fp','dwarf'],default='fp',help='use dwarf for builds without frame pointers')
    p.add_argument('--shim', type=Path)
    p.add_argument('--library', type=Path, help='shared inference library to fingerprint as well as runner')
    p.add_argument('--backend-library',type=Path,help='fingerprint the explicitly selected backend, including build-copy patches')
    p.add_argument('--warmup',type=int,default=1,help='exclude this many requests from warm statistics; use 5 for acceptance runs')
    p.add_argument('--worker-input', type=Path, action='append', default=[],
                   help='send a framed request to --worker; repeat for sequential warm inference')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd:
        p.error('missing command')
    if not 1<=a.perf_frequency<=1000:p.error('perf frequency must be 1–1000')
    if a.warmup<1 or (a.worker_input and a.warmup>=len(a.worker_input)):p.error('warmup must leave at least one timed request')
    if a.worker_input and (len(cmd)<2 or cmd[1]!='--worker' or len(a.worker_input)>64):
        p.error('--worker-input requires the native --worker command and at most 64 images')
    a.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if a.shim:
        env['LD_PRELOAD'] = str(a.shim.resolve())
        env['SAM3D_GGML_TRACE'] = str((a.output / 'ggml.csv').resolve())
    # Apply preloading only to inference, not perf itself or the GPU sampler.
    launch = ['env', *[f'{k}={v}' for k, v in env.items() if os.environ.get(k) != v], *cmd]
    if a.perf:
        launch = [str(a.perf), 'record', '--clockid', 'mono', '-e', 'cpu-clock:u', '-F', str(a.perf_frequency),
                  '--call-graph', 'dwarf,8192' if a.perf_callgraph=='dwarf' else 'fp',
                  '-o', str(a.output / 'perf.data'), '--', *launch]
    started = time.monotonic()
    started_unix = time.time()
    stages = []
    with (a.output / 'gpu.csv').open('x') as gpu, (a.output / 'stderr.log').open('x') as log:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw',
                                    '--format=csv,nounits', '-lms', '100'], stdout=gpu, stderr=subprocess.STDOUT)
        try:
            with subprocess.Popen(launch, stdout=log, stderr=subprocess.PIPE, text=True,
                                  stdin=subprocess.PIPE if a.worker_input else subprocess.DEVNULL) as process:
                writer=None;write_errors=[]
                if a.worker_input:
                    # Bound the request list; submit concurrently with stderr
                    # draining so 5 warmups + 20 measurements cannot deadlock
                    # when framed paths exceed a pipe's capacity.
                    frames = bytearray()
                    for i, source in enumerate(a.worker_input):
                        for path in (source.resolve(), (a.output/f'result-{i:03d}.bin').resolve()):
                            raw = os.fsencode(path)
                            if not 1 <= len(raw) <= 4096 or b'\0' in raw:
                                raise ValueError('invalid worker path')
                            frames += struct.pack('<I', len(raw)) + raw
                    def submit():
                        try:process.stdin.buffer.write(frames);process.stdin.close()
                        except (BrokenPipeError,OSError) as exc:write_errors.append(repr(exc))
                    writer=threading.Thread(target=submit,daemon=True);writer.start()
                for line in process.stderr:
                    elapsed = time.monotonic() - started
                    log.write(f'{elapsed:.6f} {line}'); log.flush()
                    if line.startswith(('STAGE', 'C API', 'ggml_vulkan:')):
                        stages.append(dict(elapsed=elapsed, message=line.strip()))
                        print(f'{elapsed:.3f}s {line}', end='', flush=True)
                code = process.wait()
                if writer:writer.join(timeout=5)
                if writer and writer.is_alive():raise RuntimeError('request writer did not stop with inference')
                if write_errors and code==0:raise RuntimeError('request submission incomplete: '+str(write_errors))
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill(); monitor.wait()
    elapsed = time.monotonic()-started
    with open(cmd[0], 'rb') as binary:
        runner_hash = hashlib.file_digest(binary, 'sha256').hexdigest()
    report = dict(command=cmd, elapsed=elapsed, returncode=code,
                  started_monotonic=started, started_unix=started_unix,
                  stages=stages, perf=bool(a.perf), shim=bool(a.shim),
                  perf_frequency=a.perf_frequency if a.perf else None,
                  perf_callgraph=a.perf_callgraph if a.perf else None,
                  runner_sha256=runner_hash,
                  environment={k: env.get(k) for k in (
                      'VK_DRIVER_FILES', 'GGML_VK_DISABLE_F16', 'GGML_VK_DISABLE_COOPMAT',
                      'GGML_VK_DISABLE_COOPMAT2', 'GGML_VK_PERF_LOGGER', 'GGML_VK_FUSE_BF16_ROUND', 'GGML_VK_FUSE_BF16_BINARY', 'GGML_VK_BF16_BINARY_LINEAR',
                      'SAM3D_BF16_FLASH_ATTENTION','SAM3D_BF16_COOPMAT2','SAM3D_BF16_PRECISE_PREFIX','SAM3D_BF16_PREFIX_PAD','SAM3D_BF16_PREFIX_MV','SAM3D_BF16_PREFIX_TRANSPOSE',
                      'GGML_VK_BF16_MATMUL_TILE','GGML_VK_BF16_MATMUL_TRACE','SAM3D_BATCHED_TRANSFERS','SAM3D_SIMD_SKINNING',
                      'GGML_VK_F32_MATVEC_ROWS','GGML_VK_F32_MATVEC_TRACE',
                      'GGML_VK_F32_NARROW_MATMUL','GGML_VK_F32_NARROW_TRACE',
                      'GGML_VK_F32_NARROW_TILE','SAM3D_BF16_PACKED_FFN',
                      'GGML_VK_FUSE_BF16_SILU_GATE','GGML_VK_FUSE_BF16_AFFINE','GGML_VK_FUSE_BF16_NORM_AFFINE','SAM3D_SIMD_IMAGE','SAM3D_IMAGE_GATHER')})
    if a.library:
        with a.library.open('rb') as f:
            report['library_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    if a.backend_library:
        with a.backend_library.open('rb') as f:report['backend_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    report['inference_seconds'] = []
    intervals = []
    infer_start = None
    for stage in stages:
        if stage['message'].startswith('STAGE Estimating body:'):
            infer_start = stage['elapsed']
        elif stage['message']=='STAGE Writing body result' and infer_start is not None:
            report['inference_seconds'].append(stage['elapsed']-infer_start)
            intervals.append((infer_start,stage['elapsed']))
            infer_start = None
    if a.worker_input:
        report['worker_results'] = []
        for i, source in enumerate(a.worker_input):
            target = a.output/f'result-{i:03d}.bin'
            if target.exists():
                with target.open('rb') as f:
                    sha = hashlib.file_digest(f,'sha256').hexdigest()
                report['worker_results'].append(dict(input=str(source),output=str(target),sha256=sha))
    with (a.output / 'gpu.csv').open() as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    # Whole-device observations include any unrelated GPU users, not just us.
    report['gpu_samples'] = len(rows)
    report['gpu'] = {}
    for key in ('utilization.gpu [%]', 'memory.used [MiB]', 'power.draw [W]'):
        try:
            values = [float(row[key]) for row in rows]
            if values:
                report['gpu'][key] = dict(mean=statistics.mean(values), maximum=max(values), minimum=min(values))
        except (KeyError, ValueError):
            report['gpu'][key] = dict(error='telemetry unavailable; see gpu.csv')
    if len(intervals)>a.warmup:
        report['warm_inference']=warm_summary(report['inference_seconds'],a.warmup)
        try:
            values=[float(row['utilization.gpu [%]']) for row in rows
                    if intervals[a.warmup][0] <= datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').timestamp()-started_unix <= intervals[-1][1]]
            report['warm_gpu'] = dict(samples=len(values),mean=statistics.mean(values),maximum=max(values)) if values else None
        except (ValueError,KeyError):
            report['warm_gpu'] = dict(error='cannot align telemetry timestamps')
    if a.shim:
        with (a.output/'ggml.csv').open() as f:
            calls=list(csv.DictReader(f))
        def summarize(selected):
            totals={}
            for row in selected:
                item=totals.setdefault(row['operation'],dict(calls=0,bytes=0,seconds=0.0))
                item['calls']+=1;item['bytes']+=int(row['bytes']);item['seconds']+=float(row['end'])-float(row['start'])
            return totals
        report['ggml_inclusive_totals']=summarize(calls)
        report['ggml_inference']=[summarize(row for row in calls if lo<=float(row['start'])-started<=hi) for lo,hi in intervals]
    if a.worker_input:
        report['complete_requests']=(len(report['worker_results'])==len(a.worker_input) and len(intervals)==len(a.worker_input))
        if not report['complete_requests']:code=code or 1
    (a.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
