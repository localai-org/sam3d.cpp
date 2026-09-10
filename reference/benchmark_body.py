"""Resident original Body benchmark, used only in the trusted reference image.

Same reviewed original RGB preprocessing, full six-layer pose branch and MHR as
the parity harness. No diagnostic tensor captures/hooks in timed calls. Body
only: excludes detector, segmentation, hand refinement, exports and rendering.
"""
import contextlib
import json
import statistics
import time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import save_file
from capture_mhr import digest

def benchmark(a,pipeline,trained_metadata):
    if not 1<=a.benchmark_warmup<=100 or not 2<=a.benchmark_repeats<=100:
        raise ValueError('invalid bounded benchmark iteration count')
    report_path=a.output/'benchmark.json'
    if report_path.exists():raise FileExistsError(report_path)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    report=dict(scope=__doc__,precision=a.benchmark_precision,compile=a.benchmark_compile,
        attention=a.benchmark_attention,device=a.device,threads=a.threads,
        torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,
        tf32=False,gradients=False,resident_weights=True,
        source=pipeline.metadata,trained_state=trained_metadata,
        script_sha256=digest(Path(__file__)),status='running')
    with (a.output/'image.input').open('xb') as stream:
        stream.write(b'S3DIMG01');pipeline.write_input(stream)
    report['image_input_sha256']=digest(a.output/'image.input')
    with torch.inference_mode():
        save_file({'case.0000.prepare.normalized_rgb':pipeline.normalized_image()},a.output/'normalized-input.safetensors')
    report['normalized_input_sha256']=digest(a.output/'normalized-input.safetensors')
    def write():report_path.write_text(json.dumps(report,indent=2)+'\n')
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
    def materialize(result):
        tensors={'output_tokens':result[0].detach().float().cpu().contiguous()}
        for name,value in result[1][-1].items():
            if isinstance(value,torch.Tensor):tensors[name]=value.detach().cpu().contiguous()
        if not all(torch.isfinite(v).all() for v in tensors.values()):raise ValueError('nonfinite benchmark result')
        return tensors
    def invoke():
        result=pipeline.run()
        # Match the public native result boundary: final tensors on the host,
        # including vertices/joints/projections; no intermediate layer copies.
        return materialize(result)
    backends=[SDPBackend.MATH] if a.benchmark_attention=='math' else [SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.MATH]
    write()
    try:
        with torch.inference_mode(),sdpa_kernel(backends):
            start=time.perf_counter();baseline=invoke();sync();report['first_eager_seconds']=time.perf_counter()-start
            save_file(baseline,a.output/'eager-output.safetensors')
            report['eager_output_sha256']=digest(a.output/'eager-output.safetensors')
            if a.benchmark_compile=='backbone':
                # Compile the dominant original backbone, retaining the actual
                # TorchScript MHR and Python feedback. No silent eager fallback.
                from torch._dynamo.utils import counters
                counters.clear()
                # Body calls get_intermediate_layers, not encoder.forward.
                # Compile the wrapper entry point actually used by inference.
                pipeline.holder.backbone.forward=torch.compile(pipeline.holder.backbone.forward,mode='reduce-overhead',fullgraph=True)
                start=time.perf_counter();compiled=invoke();sync()
                report['compile_and_first_call_seconds']=time.perf_counter()-start
                report['compiled_graphs']=int(counters['stats']['unique_graphs'])
                if not report['compiled_graphs']:raise RuntimeError('no compiled graph executed')
                report['compiled_vs_eager']={}
                for name,value in compiled.items():
                    ref=baseline[name].double();got=value.double();diff=got-ref
                    report['compiled_vs_eager'][name]=dict(max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/ref.norm().clamp_min(1e-12)))
                save_file(compiled,a.output/'compiled-output.safetensors')
            for _ in range(a.benchmark_warmup):invoke()
            sync()
            if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
            values=[]
            for i in range(a.benchmark_repeats):
                sync();start=time.perf_counter();result=invoke();sync();values.append(time.perf_counter()-start)
                print(json.dumps(dict(benchmark_iteration=i,seconds=values[-1])),flush=True)
            save_file(result,a.output/'final-output.safetensors')
            report.update(status='complete',warm_seconds=values,median_seconds=statistics.median(values),
                minimum_seconds=min(values),maximum_seconds=max(values),mean_seconds=statistics.mean(values),
                final_output_sha256=digest(a.output/'final-output.safetensors'),
                peak_cuda_allocated=torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,
                peak_cuda_reserved=torch.cuda.max_memory_reserved() if a.device=='cuda' else 0)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:write()
    print(json.dumps({k:report[k] for k in ['status','precision','compile','median_seconds','minimum_seconds','maximum_seconds','peak_cuda_allocated']},indent=2),flush=True)
