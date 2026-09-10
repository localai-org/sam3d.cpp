# Single-image Body inference profile

The subsequent [F32/BF16 work](BODY_PRECISION.md) reduces accepted native BF16
warm inference to **116.7 ms**, compared with upstream CUDA **84.5 / 81.6 ms**.
The demo now supports BF16 and passes real browser upload/render/export QA.
The historical strict-F32 measurements and acceptance evidence below are unchanged;
they are not the current BF16 demo's latency.

## Decoder/head residency and host-work follow-up

The third pass reduces native warm image latency from about **620 ms to 395 ms**
(median of 11 warm calls). The measured range is **393–404 ms**: approximately
the 400 ms target, not a hard sub-400 ms bound. Precision, tolerances, assertions
and UBSan are unchanged. No Nix derivations or GGML submodule edits were needed.

| Optimization step | Native warm image time |
| --- | ---: |
| Previous accepted persistent worker | 617–624 ms |
| Shared immutable decoder/head weights and device parameter cache | 455–459 ms |
| Cache expanded sparse MHR projection and smaller static parameters | 439–441 ms |
| Omit unused patch/camera/prompt/feedback/context readbacks | 415–422 ms |
| Tiled layout copies and once-per-joint skin rotation normalization | **393–404 ms** |

The final profile is serialized, with no concurrent model/test/build job. It
contains 12 alternating requests for the accepted upstream dancer and a second
image; all complete result hashes match their respective accepted outputs.
Cold model loading takes about 1.55 s; first inference another 0.91 s. Warm
performance excludes loading, image upload, export and browser rendering.

| Per warm image | Previous pass | This pass |
| --- | ---: | ---: |
| Upload volume / calls | 611.25 MB / 579 | **100.11 MB / 218** |
| Download volume / calls | 110.34 MB / 280 | **42.94 MB / 235** |
| Synchronous graph calls | 85 | 85 |
| Graph-compute host time | about 315 ms | about 313 ms |
| Upload host time | about 27.5 ms | about 4.5 ms |
| Download host time | about 15 ms | about 8.5 ms |
| Backend allocations during warm inference | 0 | 0 |

MB are decimal. Host-call timings include dispatch/waits, not only GPU execution.
Warm whole-device utilization averages **69%**, with a 99% peak over 43 samples;
the entire process including initialization/exit averages 45.6%. Peak sampled
device memory is 4,686 MiB including the desktop. Driver utilization windows and
100 ms sampling make short-run readings noisy; the earlier four-request stages
are timing checks, not sustained-utilization acceptance. **90% sustained GPU
utilization is not achieved.** The native profiling scope peaks at 1,173,151,744
bytes RAM with zero high/max/OOM events, under the unchanged 6 GiB/no-swap cap.

Implementation:

- Nested decoder/head parameter maps now share owned immutable snapshots. Raw
  diagnostic vectors are copied and checked on entry; archive loads are checked
  once. Input, output, shape and semantic validation remain. Changed caller
  arrays cannot mutate a snapshot or reuse stale validation.
- Device parameters use a canonical flat allocation with checked reshape views,
  so subsets/layout views share storage. The general immutable device-weight
  budget is now **1 GiB**, separate from the resident backbone's 4 GiB budget
  and its existing graph/scratch limits. Archive pinning remains capped at
  768 MiB. These independent limits are not a global VRAM cap.
- MHR's fixed COO projection is validated and expanded once per archive, then
  uploaded once per session. Index bounds/order/uniqueness and finite values are
  still checked. Both host snapshots and GPU storage are released with their owners.
- Diagnostic component calls retain their full traces by default. Composition
  avoids traces it never consumes, and a one-way decoder retains its unchanged
  host image context instead of downloading it after each layer. All six layers
  still compute learned pose/camera updates and full mesh feedback.
- Layout transposes use cache-sized tiles without changing values. Skinning
  computes the original two quaternion normalization steps once per joint,
  instead of repeating them for every influenced vertex; even intermediate
  diagnostic norms/rotations are unchanged. No approximate skinning was added.

Verification: all **646 unchanged upstream checks pass**, and the full **529
captured tensors remain byte-identical**, including final geometry. Both public
result hashes remain identical across all 12 requests. All 44 native tests pass
under CPU ASan/UBSan/LSan and in the Vulkan-enabled UBSan build. New regressions
cover immutable map/subset ownership, parameter reshape storage/layout, compact
versus diagnostic outputs, sparse expansion/rejection/cache reuse, tiled edge
sizes and unchanged skinning traces. NVIDIA storage/repeated-backbone tests,
96 Python tests and Go race tests pass. The six existing strict CPU full-model
comparison gaps remain open; component sanitizer success does not waive them.

The existing demo service was restarted in place. Real headless Chrome QA
passes photo upload, invalid-input handling, cancellation/recovery, a second
warm generation, orbit controls, original/native overlays, GLB reload, history
and desktop/mobile scrolling. Screenshots were visually inspected. Actual
browser jobs measure **2.669 s cold / 0.462 s warm**, including server exports;
the prior warm browser job was 0.700 s. All 19 wire fields and every exported
GLB vertex/index are unchanged. Final mesh/projection comparisons against the
original upstream example still pass, without alignment or relaxed tolerances.

Remaining cost is now about 65–75 ms outside traced graph/transfer calls.
Warm-only CPU stacks still show layout copies, finite scans, skinning and
keypoint feedback; driver fence waits are the largest sampled symbol. Those
are sampled user CPU percentages, not wall-time shares or proof of lock
contention. The next step is reusable decoder/geometry graphs and device-resident
activation/geometry feedback to reduce the remaining 85 synchronous submissions
and 100 MB of uploads. No decoder graph cache or fully GPU-resident geometry
feedback is claimed by this pass. Optimized-upstream performance parity remains open.

Ignored evidence:

- `generated/diagnostics/decoder-{weights,geometry-cache,final,hoisted}/`: per-step
  timings, CPU stacks, GPU and GGML-call traces, with corresponding budget reports.
  The final accepted serialized profile is `decoder-hoisted`. `perf` now uses
  `--clockid mono`, so warm intervals align with the profiler's monotonic stages.
- `generated/fixtures/body-decoder-hoisted-vulkan/`: final full capture and
  `parity-trained-v1.json`. Capture SHA256 remains
  `a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`.
- Final measured shared library SHA256:
  `c5154e19c8bb908206b44ee830f181f1db667048e07afc29564d6c9005bcd702`.
- `generated/demo-qa-decoder-resident/`: Chrome report, screenshots and final
  geometry/export comparison; warm job `d1598739f3311d885dd56fea`.

## Resident-backbone and persistent-worker follow-up (historical)

The second pass is implemented and deployed. It improves useful latency but
**does not yet achieve the requested approximately 90% sustained GPU utilization**.
Strict F32, assertions, UBSan and the original numerical tolerances remain intact.
No Nix derivations or GGML submodule modifications were needed.

| Optimization step | Native warm image time |
| --- | ---: |
| Resident 32-block transformer graph + persistent model session | 0.877–0.885 s |
| Hoisted per-element map lookups + omit discarded skinning operation traces | 0.664–0.678 s |
| Final-result inference omits unused boundary readbacks/copies/projections | 0.617–0.624 s |
| Actual browser warm generation, including validation and GLB/OBJ exports | **0.700 s** |

The former demo needed 3.32 s for each fresh-process job. The new browser's
first job takes **2.912 s**, then **0.700 s** with the same resident worker.
Thus the warm browser improvement is about **4.7×** versus the previous pass;
do not compare warm inference alone with cold model loading as equivalent work.
Photo preparation/upload and browser rendering are outside the job timer.

The final instrumented run contains 12 alternating requests using the accepted
upstream dancer image and a second, different image. The 11 warm calls take
0.617–0.624 s; every result exactly matches its corresponding pre-optimization
result. The measured warm interval has 68 whole-device samples: **45.7% mean,
99% maximum GPU utilization**. The whole process, including loading and exit,
averages 37.3%. Peak sampled device memory is 4,472 MiB (including the desktop).
100 ms polling does not change the driver's own utilization averaging window;
these are not GPU occupancy or per-kernel measurements. A 99% peak is not a
claim of sustained 90% utilization.

Per warm image, the GGML host-call trace records:

| Measurement | Before residency, fresh request | Final warm request |
| --- | ---: | ---: |
| Upload volume | 4,832.85 MB | 611.25 MB |
| Download volume | 278.09 MB | 110.34 MB |
| Synchronous graph calls | 122 | 85 |
| Graph-compute host time | 0.317 s | about 0.315 s |
| Upload host time | 0.269 s | about 0.0275 s |
| Download host time | 0.0364 s | about 0.015 s |
| Backend allocations during warm inference | — | **0** |

MB are decimal. Compute durations include dispatch, driver waits and device
execution, not GPU-only timestamps. The near-unchanged compute time is expected:
this pass primarily removes host work and transfers without changing neural math.
The 12-request scope peaks at **1,454,755,840 bytes RAM**, with zero high/max/OOM
events under the 6 GiB cap. It does not grow a fresh weight cache per image.

Implementation details:

- Vulkan model loading streams checked transformer weights into permanent
  device buffers, discarding large host copies. Small stem/norm parameters are
  immutable snapshots too. The model retains its existing CPU path; there is
  no automatic device fallback or claim that CPU full-model parity is fixed.
- A single reusable graph joins all 32 blocks. The shared block builder retains
  the accepted split-dot, RoPE and SDPA equations. A lifetime-aware graph
  allocator reuses temporary activation memory under a 1 GiB budget; transformer
  weights have a separate 4 GiB budget. MHR constants and other scratch pools
  retain their earlier independent limits. These are not a configurable global
  VRAM cap. Model/session destruction releases all buffers.
- Normal inference reads only the final transformer output; the diagnostic
  capture reads every block boundary from the same graph. Model-internal
  per-layer mesh feedback still runs. Only unused vertex-pixel projections and
  trace copies are skipped. The public result still contains all 19 fields.
- The default demo worker now owns one model session across requests. Its
  private stdin protocol has bounded length-prefixed paths, never shell commands.
  Cancellation, failure and shutdown discard/reap the worker; the next request
  creates a new one. An idle timer releases RAM/VRAM after 2–4 minutes by default.
  `--persistent-worker=false` retains the one-shot fallback. No new C API is
  necessary: callers reuse the existing opaque model handle.

Validation caught two implementation errors before acceptance: measuring a
GGML graph does not allocate its storage (an explicit reserve is required), and
an immutable RoPE-angle input must be protected from allocator lifetime reuse
between evaluations. The latter passed the first image but failed a changed
second image; repeated-input tests now cover it on CPU and NVIDIA Vulkan.

Final acceptance: **646 unchanged upstream comparisons pass** and all **529
capture tensors are byte-identical** to the earlier accepted capture. Both
images retain their original complete public-output hashes across all 12 calls.
All **44 native tests** pass under CPU ASan/UBSan/LSan and in the Vulkan-enabled
UBSan build; the repeated resident-backbone test also runs on the NVIDIA module.
The 96 Python tests and Go tests with the race detector pass. New tests cover
worker framing, reuse, stale outputs, timeout/reaping, failure/restart, idle
release, immutable parameters and compact versus diagnostic skinning.

Real headless Chrome QA passes upload, actual cancellation/recovery, warm repeat,
native/original overlays, all exported GLB vertices/indices, history reload,
desktop/mobile scrolling and invalid input handling. Screenshots were visually
inspected. The warm browser result and GLB are byte-identical to acceptance.

Ignored evidence:

- `generated/diagnostics/resident-{worker,host-loops,final-only,validated}/`
  contains per-step CPU stacks, GPU CSV, GGML-call CSV and JSON reports, with
  matching `*-budget.json` reports. `resident-stack*` also retains early failed
  allocator initialization evidence; it is not accepted performance evidence.
- `generated/fixtures/body-resident-final-vulkan/`: complete native capture and
  final 646-check report. Capture SHA256 remains
  `a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`.
- `generated/demo-qa-resident/`: Chrome report, screenshots and final geometry
  comparison. Final shared library SHA256:
  `8a8bf27ebfd36c4516c3242293fb0823db04ac2c8c1a020ac14a95265949ef2c`.

### Remaining performance work

About 0.26 s of each 0.62 s warm call remains outside traced transfer/compute
calls. CPU samples from the complete final profile still show copying (17.1%),
vector initialization (8.7%), archive reading/validation (7.2%), several finite
validation loops and geometry work. These percentages include model startup,
are user-mode sampled CPU time, and are not percentages of total wall time.
There is no evidence here that lock contention is the main bottleneck.

The next useful step is resident, checked-once decoder/head parameters and
reusable decoder/geometry graphs, eliminating the remaining 611 MB of warm
uploads and duplicated host parameter maps. Then move intermediate geometry
feedback onto the GPU and profile kernel/fusion choices under the same parity
gates. The CPU/GPU alternation remains why this version is around 46%, not 90%
sustained GPU utilization. Real-time and optimized-upstream performance parity
remain open; reducing arithmetic precision or adding artificial GPU work is not
part of these measurements.

### Reproducing a warm profile

Use the same accepted input file multiple times, optionally interleaving another
image. `--worker-input` can repeat up to the profiler's bounded request limit.
The output directory and budget report must be new. The optional `--perf` and
`--shim` arguments enable CPU stacks and the separately built diagnostic shim.

```sh
python scripts/run_bounded.py --report generated/profile-budget.json -- \
  python scripts/profile_body_infer.py --output generated/profile \
  --library build/vulkan-optimized/libsam3d.so \
  --worker-input /path/to/image.input --worker-input /path/to/image.input \
  -- build/vulkan-optimized/bin/sam3d-body-infer --worker \
  /path/to/libggml-vulkan.so Vulkan 0 - \
  /path/to/body-dinov3-f32.gguf /path/to/body-pose-branch-f32.gguf \
  /path/to/mhr-lod1-f32.gguf 1
```

Set the documented strict-F32 environment flags before running the native CLI.
`qa_body_demo.py --warm-repeat` adds a second real UI generation to the standard
browser acceptance test. The profiler reports warm inference timings and
whole-device telemetry separately from initialization and exit.

## First optimization follow-up (historical)

The first optimization pass is implemented and deployed to the existing demo.
The strict F32 math and original tolerance policies are unchanged.

| Same image / fresh native process | Wall time |
| --- | ---: |
| Original debug runner | 39.90 s |
| `vulkan-optimized`: `-O2 -g`, UBSan and assertions | 9.03 s |
| Plus immutable MHR caches and scratch-buffer reuse | 3.97 s |
| Plus checked-once backbone parameter propagation | 3.25 s |
| Actual browser generation including server exports | 3.32 s |

This is about 12× faster, not real time. Filesystem caches are warm; these
numbers include fresh process/model initialization, not a persistent warm
inference session. GPU cooperative matrices/F16 remain disabled. No Nix
derivations were built; existing local compilers and SDK files were used.

Implementation:

- `validated_weights` carries an owned immutable snapshot. Archive `load()`
  validates the bytes and semantic constraints once, and DINO receives this
  typed snapshot rather than repeatedly checking the same large parameter
  tensors. Diagnostic readers supplying ordinary vectors still validate them.
  Shape, mask/period semantics, external-input and computed-output checks stay.
- Archive `pin()` caches immutable snapshots, with a 768 MiB aggregate payload
  budget. Fresh `read()` calls still validate fresh bytes; a cached validation
  bit is never applied to changed files.
- The three large MHR projection matrices stay on the selected backend under
  a separate 768 MiB device-weight budget. The 663,804,000-byte pose matrix is
  now uploaded **once instead of six times per image**. Total uploads fall
  from 8.281 GB to 4.833 GB. Transient sparse-projection data is not accumulated
  in the resident cache across calls.
- Graph primitives lease scratch buffers from a per-session pool (2 GiB
  retained-capacity/graph-request cap). Allocation respects backend chunking;
  the NVIDIA backend's preferred block size is 1 GiB, while DINO's context
  requires about 1.71 GB. The first attempted single-buffer implementation
  rejected that size safely; the final implementation splits the workspace.
  Leases prevent live/nested graphs from aliasing, and buffers are reused only
  after synchronous computation/readback finishes. No GGML submodule edits.

The final traced run records 0.317 s in synchronous graph computation and
0.269 s in uploads, with 2,495 MiB maximum sampled total device memory. These
are host-call timings, not new device-timestamp measurements. Its cgroup peak
was 1,573,167,104 bytes with zero high/max/OOM events. The earlier shim tracked
context allocations; pooled raw allocations were added to subsequent profiling
support, so do not interpret three recorded context allocations as the total
number of device allocations.

Verification: all 529 captured boundary tensors are byte-identical to the old
accepted capture (`a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`),
and all **646 unchanged upstream comparisons pass**. All 19 public result
fields, final mesh/projection comparisons and GLB contents remain unchanged.
All 43 CPU tests pass under ASan/UBSan/LSan; the optimized Vulkan build also
passes the normal 43 tests. The storage test additionally runs on NVIDIA Vulkan.
New tests cover nonfinite immutable weights, reused pointers/layout rejection,
varying-input scratch reuse, nested isolation, pinned caller budgets and file
mutation versus fresh validation. The 96 Python tests pass.

Real Chrome QA uploads the photo, cancels one worker and successfully runs
another, inspects the original/native overlay, reloads/validates every GLB
vertex/index, and exercises history and desktop/mobile scrolling. Screenshots
were visually inspected. No unexpected browser errors occurred. Ignored evidence:
`generated/demo-qa-optimized/` and `generated/fixtures/body-optimized-native-vulkan/`.
Timing evidence is in `generated/diagnostics/optimized-{build-only,storage2,validated}/`
with matching memory-budget reports. Final shared inference library SHA256:
`8954284d1f0e5123efc61facbc4d59bb334d87c0d5ef907404402a102ddb985b`.

Remaining work: resident backbone/decoder weights and activations, optional
parity-only taps, persistent bounded demo worker, and optimized upstream
comparisons. Some smaller component validation/copies still remain. This pass
does not claim that every weight is permanently resident, that every diagnostic
tap has been removed, or that the existing six strict CPU full-model comparison
failures have been resolved.

## Original diagnostic baseline

Measured 2026-09-09. This diagnoses the current demo runner, not optimized
performance parity, full hand refinement, or a real-time claim. No inference
code, precision settings or live server configuration was changed.

## Result

The roughly 40-second latency is reproducible. NVIDIA Vulkan really executes
the graphs, but spends most of the request waiting for CPU work. The demo uses
the `vulkan-ubsan` **Debug build without optimization (`-g`, no `-O`)**, with
UBSan and frame pointers. Strict F32 disables F16 and cooperative matrix paths.

Hardware: Ryzen 9 7900, RTX 5070 Ti 16 GiB. Single person, original 2250×1500
image, explicit box/camera, 512×512 neural crop. Each request launches a fresh
native process. The demo was idle before profiling; no other inference job was
launched concurrently. The GPU sampler measures the whole device, not just this
process. These are warm-filesystem runs, not cold-storage benchmarks.

| Measurement | Result |
| --- | ---: |
| Uninstrumented native runner, GPU telemetry only | 39.90 s |
| Model load / initial validation / backend initialization | 0.72 s |
| `s3d_body_model_infer` stage | 38.96 s |
| Mean / maximum sampled GPU utilization (100 ms polling) | 4.85% / 77% |
| Maximum sampled total device memory | 1,837 MiB |
| CPU-stack + GGML-call instrumented run | 39.90 s |
| Synchronous graph-compute calls, 122 graphs | 0.824 s |
| Upload calls, 1,279 calls / 8.281 GB | 0.669 s |
| Download calls, 369 calls / 278 MB | 0.056 s |
| Allocation/free time, nested calls excluded | 2.410 s |
| Union of all instrumented GGML call intervals | 3.959 s |
| Separate Vulkan device-timestamp run | 40.16 s wall / 0.801 s timestamp intervals |

GB/MB above are decimal bytes. Synchronous host-call timings include dispatch,
driver work and waits: **they are not GPU kernel timestamps**. Device timestamps
come from a separate `GGML_VK_PERF_LOGGER=1` run; the logger adds synchronization
and changes execution timing. Its 122 graph totals corroborate GPU execution
being a small part of this workload, but are not a prediction of optimized FPS.
GPU polling can miss short allocations/bursts. Server photo decoding, export and
browser rendering are outside this native-runner measurement.

## CPU bottleneck and code paths

`perf record -e cpu-clock:u -F 99 --call-graph fp` collected 3,425 samples with
none lost. 2,678 samples (78.19%) have finite-validation functions/loops on their
stack. `std::isfinite(float)` alone is 38.89% self time. Another 6.86% of samples
are float zero-initialization loops. These percentages describe **sampled
user-mode CPU time**, not percentages of total wall time.

- [Archive reads](../src/tensor_archive.cpp) allocate/zero a float vector, read
  its bytes, and scan every weight for finiteness each time.
- [DINO loading](../src/dino_backbone.cpp) checks those same values again;
  [each block](../src/dino_block.cpp) validates them again. Weights and tokens
  are uploaded into a freshly allocated graph per layer; tokens return to the
  host between all 32 layers.
- [The decoder](../src/body_flow.cpp) evaluates MHR geometry on all six layers
  for feedback. [MHR geometry](../src/mhr_geometry.cpp) rereads and uploads the
  **663,804,000-byte `pose.dense.weight` matrix six times**, validating it in
  both the archive reader and projection helper each time. The trace confirms
  all six full-size uploads. Those six uploads alone total 3.983 GB. The
  computation/feedback must remain; redundant weight handling need not.
- [The image pipeline](../src/body_pipeline.cpp) always retains backbone
  boundary taps. Decoder/geometry paths create many intermediate vectors and
  parameter-map copies even when the public result needs only final fields.
- Model/session locks exist, but this is a single serialized request; the
  measured dominant cost is active CPU validation, not an established lock
  contention problem. Merely increasing the CPU thread argument will not
  parallelize these C++ host loops.

Within graph-compute host calls, DINO blocks total 0.509 s, MHR projections
0.196 s and decoder layers 0.041 s. The timestamp run's largest operation
category is MHR's dense pose correction, 0.186 s across six executions. Its
large repeated matrix-vector operation will matter more after removing CPU
overhead. Strict-F32 split dot products and disabled cooperative matrices also
remain important subsequent optimization subjects, with parity gates intact.

## Original recommended implementation order (see follow-up above)

1. Benchmark an **optimized, sanitizer-enabled** Vulkan build (`-O2 -g`,
   assertions and UBSan retained; no fast-math). Re-run numerical acceptance
   before switching the demo. CPU continues using ASan+UBSan; the previously
   documented NVIDIA ICD/ASan startup incompatibility still applies.
2. Validate immutable model weights once when loading them; retain validated
   CPU/device tensors under an explicit memory budget. Preserve external-input
   checks and checks for computed nonfinite results. Do not simply delete all
   validation or cache a validation bit while rereading mutable files.
3. Reuse graph workspaces/device buffers and keep activations GPU-resident
   across layers. In particular, retain MHR weights across decoder callbacks.
4. Separate parity capture from final-result inference, avoiding unnecessary
   taps/readbacks and parameter-map copies. Keep the same mathematical graph
   and layer-level/final-output tests.
5. Use a persistent bounded demo worker/session to amortize loading over
   successive images. This alone cannot remove the current 39-second inference
   cost, because much of the rereading is inside inference itself.
6. Reprofile, then evaluate GPU arithmetic/fusion changes against the original
   outputs and optimized upstream CUDA. Approximately 0.8 s of measured GPU
   intervals is itself still far from 30 FPS; real time is not established.

## Reproduction and correctness

New diagnostic utilities use the existing binary without modifying it:

```sh
cc -O2 -Wall -Wextra -shared -fPIC -Iggml/include \
  scripts/profile_ggml.c -o build/profile-ggml.so -ldl

# Use the same native .input file and model/backend arguments as the demo.
# Every report/output directory must be new. The optional Linux/systemd wrapper
# requires 6 GiB job budget plus 10 GiB available host headroom by default.
GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
python scripts/run_bounded.py --report generated/profile-budget.json -- \
  python scripts/profile_body_infer.py --output generated/profile \
  --perf /path/to/perf --shim build/profile-ggml.so -- \
  build/vulkan-ubsan/bin/sam3d-body-infer \
  build/vulkan-ubsan/bin/libggml-vulkan.so Vulkan 0 'YOUR NVIDIA DEVICE NAME' \
  BACKBONE.gguf BRANCH.gguf MHR.gguf IMAGE.input generated/profile/result.bin 6

perf report -i generated/profile/perf.data --stdio --no-children -g none
```

Omit `--perf` and `--shim` for the baseline. If needed, select the installed
NVIDIA ICD using `VK_DRIVER_FILES`. For device timestamps use another fresh run
with `GGML_VK_PERF_LOGGER=1`, without CPU/shim instrumentation. GPU telemetry
requires `nvidia-smi`; no Nix build/download is required by these utilities.
The shim is intended for the current serialized runner, not concurrent model
sessions. Its raw CSV durations are inclusive: GGML invokes `buffer_free`
inside some allocations, so naïvely summing all rows double-counts time. Use
the union of intervals or subtract nested calls.

All three full runs exited zero and produced **the same complete result SHA256**
as the accepted demo/C API fixture:
`d0930b74259442c3213e6827000bb75c41d884a53445b99fae6c74c59688aa7c`.
The traced scope peaked at 1,550,647,296 bytes of cgroup memory and recorded no
memory pressure-limit or OOM events. Native runner SHA256:
`09ebc17276fd1ac575d00a735e2fd2ad0c5dc03d5f299bf8e521652b044460be`.
Input SHA256:
`3badb322c580a89275c8f4257a70decad730aa0ef30f2816466d4b0e3e88366b`.
Full weight/module identities are retained with the accepted demo fixture.

Ignored local evidence: `generated/diagnostics/profile-vulkan-baseline2/`,
`profile-vulkan-trace/`, `profile-vulkan-device/` and matching `*-budget.json`
reports. The initial `profile-vulkan-baseline/` attempt safely rejected missing
strict-F32 environment variables before inference; it is not a timing result.
This is a profile of host CPU work **during Vulkan inference**, not a full CPU
backend throughput benchmark or comparison with optimized upstream PyTorch.
