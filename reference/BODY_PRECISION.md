# F32 / BF16 benchmarks and native implementation

Measured 2026-09-09 on an NVIDIA GeForce RTX 5070 Ti. Native BF16 is implemented
but **experimental**: final-mesh proximity and small operation tests do not
establish full trained-layer parity. The accepted F32 path and demo default
are unchanged.

## Upstream benchmark

These are the reviewed original Body/DINO modules, with all weights resident,
on the official dancing photograph and the same explicit person box/intrinsics
as native inference. Timed work includes RGB preparation, the 512-square image
encoder, all six pose/mesh-feedback layers, MHR, and final host outputs.
It excludes model loading, detector/segmentation, hand-crop refinement, exports
and browser rendering. This is **body-pose-branch latency**, not Meta's entire
automatic image pipeline.

| Upstream CUDA mode | Warm median | Range (20 calls) | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| F32, eager | 233.94 ms | 233.59–235.40 ms | 4.53 GB |
| F32, compiled backbone | 176.30 ms | 175.77–177.04 ms | 4.49 GB |
| BF16 backbone, eager | 84.52 ms | 84.04–85.38 ms | 2.75 GB |
| BF16 backbone, compiled | 81.65 ms | 81.31–84.34 ms | 2.75 GB |
| BF16 backbone, math-attention control | 120.65 ms | 120.20–122.66 ms | 2.93 GB |

GB are decimal; PyTorch allocated memory is not total device usage. Settings:
PyTorch 2.7.0 / CUDA 12.8, inference mode, six CPU threads, cuDNN autotuning,
TF32 disabled, five warmups then 20 synchronized timings. Automatic SDPA kernel
selection is allowed except in the explicitly labelled math control.
`torch.compile(mode='reduce-overhead', fullgraph=True)` compiles the actual
Body backbone wrapper, which calls `get_intermediate_layers`; the decoder/MHR
remain eager/TorchScript. One compiled graph is recorded in each compiled run.
First compilation/call takes 20.35 s F32 / 21.91 s BF16, excluded from warm time.

Compilation is not bit-exact: maximum final vertex coordinate differences from
the corresponding eager run are 0.000834 mm F32 and 1.059 mm BF16. Reports retain
differences for every final field; finite output and successful graph compilation
are benchmark checks, **not numerical acceptance**.

`reference/benchmark_body.py` is invoked through the trusted reference harness:
add `--benchmark-precision f32|bf16`, `--benchmark-attention auto|math`,
`--benchmark-compile none|backbone`, `--benchmark-warmup 5` and
`--benchmark-repeats 20` to its trained image-pipeline invocation. See
`capture_body_flow.py --help` for required local assets. The harness removes
observer/loading hooks before timing and calls upstream's own BF16 conversion
helper. It does not time the old per-layer disk-streaming capture mode.

Reference execution remains isolated: trusted image, network disabled,
read-only source/models, a writable output directory, no privileges, and
6 GiB RAM/no-swap limits both on the Docker container and outer bounded runner.
Compilation needs a C compiler and executable temporary storage; neither a
Nix derivation rebuild nor an inference-time Python dependency was introduced.
An early attempt compiled `encoder.forward`, bypassed by the wrapper; its
report is explicitly marked invalid, and the harness now rejects zero compiled
graphs. Other failed compiler/container attempts remain recorded, not averaged
into successful timings.

## Native BF16

The API adds opaque-options precision getters/setters, with
`S3D_BACKBONE_F32` as the default and `S3D_BACKBONE_BF16` as the explicit option.
The CLI and persistent worker take a trailing `--bf16`. Both use the same
checked F32 GGUFs: BF16 conversion happens at loading, without another download.
See [API usage](../docs/API.md#vulkan-precision).

Only the image encoder changes precision, matching upstream's published
`USE_FP16: true` / `FP16_TYPE: bfloat16` configuration. Linear weights are stored
as BF16; other constants and activations have explicit BF16 rounding at original
operation boundaries, with F32 carriers for GGML operations that lack BF16
inputs. Attention's math intermediates and RoPE trigonometry are F32. Decoder,
MHR and public output buffers retain their previous precision. This is not F16
or merely BF16 storage with unrounded F32 model arithmetic.

| Native Vulkan mode | Warm median | Warm range |
| --- | ---: | ---: |
| Accepted F32 baseline | 395 ms | 393–404 ms |
| Initial BF16 implementation | 269 ms | 267.5–269.8 ms |
| Final BF16 build, explicit resident-input rounding | 274 ms | 271.7–279.4 ms |

The BF16 profile contains 12 alternating requests for the reference dancer and
a second photograph, excluding the first call from the warm summary. Every
repeat of a given input has the same complete result hash. It uses UBSan,
assertions, one native CPU thread, a resident encoder and the unchanged strict
Vulkan flags. Timings are inference stages, not HTTP/upload/export times; the
small tracing overhead and different host preprocessing implementations remain.
Cold loading/conversion takes 5.24 s and first inference another 1.58 s.

Warm graph-call time falls to about 182 ms; uploads are 98.14 MB / 218 calls,
downloads 42.94 MB / 235 calls, with no warm backend allocations. Whole-device
warm utilization averages 65.8% (100% peak, 30 samples); sampled VRAM peaks at
3,090 MiB including the desktop, versus 4,686 MiB in the prior F32 profile.
This is **not 90% sustained utilization or performance parity**. The bounded
native profile peaks at 4.48 GiB cgroup RAM, with no high/max/OOM events.

BF16 tensor-core acceleration is not yet independently enabled: current GGML's
cooperative-matrix mode can implicitly convert F32 operands to F16. The strict
flags remain mandatory to protect the F32 decoder. Selectively accelerating
BF16 while preserving F32 operations, reducing remaining host work/transfers,
and reusing decoder graphs are the next performance work.

The final build additionally rounds direct resident-stack inputs, making its
internal entry point consistent with streamed BF16 execution. Re-profiling all
12 inputs gives 273.86 ms median and unchanged complete output hashes. Its
31 warm GPU samples average 57.6% (100% peak), illustrating the noise in short
utilization windows, not satisfying the 90% target. The final library SHA-256 is
`329196cc89991e67cb382dc23df5b86ba0cecf863f08bc9620b918a84f23a60e`.

## Numerical and safety checks

- Actual original tiny BF16 transformer fixture: all BF16-valued operation
  outputs match exactly on CPU and NVIDIA Vulkan; F32 trig/logit/softmax taps
  retain strict numerical checks. A separate actual CUDA Conv2d fixture guards
  its two BF16 rounding points. Synthetic fixtures do not imply trained parity.
- The first trained discrepancy exposed a real implementation error: upstream
  BF16 convolution rounds **before** its separate bias add. The native graph
  now does so too; an upstream control confirms the separate-bias result exactly.
- All 36 trained backbone boundaries and all 32 blocks' operations were captured
  with their own intermediates. Original traced/untraced results are exact.
  After the fix, 32 of 1,310,720 patch values still differ, with maximum 0.015625
  and relative L2 2.68e-5. The final feature error is 0.109375 / relative L2
  0.00908. **All 36 boundaries fail the strict 1e-4 / 2e-5 test.** These failures
  remain failures; a defensible BF16-specific policy and further operation-level
  investigation are unfinished. No F32 tolerances were weakened.
- On the dancer, native Vulkan final vertices differ from upstream eager BF16
  auto-SDPA by 0.840 mm maximum coordinate / 0.306 mm mean vertex distance.
  Against the explicit BF16 math-SDPA control this is 1.671 / 0.513 mm. Both
  comparisons are retained; selecting the smaller one does not establish parity.
- Full native BF16 CPU image-to-mesh inference exits successfully with
  ASan/UBSan/LSan. Its final vertices differ from the CUDA eager reference by
  0.772 mm maximum coordinate / 0.397 mm mean distance. This is not a matched
  CPU upstream parity claim. The run takes 112 s including loading and sanitizer
  overhead, peaks near the 5 GiB soft cap, records memory-high throttling but no
  hard-limit/OOM event. It is **not a fair CPU performance benchmark**.
- F32 remains unchanged: all 646 full-model checks pass, with all 529 tensors
  byte-identical. Existing six strict full-model CPU F32 failures remain open.
- Updated non-GGUF C API passes 100,000 libFuzzer cases with ASan/UBSan. All 46
  native regressions pass in both builds (CPU ASan/UBSan/LSan and Vulkan UBSan),
  with actual NVIDIA BF16 block/convolution/repeated-backbone checks. Normal
  regressions cover option defaults, invalid precision/state preservation, null
  pointers, caller error buffers, BF16 conversion rejection and repeated resident
  graph inputs. No UI/default-model change is made by this feature.

## BF16 goal follow-up: original-only calibration and profiling

The earlier strict-F32 BF16 failures above remain recorded. Separate frozen
policies now use only original execution controls—not native candidate errors:

- `bf16-backbone-policy-v1.json`: two times the original CPU/math-CUDA and
  auto/math-CUDA variation, with 5% relative-L2 and 10%-of-peak absolute safety
  ceilings. Original captures are repeatable and observer-neutral. All 36
  baseline native Vulkan stages pass for the named dancer image.
- `bf16-body-policy-v1.json`: two times original CUDA math/automatic or
  compiled/automatic variation, with independent ceilings (10 mm geometry,
  2 pixels projections, 5% relative-L2). Baseline CPU and Vulkan pass this final
  own-intermediate gate; Vulkan hand boxes/logits miss their numerical limits
  but are explicitly non-blocking for the scoped goal. Nonfinite hands,
  incorrect shapes and invalid topology always fail. Hand outputs are retained.

These are one-image gates, not a complete trained-layer/multi-image acceptance
claim. Both calibration scripts have no native-candidate argument; checked-in
policies preserve original hashes and control measurements. Negative tests
reject reordered/scaled/offset/nonfinite signals and invalid hand/geometry data.

Warm CPU-stack and GPU-timestamp profiling identifies about 162 ms of encoder
GPU time: 97 ms BF16 matrix products, 41 ms attention/softmax, and the remainder
elementwise/copy kernels. Overall graph calls take 182 ms; transfers another
12–14 ms. Remaining host time includes transposes, skinning, preprocessing,
allocation and parameter-map construction. Timings with GPU logging are
instrumented and are not substituted for the uninstrumented latency summary.

`SAM3D_BF16_FLASH_ATTENTION=1` is an explicit experimental fused-attention path.
It measures about 262 ms warm (12 alternating requests, first excluded), versus
274 ms baseline. Final output passes the frozen body gate with one reported hand
logit miss, but seven encoder absolute-error gates fail starting at block 22
(relative-L2 gates pass). Those failures prevent declaring this path accepted.
The default math path and live F32 demo are unchanged. Fused logits cannot be
observed: requesting full math-operation taps in this mode fails explicitly.

The experimental [GGML patch](../patches/ggml/README.md) adds strict F32 matrix
pipelines alongside BF16 CM2 tensor cores, only in a build-tree copy. It needs
real-device operand-range/strided-matrix tests and complete model rechecks before
acceptance. It does not mutate the upstream submodule or rebuild Nix packages.

Evidence: `generated/diagnostics/bf16-{baseline-stage,native-final,cpu-final}-acceptance-v1.json`,
`bf16-{cpu-stacks,gpu-kernels,flash-scalar}/`, and
`bf16-flash-{stages,final}-acceptance-v1.json`.

### Selective BF16 tensor cores: measured follow-up

The build-copy patch now passes 12 actual NVIDIA F32 matrix tests: dense and
strided inputs, vector/matrix/batched shapes, values beyond F16 range and below
its resolution. The first patch incorrectly selected the F32-input shader that
internally uses F16; this guard caught it. The corrected patch explicitly selects
the `_fp32` shader variants. The pristine upstream submodule is unchanged.

| Vulkan BF16 mode, UBSan retained | Median | Range, 20 timed calls | Graph calls |
| --- | ---: | ---: | ---: |
| Tensor cores, math attention | 208.60 ms | 206.02–214.48 ms | about 116 ms |
| Tensor cores, fused attention (experimental) | 168.60 ms | 167.23–176.15 ms | about 78 ms |

Both use five warmups followed by 20 alternating-image calls, with identical
complete-output hashes on repeated inputs. Uploads/downloads remain about
4.4/8.0 ms, 98.14/42.94 MB per call. Sampled warm GPU utilization averages
56.5%/49.2% respectively (41/34 samples); these are short, whole-device samples,
not sustained utilization claims. Neither mode reaches the 80–85 ms target.

Math attention passes all 36 frozen encoder gates and all blocking final-body
gates on the reference dancer; hand logits are the only nonblocking final miss.
Fused attention passes every final-body gate, but fails six encoder absolute
gates starting at block 22. Relative-L2 gates and final features pass. This is
not declared full layer parity and the demo remains F32. A separate performance
build without sanitizer instrumentation is being measured; sanitizer builds
are retained for correctness. No fast-math or precision change is implied.

The updated host profile identifies feedback sampling, layout transposes,
finite-value scans, allocation/copies and preprocessing in addition to driver
fence waits. All 46 normal native UBSan regressions pass. Full trained F32
revalidation under actual CM2, isolated fused-operation checks and held-out
official-image/visual checks remain required.

Evidence: `generated/diagnostics/bf16-cm2-{math,flash,flash-host}-v5/`,
`bf16-cm2-{math,flash}-{stages,final}-acceptance-v5.json`, and
`bf16-cm2-precision-v5-budget.json`. Backend and runner/library hashes are in
the profiles; the corrected patched-source fingerprint is
`cdb19f85b9d833fa0ca0ca8d31782c73657facc4abec8358c830293d0b8116c7`.

### Performance build, resident layouts and isolated attention

A separate `-O2 -g` performance build disables sanitizer instrumentation, not
assertions, F32 protection or finite-value validation. With tensor cores/fused
attention it measures **134.00 ms** (132.36–136.06 ms). All 25 complete results
are byte-identical to the UBSan build, and all 46 normal tests pass. Existing
ASan/UBSan builds remain the correctness configurations, not replaced by this
benchmark configuration.

Keeping final encoder normalization/prefix removal/layout on the GPU, then
moving the camera encoder's input/output transposes onto the GPU, reduces this
to **126.83 ms** (125.93–128.44 ms). Both profiles use five warmups and twenty
timed alternating-image calls. Complete result hashes remain identical. The
resident encoder's 36 captured stages are byte-identical to the previous
streamed capture. Warm transfers are now 92.86 MB upload / 37.65 MB download,
215/234 calls, about 4.2/7.4 ms; graph submissions fall from 85 to 84.
Warm whole-device GPU samples average 55.8% (67% maximum, 26 samples).

Timestamped kernel profiling (separate, with logging overhead) measures 57.89 ms
encoder GPU work and 20.84 ms elsewhere. Encoder breakdown: 31.43 ms matrix
products, 8.06 ms casts/copies, 6.08 ms adds, 4.71 ms multiplies and 3.32 ms fused
attention. Elementwise fusion and remaining host/feedback/dispatch work are the
next targets. This is still **not 80–85 ms or sustained 90% utilization**.

All 32 isolated trained attention operations pass a separately frozen
original-only policy. Actual upstream post-RoPE Q/K and V are shared with
original CPU math, CUDA math, Flash and efficient SDPA controls. Repeated
original results are exact, and isolated CUDA math matches the recorded original
model attention output exactly. Native relative-L2 errors are within original
kernel variation; negative controls reject layout, scale, offset and nonfinite
errors. The policy is `bf16-attention-policy-v1.json`. This isolates an operation;
it does not erase the six full-trajectory encoder absolute-limit misses.

The additional official yoga photograph now has original eager/math/compiled
BF16 captures (84.68/120.70/81.08 ms medians) with exact shared RGB, box and
intrinsics. Original-only calibration rejects this case: the derived joint
rotation limit exceeds its 0.1 ceiling, and original projection differences
already exceed two pixels. No yoga policy was written or silently widened.
Native repeated yoga outputs were captured, but are not declared accepted.
Representative-image numerical and visual acceptance remains unfinished.

The current CPU build passes all 46 tests with ASan/UBSan/LSan. An initial test
invocation inside the restricted sandbox failed because LSan cannot inspect
threads under ptrace; the unchanged tests pass outside that restriction in the
bounded runner. Python regressions now total 103, including nearest-rank p95
tests. The profiling harness also rejects incomplete worker result/timing sets. Earlier profiles'
median/range are unchanged; their 20-call p95 used the maximum, so use the
recorded individual timings or the corrected harness for precise percentiles.
The final updated Vulkan capture still passes all 646 strict F32 comparisons,
with every one of its 529 tensors byte-identical to the accepted baseline.
The refreshed non-GGUF C API fuzzer completes 100,000 cases with ASan/UBSan,
no findings and a clean exit (`bf16-current-api-fuzz-100k-budget.json`).

Evidence: `generated/diagnostics/bf16-{performance-flash,gpu-layout-flash,gpu-layout-kernels,resident-tail-flash}-v1/`,
`bf16-attention-native-v1/`, `bf16-cm2-resident-math-stages-acceptance-v1.json`,
`bf16-gpu-layout-final-acceptance-v1.json`, `bf16-current-cpu-asan-tests-budget.json`,
and `generated/benchmarks/body-yoga-bf16-{eager,math,compiled}/`.

## Exact rounding fusion and feedback locality

The second build-copy GGML patch combines eligible F32 → BF16 → F32 casts
into one integer-rounding Vulkan shader, selected with
`GGML_VK_FUSE_BF16_ROUND=1`. It preserves the actual BF16 arithmetic boundary.
Twenty-four device cases check exact bits (including subnormals, ties, signed
zero and overflow), non-multiple workgroup sizes, strided fallbacks and retained
shared/observed intermediates. The GPU logger confirms the expected fusion and
fallback decisions. All twelve strict F32 matrix precision cases still pass.

The latest sequential, alternating-image performance measurements are:

| Change | Warm median | Range |
| --- | ---: | ---: |
| Previous GPU layout changes | 126.83 ms | 125.93–128.44 ms |
| Exact BF16 cast-pair fusion | 125.31 ms | 123.90–128.09 ms |
| Reuse contiguous-channel feedback image | **118.96 ms** | **118.01–121.11 ms** |

Each uses five warmups and twenty timed complete image-to-mesh requests,
resident weights, the same no-sanitizer `-O2 -g` performance configuration,
CM2 BF16 and fused attention. The final p95 is 120.50 ms (nearest rank).
Every one of the 25 result files remains byte-identical to the previous
126.83 ms implementation, including alternating-input repeats. No layer,
geometry-feedback iteration or final output was omitted. The demo remains F32;
these are native inference measurements, not browser round-trip measurements.

Warm CPU profiling at 999 Hz with DWARF unwinding (the performance build has no
frame-pointer guarantee) identified feedback sampling as 9.99% of CPU samples.
It previously read channels 4096 bytes apart; the decoder already had a
contiguous-channel copy of the same image. Reusing that copy drops feedback
below 1% of CPU samples, without changing interpolation coordinates, corner
accumulation order or masking. Two-way callers retain a separate immutable
sampling image rather than sampling their updated context. All original small
feedback tensors agree exactly between the two storage layouts.

Separate timestamped GPU profiling measures 55.25 ms encoder and 20.85 ms
other GPU work, versus 57.89/20.84 ms previously. The encoder fuses 545 cast
pairs per image; fused casts plus remaining copies cost 5.71 ms versus 8.06 ms.
Matrix products still cost about 31.2 ms, adds 6.04 ms, multiplies 4.71 ms and
attention 3.29 ms. Logging adds overhead and these timings are not the
uninstrumented latency. Warm ordinary graph-call time is about 75.8 ms over
84 submissions; 92.86 MB uploads / 37.65 MB downloads remain. Whole-device
telemetry averages 61.9% utilization, peaks at 69% (24 warm samples): neither
80–85 ms nor sustained 90% utilization has been achieved.

Both updated CPU ASan/UBSan/LSan and UBSan suites pass all 46 normal tests;
103 Python tests pass. The full trained F32 capture passes all 646 comparisons,
and all 529 tensors remain byte-identical to the prior accepted capture. The
fused-rounding resident **math-attention** encoder passes all 36 frozen gates;
the latest **fused-attention** final body output passes all 18 final-field gates.
These exact-preserving optimizations do **not** resolve the six previously
reported fused-attention trajectory misses or held-out-image acceptance.
No numerical tolerance changed. Remaining targets include decoder/feedback
residency and copies, host layout/validation costs and encoder elementwise work.
The rebuilt non-GGUF public C API fuzzer also completes another 100,000 runs
under ASan/UBSan, with no findings (6.26 seconds, 326 MB cgroup peak;
`bf16-feedback-layout-api-fuzz-100k-v1-budget.json`).

Ignored evidence: `generated/diagnostics/bf16-round-fusion-{performance,host}-v1/`,
`bf16-feedback-layout-{performance,host,kernels}-v1/`,
`bf16-feedback-layout-final-acceptance-v1.json`,
`bf16-round-fusion-stages-acceptance-v1.json`, and
`generated/fixtures/bf16-feedback-layout-full-f32-v1/parity-v1.json`.
The patched-source fingerprint is
`6159e1a30f19aa22c0a3495f9b1dc4a8c28a2602d267beaeac4e009dfbd796a8`;
the upstream GGML submodule is unchanged. Bounded builds/captures finished
without OOM events; no Nix derivations were rebuilt.

## Per-inference decoder image residency

The one-way decoder now uploads one immutable image/PE snapshot for its six
layers. This is scoped to one inference, not a cache keyed by pointers or image
identity. Snapshots own device storage, retain their scratch leases until every
consumer finishes and reject cross-session/shape mismatches. Host image/PE
arguments must be empty when a snapshot is supplied, so conflicting inputs are
never silently ignored. Two-way composition retains its original changing-image
path. Public C API signatures are unchanged.

This reduces uploads from 92.86 to **40.43 MB** per image (215 to 205 calls),
about 4.2 to 1.85 ms, and avoids repeated image/PE finite scans. The measured
warm median is **113.37 ms**, range 112.75–115.57 ms, p95 115.06 ms: five
warmups and twenty timed alternating inputs in the same performance build.
All 25 outputs are byte-identical to the 118.96 ms baseline. Downloads remain
37.65 MB/234 calls and graph count remains 84; this is not yet graph reuse or
the 80–85 ms target. Whole-device GPU samples average 61.7%, peak 70% (23 samples).

All 46 CPU ASan/UBSan/LSan tests pass. The decoder test compares every original
boundary with and without resident image/PE, including two-way single layers,
broadcast/absent PE, overlapping snapshots, caller-buffer mutation and foreign
session rejection. Complete Vulkan UBSan repeated outputs remain exact. The
new F32 capture passes all 646 comparisons and all 529 tensors are byte-identical
to the previous capture. Existing BF16 attention trajectory/held-out-image
limitations remain; this change does not alter any numerical tolerance or the
F32 demo default.

Evidence: `generated/diagnostics/bf16-decoder-resident-{ubsan,performance}-v1/`,
`bf16-decoder-resident-asan-tests-v2-budget.json`, and
`generated/fixtures/bf16-decoder-resident-full-f32-v1/parity-v1.json`.
The refreshed non-GGUF C API fuzzer completes 100,000 ASan/UBSan cases without
findings (`bf16-decoder-resident-api-fuzz-100k-v1-budget.json`, 6.54 seconds,
339 MB cgroup peak and no OOM events).

## Additional official rider-image control

The official `sample2/input_bbox.png` now has original BF16 eager, math-attention
and genuinely compiled-backbone controls: **83.89 / 119.71 / 81.10 ms** medians,
each with five warmups and twenty measurements. Identical original pixels,
explicit box, intrinsics, configuration and weight identities are checked.
The original-only calibration passes the existing safety ceilings; its policy
was frozen before running native rider inference in
`bf16-body-rider-policy-v1.json` (SHA-256
`4dba2bc8fcde92da6a60251e62c15b0f01ae69e9f6502fb6721e1dadf715c1e8`).
The existing dancer/attention policies and the rejected yoga calibration are
unchanged. A hash/formula/negative-control regression covers the new policy;
Python tests now total 104.

Native passes all 18 rider final-output gates, with mean/max vertex distance
**0.475 / 1.723 mm**, mean/max joint distance **0.443 / 1.659 mm**, and maximum
projected-vertex coordinate error **0.371 pixels**. These are complete
own-intermediate outputs, not a decoder supplied with reference features.
Twenty-five alternating rider/dancer requests give one exact result hash per
image and a 113.76 ms warm median. This is a second final-output numerical case,
**not** held-out layer-by-layer or visual/demo acceptance; those remain open.

Evidence: `generated/benchmarks/body-rider-bf16-{eager,math,compiled}/`,
`generated/diagnostics/bf16-rider-resident-performance-v1/`,
and `bf16-rider-final-acceptance-v1.json`.

## Saved-output visual verification

Actual headless Chrome now renders verified dancer and rider captures in front,
side and oblique views. All six screenshots were inspected: original/native
body, arm and wrist placement agree without independent alignment, and all
browser/GL/nonempty-render checks pass. The exact original RGB and topology
are verified. The frozen original reference outputs and tolerances are unchanged.
This is saved-output visual evidence, not demo upload/inference/export QA.
See [BODY_BF16_VISUAL.md](BODY_BF16_VISUAL.md) for provenance, limitations and
reproduction; 106 Python tests pass including new provenance rejection cases.

## Earlier local ignored evidence

Successful upstream reports are under `generated/benchmarks/body-{f32,bf16}-eager`,
`body-{f32,bf16}-compiled-exec` and `body-bf16-math`. Each records settings,
source/weight hashes, output hashes and the actual warm timings. Pinned originals
are Body `b5c765a0d89d789985e186d396315e7590887b94` and DINO
`6876159a11b4df116f30f667f8c9888617df0751`; GGML remains unmodified at
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`.

Native profiles: `generated/diagnostics/bf16-native-{initial,final}/`, including
GPU/GGML traces and unchanged result hashes; the initial directory also contains
both original final-field comparisons. Trained captures:
`generated/fixtures/body-trained-{backbone,operations}-bf16-cuda/`; strict report:
`body-trained-backbone-bf16-cuda/strict-native-vulkan.json`. F32 recheck:
`generated/fixtures/bf16-changes-f32-vulkan/parity-trained-v1.json`. CPU sanitizer
result/budget: `generated/diagnostics/bf16-full-cpu-asan*`.

Use `scripts/compare_body_precision.py` for all public final-field diagnostics
and `scripts/compare_backbone_binary.py` for strict captured boundary assertions.
Neither tool silently intersects field sets or substitutes a mesh-only test for
layer-level acceptance. Captures and weights remain ignored, not repository assets.

### Compiled encoder instrumentation audit

`capture_trained_backbone.py --compile` now compiles the actual original Body
wrapper entry point with `fullgraph=True`, not the unused DINO `forward`.
Both unobserved and observed versions must compile, with no eager fallback;
the usual three bit-exact unobserved repeats and observer-neutrality requirement
remain mandatory. Stage taps stay on-device until the full observed forward
returns, avoiding host-copy graph breaks.

The compiled dancer experiment is **rejected**, not a new calibration control.
With deterministic cuBLAS (`CUBLAS_WORKSPACE_CONFIG=:4096:8`), the unobserved
model repeats exactly and each version executes one compiled graph. However,
making the stages observable changes 391,531 final-feature elements:
maximum absolute difference **0.03125**, relative-L2 **0.00282802**. Compilation
can fuse differently when intermediate results become outputs. The harness
records a rejection and both final tensors, but publishes no accepted stage
fixture. The initial attempt without the cuBLAS workspace setting exits with
PyTorch's deterministic-execution error; it is not an OOM or a parity result.

Evidence: `generated/fixtures/body-bf16-control-cuda-compiled-v3/rejected.json`
and its hashed `rejected-observation.safetensors`, plus
`generated/diagnostics/bf16-backbone-compiled-v{1,2,3}-budget.json`.
Existing eager layer controls, uninstrumented compiled full-body controls and
all frozen numerical limits remain unchanged. This audit did not waive the
six native encoder absolute-limit misses; the shader correction below resolves
them independently.

The eager capture regression (`body-bf16-control-cuda-auto-audit-v1`) still
passes three exact unobserved repeats and observer neutrality. Its complete
36-stage safetensors hash remains identical to the frozen automatic-CUDA
control: `8aee354fd2a65224ee944a48631f728583af507ff48fcd453e9df05ef2835de6`.

### BF16 fused-attention denominator correction

The original GGML CM2 BF16 shader forms its softmax row sum by multiplying
the already-rounded BF16 probability matrix by ones. This quantizes the
denominator as well as the numerator's P×V operands. Build-copy patch `0003`
instead reduces the F32 probability accumulator before that conversion,
following the precision split in
[FlashAttention's softmax](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/softmax.h).
Padding is still cleared first; non-BF16 shaders are unchanged.

The new three-case real-device regression distinguishes the algorithms with
constant signed V and varied logits. The old module fails immediately
(`expected=0.999459`, `got=1.000000`); corrected UBSan and performance modules
pass all three cases, plus the existing 12 F32 and 24 bit-exact BF16 cast cases.
All **32 trained isolated attention checks and all 36 complete dancer encoder
stages now pass the unchanged frozen policies**. The six previous encoder
absolute-limit failures are resolved. All **646 strict F32 checks pass**, and
all 529 captured tensors retain their previous exact hash.

Complete final-output checks remain image-specific:

| Original image | Final result after correction |
| --- | --- |
| Dancer | All 18 checks pass; vertex distance mean/max **0.166/0.712 mm** |
| Rider | 3D geometry passes, vertex maximum **2.364 mm**; two projection relative-L2 checks fail |

Rider keypoint projection relative-L2 is **0.000414667**, versus the frozen
**0.000369697** limit (12.16% over); vertex projection is **0.000335855**,
versus **0.000335587** (0.080% over). Their maximum absolute errors remain
within the frozen limits (0.387/0.390 pixels). The rider hand-box head also
misses its relative limit and remains reported/non-blocking. These are not
reclassified as passes. The previous all-green rider output predates the
correction; representative-image acceptance and the BF16 demo switch are open.

Performance is effectively unchanged: **113.99 ms median**, **113.10–115.79 ms**,
**115.62 ms p95**, with five warmups and twenty timed alternating dancer/rider
inferences. All 25 outputs are byte-identical between UBSan and performance
builds; repeated same-image outputs are exact across intervening other images.
The UBSan build measures 148.82 ms. Warm GPU telemetry averages **66.4%**,
peaks at **70%** (23 samples). Per-image transfers remain **40.43 MB upload**,
**37.65 MB download**, with **84 graph submissions**. This numerical fix is
not an 80–85 ms result or sustained 90% utilization.

Both normal native suites pass all 46 tests (CPU ASan/UBSan/LSan and the rebuilt
UBSan configuration); 106 Python tests pass. Builds and full-model checks were
serialized and capped, with no recorded OOM events; no Nix derivations were
built. The GGML submodule remains pristine. Patched source fingerprint:
`8f989bbb9b9efb5de01d99ca570dbb99307a5bae1847f8359162f9c5dd6fe6db`.

Evidence: `generated/diagnostics/bf16-denominator-{attention-v1,performance-v1}/`,
`bf16-denominator-stages-acceptance-v1.json`,
`bf16-denominator-{dancer,rider}-acceptance-v1.json`, the correspondingly named
budget reports, and `generated/fixtures/bf16-denominator-full-f32-v1/parity-v1.json`.
Corrected dancer/rider result hashes are respectively
`377c02b8466e76c5924553461c4947da63f8ba7fbe560a3ee984b610f4cd53b5` and
`a433bb61896e971ba31cd42c47b6af5dd103b4d23dea84e231081a04f1498e5b`.

### Held-out rider encoder controls

Original rider CUDA math, CUDA automatic attention and CPU math captures now
each pass three exact unobserved repeats and observer neutrality. All contain
the same 36 stages and normalized image, and retain source/input/weight hashes:
`generated/fixtures/body-rider-backbone-{cuda-math,cuda-auto,cpu-math}-v1/`.

The legacy 2×-headroom calibration is rejected at `04.features` only. Its
reference peak is **4.40625**; CPU/math-CUDA maximum discrepancy **0.266602**
and relative-L2 **0.0128032**, automatic/math-CUDA **0.137695** and **0.0099742**.
The measured discrepancies fit the existing ceilings, but doubling the CPU
maximum would reserve **0.533203**, over the unchanged **0.440625** ceiling.
The rejected attempt is retained in `bf16-rider-backbone-policy-v1-budget.json`;
it produces no accepted policy file.

For a new held-out policy only, `calibrate_bf16_backbone.py --clip-headroom`
supports `min(unchanged safety ceiling, max(legacy floor, 2× original variation))`.
Every measured original control must itself fit the same ceiling, otherwise
calibration still fails. This reduces unused margin, never raises a cap.
It is selected before capturing/evaluating native rider layers and does not
replace any frozen dancer or final-body policy. The default legacy calibration
still rejects this case. Unit tests check both behaviors, rejection of controls
beyond the caps, invalid values, and unchanged limits for well-conditioned
controls. No native result is an input to either calibration mode.

The new policy is frozen as `bf16-backbone-rider-policy-v2.json`, SHA-256
`33f485de2361ff8521ca7105d8bc8edc86299b20599cea21e34300993cd887f0`.
Its complete formula, all original measurements and source hashes are recorded;
the final-feature absolute limit is 0.440625. Tests lock the file hash and
recompute every threshold, verify each original control fits, and reject
nonfinite, permuted, zeroed and offset candidates.

### Rejected BF16 softmax-offset experiment

Removing GGML's F16-protection softmax offset from the BF16 CM2 path was tested
after the denominator correction. The nine focused tests passed, and correctly
rejected the earlier shader, but the complete dancer encoder regressed at
blocks **22–26 and 29**; rider failed at **22–25** under its pre-frozen policy.
The dancer final keypoint projection also failed; rider still failed keypoint
and vertex projections. No numerical limits were changed to accept this.

This experiment is excluded from builds. Its patch is retained only as
`reference/experiments/rejected-bf16-unshifted-softmax.patch` alongside ignored
evidence `generated/diagnostics/bf16-unshifted-{dancer,rider}-stages-acceptance-v1.json`,
the final-output acceptance reports and `bf16-unshifted-body-ubsan-v1/`.
Its source fingerprint was
`e8d08941b93cc7305fef6ba1604979a15a99184b0100496d5120c99ea71dc782`.
This demonstrates why isolated operation tests do not establish full-model
parity, even for a mathematically invariant change.

### Probability residual and precise prefix: two-image encoder acceptance

Build-copy patch `0004-vulkan-bf16-flash-probability-residual.patch` retains the
original softmax offset and adds a second BF16 cooperative-matrix P×V product
for the F32 probability's rounding residual. Q/K/V stay BF16; denominator and
accumulation stay F32. This is a closer approximation to original math-SDPA,
not a claim of bit-exact F32 probabilities. All 32 trained isolated attention
checks pass, with mean relative-L2 error **0.000119787**, versus **0.00119840**
for denominator-only attention; all 32 improve. Residual correction alone still
fails six dancer and eleven rider complete encoder maximum-error checks, despite
both final-body gates passing. Those failures are retained as evidence.

`SAM3D_BF16_PRECISE_PREFIX=1`, alongside the experimental fused-attention flag,
uses the existing F32 math-SDPA path for the five class/register query rows,
whose magnitudes amplify small attention errors. The other 1024 query rows use
the residual-corrected fused kernel. Every query still attends to every original
key/value; concatenation restores the original token order before the same BF16
rounding boundary. No tokens, interactions or outputs are removed.

With this split, **all 36 complete encoder-stage gates pass on each of dancer
and rider**, under their previously frozen policies. Final-body policies are
unchanged and also pass their blocking gates:

| Image | Vertex distance mean / max | Joint distance mean / max | Non-blocking report |
| --- | --- | --- | --- |
| Dancer | 0.300 / 0.969 mm | 0.271 / 0.927 mm | `hand_logits` misses its limit |
| Rider | 0.509 / 1.468 mm | 0.483 / 1.345 mm | None; all 18 fields pass |

All **646 strict F32 checks** pass and all **529 captured tensors** retain
SHA-256 `a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`.
The real NVIDIA precision executable passes **58 cases**: 12 F32, 24 exact BF16
casts, 18 probability/shift/padding cases and four whole-block prefix cases.
The latter cover batches 1/2 and prefix counts 1/5, compare prefix results
bit-exactly against full math attention, and perturb only image tokens to
verify prefix outputs retain their image-key/value dependence. Python regression
is 108 passing tests. The first prefix implementation hit a non-contiguous
`ggml_scale` assertion; making the sliced queries contiguous fixed it. Its failed
run is not counted as evidence of success or OOM.

The complete UBSan candidate measures **156.59 ms median**, **155.38–158.54 ms**,
**158.23 ms p95** over five warmups and twenty timed alternating dancer/rider
requests. GPU compute calls total approximately **83.8 ms** per warm image;
there are still 84 submissions and 40.43/37.65 MB upload/download. The older
114 ms unsanitized result is not a measurement of this new precision split.
Current unsanitized timing, updated visual QA and the BF16 demo switch remain
pending at this checkpoint; **80–85 ms is not yet achieved**.

Evidence: `generated/diagnostics/bf16-precise-prefix-body-ubsan-v1/`,
`bf16-precise-prefix-{dancer,rider}-acceptance-v1.json`, dancer stage acceptance
`v2` and rider stage acceptance `v1`, `bf16-precise-prefix-precision-v1-budget.json`,
and `generated/fixtures/bf16-precise-prefix-full-f32-v1/parity-v1.json`.
Residual-only evidence is separately retained under `bf16-probability-residual-*`.
Current patched GGML source fingerprint:
`b517955e926240e1fd170902dcd3b98fa3229075d202305f08401d41e181146e`.

### Precise-prefix performance, CPU and demo follow-up

The current unsanitized build measures **121.54 ms median**, **120.81–123.21 ms**,
**123.06 ms p95** (five warmups, twenty timed alternating dancer/rider images).
All 25 full outputs are byte-identical to the UBSan candidate; repeats remain
exact across intervening different inputs. All 58 real-device precision tests
also pass in this build. Warm GPU utilization averages **71.6%**, peaks **74%**
(25 samples). The denominator-only 114 ms result is historical, not the current
quality candidate. Evidence: `bf16-precise-prefix-performance-v1/`.

CPU ASan/UBSan/LSan passes all **36 held-out rider encoder stages** and the
unchanged final-body gate (vertex mean/max **0.605/2.328 mm**; hand boxes are a
reported non-blocking miss). Full CPU inference completed in 83.32 seconds,
peak cgroup RAM **3.15 GB**, without OOM or sanitizer errors. All 46 normal CPU
sanitizer regressions and 108 Python tests pass. Evidence:
`bf16-precise-prefix-rider-cpu-{body,stages}-v1*` and their acceptance reports.

The updated dancer/rider saved-output Chrome views and the actual BF16 demo
upload/render/overlay/export workflow now pass; see [visual/demo QA](BODY_BF16_VISUAL.md).
The demo runs the UBSan build, retaining F32 as a selectable launch option and
the default for other installations. Warm browser inference+export is **0.205 s**;
the fresh-worker job is **4.258 s**, including model validation/loading and export.
This is not the 121.54 ms unsanitized native benchmark.

A warm-only 999 Hz CPU-stack profile spans 20 complete requests and loses no
samples. Largest self-time entries include GGML's fence wait (33.89%), memory
copy (6.32%), prompt encoding (4.57%), image preparation (4.27%), CPU skinning
(3.96%) and conditioning-layout flattening (2.89%). These are CPU-sample shares,
not wall-clock percentages. Synchronous GGML graph calls account for about
83.8 ms per request, including CPU dispatch/waits; they are not GPU timestamps.
The encoder graph accounts for about 61.1 ms; the six decoder layers about
9.85 ms. Transfers remain 40.43 MB up / 37.65 MB down across 84 submissions.
Profile artifact: `bf16-precise-prefix-host-profile-v1/`; warm monotonic interval
2025051.491717007–2025053.958088971. This motivates removing the conditioning
layout round-trip next; further GPU/host optimization is needed for 80–85 ms.

### Eliminate the conditioning layout round-trip

The camera encoder and dense positional encoding already compute token-major
arrays. Normal inference now retains that layout through conditioning instead
of transposing each to NCHW and back before the first decoder layer. Three large
host transposes, one GPU transpose and their intermediate copies disappear;
learned arithmetic, decoder feedback, transfers and output shapes are unchanged.
Diagnostic captures retain their original NCHW boundary contract. The internal
optional-layout API documents the alternate shape; no public C API struct or
ABI changes were introduced.

Exact-equivalence regressions cover camera output indexing, both prompt capture
modes (including preserved diagnostic tensors), sparse prompt outputs, and three
conditioning-plus-first-decoder cases with an independent scalar layout inverse.
All 46 CPU ASan/UBSan/LSan tests and all 46 rebuilt UBSan tests pass. All 25
alternating full BF16 outputs are byte-identical before/after and across release
and UBSan builds. The public F32 path independently passes all 19 exact output
comparisons. The complete F32 operation capture still passes **646 checks** and
all 529 tensors retain the earlier exact SHA-256.

| Complete warm image-to-mesh | Median | Range | p95 |
| --- | ---: | ---: | ---: |
| Unsanitized | **116.74 ms** | 116.14–119.71 ms | 119.31 ms |
| UBSan | **149.32 ms** | 148.91–151.70 ms | 151.51 ms |

Both use five warmups and twenty timed alternating original dancer/rider images,
resident weights and all six feedback layers/final host outputs. Unsanitized
warm GPU telemetry averages 69.5%, peak 74%, over 24 samples; transfer volumes
and 84 graph calls are unchanged. This is a **4.80 ms** improvement over the
preceding 121.54 ms candidate, not an 80–85 ms result. GPU-compute call time is
essentially unchanged, about 83.6 ms. A current call-site/source mapping confirms
61.2 ms in `dino_resident_stack::execute`, 9.79 ms across the six
`body_decoder_layer` calls, 6.23 ms across 24 MHR projections and 2.81 ms across
ten feedback `graph_run::run` calls. Further encoder-kernel and pipeline
residency/dispatch work is needed; removing host copies alone cannot reach the
target.

The same live demo instance was updated after verifying sanitizer/release
identity. Real Chrome QA passes again, including new upload, cancellation and
recovery, repeated inference, original overlay, exact GLB export/reload and
history/mobile scrolling. Its body result is the unchanged accepted BF16 hash;
the latest warm browser job takes **0.303 s including export**, cold **4.131 s**.
These browser jobs are not the native 20-run median.

Evidence: `generated/diagnostics/bf16-condition-layout-{performance,ubsan}-v1/`,
`bf16-condition-layout-f32-body-acceptance-v1.json`,
`generated/fixtures/bf16-condition-layout-full-f32-v1/parity-v1.json`,
`bf16-condition-layout-demo-qa-v1/`, and corresponding build/test budget reports.

Fresh Clang ASan/UBSan/LSan C API fuzz runs pass **100,000 cases each** for
request/options (seed 934) and result/getters (seed 935), with GGUF loading
explicitly excluded. The targets are rebuilt against the current source; no
sanitizer exception is applied to these CPU checks. Budget artifacts are
`bf16-condition-layout-{api,result}-fuzz-100k-v1-budget.json`.

The full CPU ASan/UBSan/LSan rider run after the layout change is byte-identical
to the earlier accepted CPU output, SHA-256
`a1175d3e19647c1d2059808f7e3fa13a642eecf6f39aca976d18329747f89681`.
It completes in 84.23 s with peak cgroup RAM 5.369 GB. The 5 GiB high watermark
records 423 throttling events, but the 6 GiB hard limit is not reached and no
OOM, kill or sanitizer error occurs. Evidence:
`bf16-condition-layout-rider-cpu-body-v1-budget.json` and its complete result.

### Current kernel-level performance targets

`bf16-condition-layout-kernels-v1/` records actual Vulkan operation timestamps
over the same five warmups and twenty timed alternating images. All 25 outputs
are unchanged. Logging perturbs latency: **126.25 ms** median versus **116.74 ms**
without it, so it is diagnostic, not the headline benchmark. Summed reported GPU
operation time is **84.58 ms/image** across 1,680 warm graph submissions.

| Operation group | Reported GPU time / image |
| --- | ---: |
| Two BF16 MLP input projections, 64 calls | 14.33 ms |
| BF16 MLP output projection, 32 calls | 8.96 ms |
| ADD, 592 calls | 7.16 ms |
| BF16 QKV projection, 32 calls | 5.13 ms |
| MUL, 372 calls | 5.00 ms |
| Residual-corrected image flash attention, 32 calls | 4.79 ms |
| F32 MHR 3000→55317 projection, six calls | 4.78 ms |
| Precise-prefix F32 P×V, 32 calls | 4.77 ms |
| Exact BF16-round copies, 547 calls | 4.35 ms |

The next useful target is reducing encoder pointwise/rounding dispatch and
intermediate traffic while preserving exact rounding boundaries; ADD/MUL/round
alone account for 16.5 ms here. The 32 small precise-prefix P×V products are
another measured target. Do not drop that precision path to recover speed—the
raw register-token gates failed without it. Broader decoder/geometry residency
can reduce host/transfer overhead, but arithmetic changes still require the
frozen layer/final-body checks. Neither 80–85 ms nor sustained 90% utilization
has been achieved.

### Exact binary-operation BF16-round fusion

The next measured optimization is patch `0005`, selected with
`GGML_VK_FUSE_BF16_BINARY=1` in addition to the accepted flags above. F32 ADD
or MUL plus immediate BF16 narrowing/widening executes in one shader, retaining
the exact F32 operation and RNE rounding boundary. Shared/observed values,
aliases, strides and intervening views retain the original path. The applied
GGML source fingerprint is
`988dd8f4399b9093b4c084cec27ba288f25db94cb66c462cfdda579e51895447`;
the upstream submodule is still pristine.

All 250 model-free NVIDIA precision cases pass under UBSan. The 192 new binary
cases each execute with fusion off/on and compare exact scalar/device bits.
The separate log checker verifies 384 executions, including 24 actual eligible
fusions and 360 disabled/refused cases. Negative parser tests reject missing
logging, ignored flags, wrong operations, guard decisions and incomplete runs.
The initial test failed with fusion **disabled** because its SCALE oracle
omitted GGML's zero bias (`fma(-0, 2, +0)` is `+0`); the corrected test passes.
Earlier build failures were an incorrectly anchored MUL patch hunk, not OOM.
Those failed logs/reports are retained rather than counted as successes.

Both complete encoder captures are byte-identical to the preceding accepted
candidate, preserving all 36 frozen stage assertions per image. All 25 complete
outputs in the UBSan run, both release repetitions and kernel/host profiles
match the accepted per-image hashes. All 529 strict F32 captured tensors retain
SHA-256 `a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`,
and the fresh original comparison passes all 646 assertions. The isolated
UBSan build passes all 46 normal tests; 112 Python and Go race tests pass.
No frozen tolerance or model arithmetic contract changed.

Each timing run uses five warmups and twenty alternating dancer/rider requests,
the full pose branch with all six geometry-feedback stages and host results:

| Configuration | Median | Range | p95 |
| --- | ---: | ---: | ---: |
| Release | **114.62 ms** | 113.80–116.51 ms | 116.04 ms |
| Release repeat | **115.05 ms** | 114.30–116.85 ms | 116.64 ms |
| UBSan | **147.45 ms** | 146.91–148.78 ms | 148.75 ms |

The preceding release/UBSan medians were 116.74/149.32 ms. Release GPU usage
averages 69.1% / 68.6% in the two runs, peaks 73% / 75% (23 samples each).
This does **not** reach 80–85 ms or 90% utilization.

Timestamp profiling reports **82.09 ms/image** summed GPU operation time,
down from 84.58 ms; logging raises measured wall latency to 122.94 ms. It
shows 354 ADD-round and 128 MUL-round fused operations per image. MLP input
and output GEMMs still take 14.33 and 8.91 ms, fused ADD-round 6.52 ms,
precise-prefix F32 P×V 4.77 ms and six MHR projections 4.78 ms. There remain
84 graph calls per inference; this patch changes dispatches, not public output
scope or host/device transfer volume.

A fresh 999 Hz CPU profile restricted to warm requests identifies GPU-fence
polling as 35.7% of sampled CPU time (not 35.7% of wall time); memory movement
7.2%, image preparation 4.7%, CPU skinning 4.4%, and two finite-check routines
5.3% combined. BF16 host conversion also remains visible. Follow-up profiling
should distinguish removable host roundtrips/graph construction from waiting
on real GPU work, and investigate pointwise addressing and the small precise-
prefix P×V kernels without discarding their required precision.

The existing demo instance now runs this UBSan candidate. Real Chrome 151
upload, cancellation/recovery, warm reuse, rendering, original overlay, history,
mobile scrolling and exact GLB export/reload QA all pass. The screenshot was
inspected: no new gross arm/body or geometry defect. Its native result and GLB
retain the preceding accepted hashes. Cold/warm job times are 4.435/0.193 s
including export; these are individual browser jobs, not native medians.

Local ignored evidence: `generated/diagnostics/bf16-pointwise-*`, especially
`precision-v2.log`, `fusion-guards-v1.json`, `ubsan-v1/`,
`performance-v{1,2}/`, `kernels-v1/`, `host-profile-v1/`, and `demo-qa-v1/`;
the full F32 capture/report is `generated/fixtures/bf16-pointwise-full-f32-v1/`.
Memory-bound job reports record clean completion without OOM. Existing CPU
ASan/UBSan/LSan full-model and C API fuzz evidence is unchanged; this new
device patch is exercised under the documented NVIDIA-ICD UBSan exception.

### Resident patch embedding and prefix assembly

`dino_resident_stack::run_image` now retains the patch weights and constant
prefix alongside the transformer stack. Its image graph performs input BF16
rounding, patch convolution, the separate convolution/bias rounding boundaries,
prefix concatenation, all 32 blocks and final normalization/layout on device.
It explicitly copies the assembled tokens into the existing stack input before
the block nodes execute; backend overlap tracking supplies the dependency.
The original token-input route remains available for operation tests. Stem
tensors have a separate persistent allocation, outside stack lifetime reuse;
the combined activation budget remains checked at 1 GiB.

This removes per-request stem weight conversion/upload, patch-token download,
CPU prefix assembly/rounding and the stack-token reupload. Diagnostic observers
still expose the same patch, prefix, block and normalization boundaries. The
new tests alternate image/token graphs and repeated inputs, vary batch 1/2 and
storage-token counts 0/1/7, and verify malformed/nonfinite input rejection and
recovery. Both final CPU ASan/UBSan/LSan and Vulkan UBSan builds pass all 46
normal tests. Go race tests and 112 Python tests pass.

Both 36-stage dancer/rider captures remain byte-identical to the accepted
candidate. All 25 complete outputs in the UBSan run, two release runs and
kernel profile remain exact. All 529 strict F32 tensors retain SHA-256
`a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`;
the fresh original comparison passes all 646 assertions. There is no new
precision policy or tolerance adjustment.

Five warmups and twenty timed alternating dancer/rider requests per run:

| Configuration | Median | Range | p95 |
| --- | ---: | ---: | ---: |
| Release | **111.42 ms** | 111.06–113.96 ms | 113.31 ms |
| Release repeat after cleanup | **111.86 ms** | 110.81–113.48 ms | 113.29 ms |
| UBSan | **136.24 ms** | 135.49–138.33 ms | 137.33 ms |

Compared with the preceding 114.62/115.05 ms release and 147.45 ms UBSan
measurements, this is primarily a host/transfer improvement. Uploads fall from
40,432,816 to **33,193,136 bytes**, downloads from 37,646,476 to
**32,403,596 bytes**, and graph calls from 84 to **83** per image. Release GPU
utilization averages 69.5% / 71.3%, peaks 74% / 77% (22/23 samples).

The final timestamp profile reports **82.74 ms/image** summed GPU time and
120.43 ms wall median with logging. The GPU sum is not improved versus the
preceding 82.09 ms sample; do not attribute this host optimization to faster
GPU kernels. Main remaining groups are BF16 MLP input/output projections
(14.33/8.93 ms), fused ADD-round (6.51 ms), strict prefix P×V (5.29 ms),
QKV projection (5.14 ms), six MHR projections (4.79 ms) and image attention
(4.74 ms). The **80–85 ms goal remains open**.

#### Prefix-layout experiments, not production

`reference/experiments/rejected-prefix-layouts.patch` preserves three separately
tested variants; it is not applied by the build. Ninety-six model-free F32
P×V cases exercise their padding/layouts against an independent double oracle,
including 1,029 keys, twenty heads and five queries. Maximum absolute error
is `8.04384e-7` against a preselected `1e-5` limit. The existing 250 device
precision cases also pass. These isolated results do not establish full-model
acceptance:

- Padding K to a multiple of four **after softmax** keeps all 25 outputs exact,
  but release latency is 111.67 ms and UBSan 136.06 ms: no useful gain.
- Moving queries into a broadcast batch axis selects GGML's single-column
  vector kernel. The focused real-shape product falls from roughly 151 to
  23 microseconds and UBSan full inference reaches 131.46 ms. Final rider body
  checks pass, but four complete encoder stages (22–25) fail their unchanged
  maximum-absolute limits: error 256 at flat index 3443 versus limits
  76/72/104/180. Relative-L2 checks pass, but that does not override the failed
  absolute gates. This candidate is **rejected**, not promoted based on final
  mesh or synthetic tests alone.
- Transposing the GEMM operands/output preserves all 25 results but gives
  only 135.75 ms UBSan versus 136.24 ms baseline, within overlapping run ranges.
  It is not retained as a proven optimization.

The upstream logger calls the five-query batched product `MUL_MAT_VEC`, but
dispatch inspection shows it uses the **matrix** path: batched vector dispatch
requires a single column (or an unbatched multi-column case). This explains
why padding alone did not activate the expected vector-load optimization.
The failed first rider-comparison invocation used a nonexistent reference path;
the corrected invocation verified the original hash and recorded the real
four-stage failure. All failed and successful reports are retained.

Evidence is under `generated/diagnostics/bf16-resident-stem-*` and
`bf16-prefix-{pad,mv,transpose}-*`; the strict F32 capture is
`generated/fixtures/bf16-resident-stem-full-f32-v1/`. No GGML patch or submodule
revision changed in this optimization. Existing C API fuzz evidence remains
applicable to its unchanged API implementation; fresh CPU sanitizer regression
tests cover the new resident graph and its lifetimes.

The existing demo instance was replaced in place with the cleaned UBSan build.
Actual Chrome 151 upload, cancellation/recovery, warm reuse, rendering, original
overlay, GLB export/reload and history/mobile QA pass in
`bf16-resident-stem-demo-qa-v1/`. The front overlay was inspected and the body
hash remains `d9cba0af27668be06acfb2491fae838cef7a3128f131f2f94ec466b2492d2fb6`.
All exported vertices/indices are exact. Cold/warm browser jobs took
4.148/0.369 s including export (individual UI jobs, not native benchmark medians).

### Linear binary-round shader indexing

The next accepted optimization is build-copy GGML patch `0006`, enabled with
`GGML_VK_BF16_BINARY_LINEAR=1` in addition to the existing binary/round fusion
flags. It changes **addresses only**, not F32 arithmetic, BF16 round-to-nearest-even,
tensor precision, model dimensions or outputs. Direct linear indexing replaces
general four-dimensional decomposition for dense same-shape, repeated-row and
scalar operands. Other broadcasts retain the general shader. All original
fresh/single-use/unobserved/alias/overlap guards remain, with additional explicit
dense-layout, F32-type and destination-shape checks.

The new model-free test runs 64 cases with both indexing settings, comparing
exact bits against an independent F32-operation/BF16-round scalar oracle and
the existing device shader. Cases include four-dimensional batches, 12-byte
descriptor offsets, repeated rows/scalars, general-broadcast fallback, tails
and more than 262144 elements. Input buffers and their guard regions must remain
unchanged. All 250 earlier device precision cases and 96 prefix-layout cases
also pass. The existing fusion log gate still verifies 384 executions: 24
eligible and 360 disabled/refused paths.

Both full encoder captures remain byte-identical to the accepted resident-stem
captures (36 stages per image), as do all 529 strict F32 capture tensors. A fresh
comparison passes all **646 F32 assertions** without changed limits. Every one
of the 25 full outputs matches its baseline in each of five runs: UBSan,
release, same-binary disabled control, release repeat, and kernel logging.
Thus the previously accepted frozen upstream-only stage/final policies and
reported hand exceptions remain unchanged; this is not synthetic-only parity.

All timings retain five warmups plus twenty timed alternating dancer/rider
images, complete image-to-mesh scope, six geometry-feedback layers and final
host outputs. Model loading, detector, detailed hand refinement, HTTP and exports
are excluded from native timings as before.

| Configuration | Median | Range | p95 |
| --- | ---: | ---: | ---: |
| Release, linear indexing | **109.58 ms** | 108.96–111.49 ms | 111.27 ms |
| Same release binary, indexing disabled | 111.34 ms | 110.90–114.71 ms | 113.26 ms |
| Release, linear indexing repeated after control | **109.48 ms** | 109.00–110.90 ms | 110.84 ms |
| UBSan, linear indexing | **133.89 ms** | 132.81–136.94 ms | 135.67 ms |

The separate timestamp-logging run has 119.36 ms median wall time, including
instrumentation, and **80.52 ms summed GPU timestamps/image**, down from
82.74 ms. Fused ADD-round falls from 6.512 to **4.605 ms**; MUL-round from
2.548 to **2.247 ms**. Graph count and transfers are unchanged: 83 computes,
33,193,136 uploaded bytes and 32,403,596 downloaded bytes/image. Uninstrumented
release GPU samples average 70.1%/66.8% (74%/73% peaks), not sustained 90%.
The **80–85 ms complete-inference goal remains open**. GPU time alone is not
end-to-end latency; host preprocessing, transfers and graph/dispatch work
remain targets alongside further kernel optimizations.

Both normal native suites pass 46/46 tests (CPU ASan/UBSan/LSan and Vulkan
UBSan), and Go race tests pass. Public C ABI and parsers are unchanged; this
phase does not claim a fresh C API fuzz run in place of the prior 100k-case
sanitizer coverage. Heavy jobs remained serialized and memory-bounded, with no
OOM events. Cached compilers were used, with no Nix derivation builds. The
public GGML submodule remains pristine; the six-patch fingerprint is
`d8aadcc89f7b99e51307ef77cf902374eae323922afb497a5819e678719058b6`.

The existing demo was replaced in place with the UBSan candidate, retaining
history. Actual Chrome 151 QA passes upload, box selection, invalid/oversized
rejection, cancellation/recovery, warm reuse, original overlay, orbit, history,
desktop/mobile scrolling and GLB export/reload. The front overlay was inspected;
body SHA remains `d9cba0af27668be06acfb2491fae838cef7a3128f131f2f94ec466b2492d2fb6`,
and all exported vertices/indices remain exact. Cold/warm browser jobs took
**4.184/0.185 s** including export (individual UI jobs, not benchmark medians).
The demo owns the new flag: Vulkan BF16 enables it, while CPU/F32 strip inherited
experimental flags.

Local ignored evidence: `generated/diagnostics/bf16-linear-binary-*`,
`generated/fixtures/bf16-linear-binary-full-f32-v1/`, and
`generated/diagnostics/bf16-linear-binary-demo-qa-v1/`. Each benchmark records
model input/output identities, runner/library/backend hashes and the indexing
flag. An initial patch-context build failure is retained separately; it was
corrected before compilation and is not successful build evidence.

### Host finite-value validation: baseline SIMD, no removed checks

Fresh CPU profiling of the linear-indexing baseline identified scalar finite
scans as a measurable host cost. Two out-of-line scan symbols accounted for
4.48% and 2.28% of warm CPU-clock samples, with further inlined scanning in the
decoder. These are **CPU sample fractions**, not percentages of end-to-end
wall time; 36.38% of sampled CPU activity was fence polling.

`src/finite.hpp` now provides exact F32 classification using baseline SSE2
integer masks on x86, retaining the original scalar `std::all_of` implementation
on other targets. Four values are loaded only when four remain in the span;
remaining elements use scalar bit classification. Exponent 255 rejects both
infinities and every quiet/signaling NaN, independent of sign or payload.
No input is changed, no arithmetic is approximated, no validation is omitted,
and no native-only ISA or fast-math flags are needed. Existing size checks,
error messages and C ABI exception boundaries remain. The helper is used by
Body decoder/conditioning/feedback/flow/pose/output and MHR geometry scans.
GGUF loading and immutable checkpoint validation are not changed.

The new normal test covers lengths 1–129 at offsets 0/1/3, every invalid
position and ten nonfinite patterns, empty spans, invalid guards outside the
span, subnormals/signed zero/maximum finite values, large tails and over one
million deterministic random bit patterns against `std::isfinite`. CPU
ASan/UBSan/LSan checks exact allocation boundaries. Disassembly confirms
`pcmpeqd`/`pmovmskb` SIMD operations in the final performance object.

Two experiments were **not** retained: a scalar integer reduction that GCC
did not auto-vectorize (115.24 ms), and an exact per-request byte-normalization
lookup that did not improve on SIMD scans alone (105.68 ms). Original affine
sampling and per-pixel normalization remain unchanged; a new C API regression
checks exact normalization bits for all 256 byte values in all three channels.
See [rejected experiments](experiments/rejected-host-validation.md).

The cleaned candidate retains five warmups and twenty timed alternating
dancer/rider requests, all six geometry-feedback layers and final host outputs:

| Configuration | Median | Range | p95 |
| --- | ---: | ---: | ---: |
| Unsanitized, final clean run | **104.98 ms** | 104.57–106.72 ms | 106.66 ms |
| Unsanitized, final clean repeat | **105.02 ms** | 104.76–106.70 ms | 106.68 ms |
| UBSan, final clean run | **125.80 ms** | 125.13–128.04 ms | 127.53 ms |

This improves on 109.5–109.6 ms unsanitized / 133.9 ms UBSan. The separate
kernel-logging run measures 113.59 ms median wall time and **80.46 ms summed
GPU timestamps/image**, essentially unchanged from 80.52 ms. Transfers and
work are unchanged: 202 uploads/33,193,136 bytes, 233 downloads/32,403,596 bytes,
83 compute calls. Warm GPU telemetry averages 72.2%/72.1%, peaks 74%/74%, with
21 samples per clean unsanitized run. The **80–85 ms end-to-end goal is open**.

Build-label clarification: the unsanitized benchmark directory uses
`RelWithDebInfo`, **`-O2 -g`**, with sanitizers disabled—not CMake's stock
`Release` (`-O3 -DNDEBUG`). Earlier tables used “release” to distinguish it from
the sanitizer build. The comparisons above retain the same optimization flags
to isolate this change. Testing stronger ordinary host optimization remains a
potential next step; it is not yet measured or accepted here.

Fresh warm CPU samples show fence polling 39.75%, memory copies 8.10%, CPU
skinning 6.39%, image preparation 5.34%, and the two remaining scan symbols
1.24%/0.86%. Copy stacks identify Vulkan upload/readback and the decoder image
snapshot as contributors. These are sampling observations, not disjoint stage
latencies. The fresh reports preserve matching symbols before any later binary
rebuild. Further host/transfer reductions and GPU kernel work remain necessary.

Every output is byte-identical in both final clean runs, UBSan, CPU-profile
and kernel-profile runs (25 each). Both 36-stage encoder captures retain the
accepted hashes; all 529 strict F32 tensors retain their hash, and a fresh
reference comparison passes **646/646 assertions**. Frozen BF16 policies and
reported nonblocking hand exceptions remain unchanged. Both normal suites
pass **47/47** tests (CPU ASan/UBSan/LSan and Vulkan UBSan).

Fresh CPU ASan/UBSan/LSan fuzz runs pass 100,000 cases each for model-request
API, result ownership/parser API, and crop/image API, seeds 936/937/938.
Peak cgroup RAM is 328,536,064 / 129,839,104 / 445,358,080 bytes. These exclude
GGUF loading and are not claims of fuzzing loaded neural weights. All heavy
jobs remain serialized and memory-bounded; no OOM events or Nix derivation
builds occurred. No new GGML patch or submodule change was needed.

The existing demo was replaced in place with the final UBSan build. Actual
Chrome 151 upload/render/original-overlay/history/cancellation/mobile/export QA
passes, and the front overlay was inspected. The accepted body and exported
GLB hashes remain unchanged, with every exported vertex/index exact. Cold/warm
browser jobs took **4.312/0.177 s** including export (single UI requests, not
native benchmark medians). Evidence: `generated/diagnostics/bf16-host-demo-qa-v2/`.

Local ignored numerical/profiling evidence: `generated/diagnostics/bf16-host-*`,
`bf16-finite-*`, `bf16-linear-host-profile-v1/`, and
`generated/fixtures/bf16-host-full-f32-v2/`. Final clean timings are
`bf16-host-performance-v2/` and `v3/`; `bf16-host-performance-v1/` is the
unaccepted normalization-lookup experiment.

## Smaller BF16 matrix tiles (2026-09-09)

The BF16-only CM2 tile selector in patch `0007` selects GGML's existing
small kernels, including the model's F32-carrier/BF16-product path. It does
not alter F32 decoder arithmetic, remove work, or change rounding boundaries.
The actual BF16 pipeline is checked, rather than relying on the source RHS
tensor type. Unset or unrecognized choices retain the original heuristic.

With `GGML_VK_BF16_MATMUL_TILE=small`, repeated unsanitized full image-to-mesh
runs measure **99.460 / 99.128 ms** median. Each run has five warmups and
twenty timed alternating-image requests. Ranges are 98.985–102.158 and
98.876–100.968 ms; p95 is 102.087 / 100.844 ms. The original heuristic in
the **same binary** measures **104.870 ms**. Compiler flags are `-O3 -g`,
assertions retained, no fast-math. A preceding six-patch O3 control measured
104.744 ms, only marginally different from the O2 baseline; the tile change,
not compiler optimization, accounts for the demonstrated gain.

The separate UBSan-O2 full run measures **120.119 ms**, down from about
125.8 ms. Every one of the 25 outputs in each timing run is byte-identical
to the accepted baseline. Both 36-stage encoder captures and all 529 strict
F32 tensors are also byte-identical. Frozen numerical policies are unchanged.

Fresh instrumented GPU timestamps sum to **74.923 ms/image**, down from
80.462 ms. The two BF16 MLP shapes total 18.830 ms (12.333 + 6.497), down
from about 22.9 ms; QKV is 4.935 ms and output projection 1.930 ms.
Transfers remain 202 uploads/33,193,136 bytes and 233 downloads/32,403,596
bytes, with 83 compute calls. Instrumented full latency is 107.866 ms:
timestamp logging is not used for the clean benchmark above. Clean sampled
GPU utilization averages 67.1% / 69.8%, peaks 73% / 72%; neither the 90%
utilization aspiration nor the **80–85 ms full-inference goal** is met.

The new model-free matrix oracle covers 64 exact dyadic cases with both
BF16 and F32 carriers, trained shapes, tails, batching and repeated resident
execution. Existing device arithmetic and fusion-guard tests also pass.
An initially ineffective selector and a tiny unsupported BF16-vector test
are documented in [the experiment record](experiments/bf16-matrix-tiles.md),
not counted as successful full-model optimization evidence.

Local ignored evidence: `generated/diagnostics/bf16-tiles-*` and
`generated/fixtures/bf16-tiles-full-f32-v1/`. The GGML submodule remains
pristine; this is a build-copy patch, not a private upstream commit.

Fresh policy comparisons pass both 36-stage BF16 encoders, both final-body
gates (the same dancer hand-logit exception remains nonblocking), and all
646 strict F32 assertions. CPU ASan/UBSan/LSan, Vulkan UBSan and unsanitized
builds pass 47/47 normal tests; 112 Python tests and Go race tests pass.
The existing demo was replaced in place with the UBSan small-tile build.
Actual Chrome 151 upload/overlay/cancellation/history/mobile/export QA passes;
the inspected overlay and all exported vertices/indices remain unchanged.
Cold/warm UI requests took 4.188 / 0.283 s including export, not benchmark
medians. The first QA attempt's safe RAM-budget refusal and the explicit
smaller worker cap are recorded in [visual evidence](BODY_BF16_VISUAL.md).

## Bounded pinned-transfer batching (2026-09-10)

The runtime now offers `SAM3D_BATCHED_TRANSFERS=1`: complete validated tensor
uploads and downloads use disjoint regions of a reusable, at-most-32-MiB pinned
host buffer, with one final synchronization. Unsupported backends/devices and
oversized valid requests retain synchronous IO. Device capabilities are cached
once; querying them on every graph cost 1.83 ms in the initial prototype.
The public calls remain synchronous and keep their existing ownership contract.
No new GGML patch, arithmetic change, skipped layer or reduced output is involved.

Across five warmups and twenty timed alternating images, unsanitized medians
are **95.417 / 95.401 / 95.021 ms**. The final run's p95 is **95.902 ms**, range
**94.745–95.935 ms**. The same-binary batching-off control is **100.406 ms**.
The final UBSan-O2 median is **115.942 ms**. Warm GPU utilization samples average
76.6%, peak 78%; kernel timestamps total **75.943 ms**. The **80–85 ms target
remains open**. These retain the complete body-only scope, not model loading,
detector/refinement or export/HTTP time.

Batching covers 82 of 83 graphs. Per image there are 199 async uploads and 232
async downloads, plus three synchronous uploads and one download; bytes are
unchanged at **33,193,136 up / 32,403,596 down**. All 25 outputs in every accepted
timing/profile run match the previous baseline exactly. All 25 F32 public
requests also match with batching on/off, exercising the actual async path.
All **529** F32 operation tensors retain their previous hash; all **646** strict
original assertions pass. Both final BF16 policies pass with the same isolated
non-blocking dancer hand-logit result. No tolerance changed.

CPU ASan/UBSan/LSan and Vulkan UBSan each pass **48/48** normal tests; the actual
NVIDIA transfer test passes **84** alternating cases/84 pinned batches, including
guards, malformed requests, recovery, tensor views and a 36 MiB fallback. The
112 Python tests and Go race tests pass. The first timestamp-profile run hit a
GGML assertion (SIGABRT, not OOM): its logger requires an empty compute context
at graph entry. An extra synchronization now applies **only when that logger
is enabled**. The device test and 25-request kernel profile then pass exactly.
This instrumentation overhead is excluded from production timing.

Warm CPU sampling (95.543 ms instrumented) attributes 42.68% to fence polling,
8.15% to skinning, 6.69% to copying and 4.75% to image preparation. These are
sample fractions, not additive wall times. Remaining GPU work and CPU geometry
are useful next profiling targets. See [the detailed record](experiments/batched-transfers.md).

The existing demo is updated with the UBSan build and selects batching only
for Vulkan BF16. Actual Chrome 151 QA passes upload, cancellation/recovery,
warm reuse, original overlay, history/mobile scrolling and exact GLB export.
The warm browser job took **0.174 s including export** (one QA request, not a
benchmark median). Its overlay was visually inspected and output/GLB hashes
are unchanged. Local evidence: `generated/diagnostics/bf16-transfers-*` and
`generated/fixtures/bf16-transfers-full-f32-v1/`. Heavy jobs were serialized and
memory bounded with the unchanged 10 GiB reserve; no Nix derivation rebuild.

## Exact SIMD CPU skinning (2026-09-10)

Four influences can now be transformed together with baseline SSE2 on x86,
while maintaining the scalar multiply/add order, both quaternion normalizations
and original serial scatter-add order. `SAM3D_SIMD_SKINNING=1` enables the path;
other architectures, tails and diagnostic operation capture remain scalar.
No validation, influence, layer or output is removed. The demo selects it for
Vulkan BF16. No new GGML patch or C ABI change is needed.

Warm full-inference medians are **94.165 / 93.803 ms**, versus **95.573 / 95.294 ms**
with the option disabled in the same binary. Five warmups and twenty timed
alternating images are retained. Repeat p95 is **94.597 ms**, range
**93.539–95.853 ms**. UBSan-O2 is **114.150 ms**. CPU sampling shows skinning's
sample share falling from 8.15% to **3.52%**, with fence polling 44.51% and image
preparation 4.93%; these are sample fractions, not additive wall times.
GPU arithmetic/transfers remain unchanged. The **80–85 ms goal stays open**.

All 25 BF16 outputs in each timing/profile run remain byte-identical; all 25
normal F32 public outputs match the previous implementation. The latter is
necessary because operation capture intentionally uses the scalar path.
All 529 F32 diagnostic tensors retain their hash and all 646 strict assertions
pass. Both final BF16 policies pass without new exceptions or changed tolerances.
CPU ASan/UBSan/LSan and Vulkan UBSan pass 48/48 normal tests, including 96 new
bit-exact scalar/SIMD/captured skinning cases; 112 Python and Go race tests pass.

The updated UBSan demo passes actual Chrome 151 upload, cancellation/recovery,
warm reuse, original overlay, history/mobile and exact GLB export/reload QA.
The overlay was inspected and all vertices/indices and hashes remain unchanged.
Cold/warm browser requests were **4.349 / 0.169 s including export**, not benchmark
medians. See [the detailed experiment](experiments/simd-skinning.md). Local evidence
uses `generated/diagnostics/bf16-skin-simd-*` and
`generated/fixtures/bf16-skin-simd-full-f32-v1/`.

## Narrow F32 prefix GEMM (2026-09-10)

Warm full inference now measures **90.806 / 91.071 ms** unsanitized, versus
**94.491 ms** with the change disabled in the same binary. Five warmups and
twenty timed alternating images remain required. First-run p95 is 91.137 ms,
range 90.546–91.659 ms; repeat p95 is 91.753 ms. UBSan-O2 is **111.243 ms**.
The complete body-only scope, all six feedback layers and final host outputs
are unchanged; the **80–85 ms target is still open**.

The first experiment grouped F32 matvec rows. It passed independent device
tests but changed some rounding and gave no meaningful full-model gain:
114.263 / 114.125 / 114.039 / 115.282 ms UBSan for 1/2/4/8 rows. It is archived
in `reference/experiments/rejected-f32-matvec-rows.patch`, not applied. Source
inspection revealed that the five-query/twenty-head prefix uses **GEMM**, not
the matvec path implied by the profiler's generic label.

Applied patch `0008` instead selects a 16×8 F32 GEMM tile for small-N batched
products, retaining the original hard-coded K=32 tile and ordered F32 dot-product helper.
It explicitly disables split-K for that tile and is guarded to NVIDIA CM2,
32-lane subgroups and F32×F32. No new shader or lower-precision conversion is
introduced. `GGML_VK_F32_NARROW_MATMUL=1` opts in; the demo selects it only for
Vulkan BF16. The independent 56-case device test passes bit-exactly off/on,
including tails, signed non-dyadic data and the long-K/N=8 split guard.

All 25 results in every accepted full timing/profile run remain byte-identical
to the prior baseline; all 25 normal F32 public results are also exact. Both
36-stage encoder captures retain their hashes and pass frozen policies. All
529 F32 operation tensors retain their hash and all 646 strict assertions pass.
Both final BF16 policies pass with the same non-blocking dancer hand-logit
exception. Existing device precision/fusion/attention tests, 48 CPU
ASan/UBSan/LSan tests, 48 Vulkan-UBSan build tests, 112 Python tests and Go race
tests pass. No tolerance changed.

GPU kernel timestamps total **72.284 ms**. The targeted prefix product falls
from approximately 5.115 ms to **2.553 ms** across the 32 blocks. Graph and
transfer counts remain 83 computes, 33,193,136 bytes up / 32,403,596 down.
Warm GPU samples average 77.3%, peak 79%, not 90%. The existing demo is updated
with UBSan and passes actual Chrome 151 upload/overlay/cancellation/history/
mobile/exact-GLB QA. Its visually inspected overlay and native/export hashes
are unchanged; cold/warm UI requests took 4.211 / 0.253 s including export.

See [the detailed record](experiments/f32-narrow-matmul.md). Local evidence:
`generated/diagnostics/bf16-narrow-matmul-*` and
`generated/fixtures/bf16-narrow-matmul-full-f32-v1/`. Patch fingerprint is
`eba02773aeaa5ed1d92077066ea98b0cd9dba27e400d8a7cebf8fb41ee0d96fd`;
upstream GGML remains pristine. Bounded serialized jobs recorded no OOM events;
the source/compiler builds did not invoke Nix derivation rebuilding.

## Smaller narrow F32 tile (2026-09-10)

The accepted follow-up selects an 8×8 output tile, keeping the original F32
K=32 reduction and ordered dot products. The previous K=16 description was
incorrect: F32 ignored that specialization slot and hard-coded 32. Patch `0009`
also permits controlled K=64 experiments, but they did not improve on K=32 and
are not selected. The demo explicitly sets `GGML_VK_F32_NARROW_TILE=tiny32`
only for Vulkan BF16, stripping conflicting inherited values.

Release medians are **90.798 / 90.574 ms**, compared with **91.531 ms** for the
same binary's 16×8 control; UBSan measures **110.812 ms**. Every run retains five
warmups and twenty timed alternating-image complete requests. No outputs,
feedback layers or geometry are omitted. Both encoder hashes, all 529 F32
tensor captures, all 25 public F32 results and all full BF16 outputs are exact.
Frozen encoder/final-body policies and 646 strict F32 assertions pass, with only
the unchanged non-blocking dancer hand-logit difference. Both 48-test sanitizer
suites, real GPU precision/fusion/attention tests, 112 Python tests and Go race
tests pass. All four tile controls additionally pass the 56-case device oracle.

GPU timestamp total is **72.232 ms**, prefix P×V **1.826 ms**, and prefix Q×K
**0.477 ms** across 32 blocks. Graph and transfer structure are unchanged.
The repeat's 19 warm GPU samples average 72.5%, peak 79%; utilization is still
below 90%. The **80–85 ms goal remains open**. See
[the detailed tile follow-up](experiments/f32-narrow-depth.md), including
candidate p95/ranges, rejected depth variants and the next profiling hypothesis.

The updated UBSan demo passes actual Chrome 151 QA with the unchanged
upstream/native overlay, exact GLB export/reload and no browser errors.
Individual cold/warm UI jobs took **5.001 / 0.161 s including export**; these
are not benchmark medians. Local QA is in
`generated/diagnostics/bf16-narrow-depth-demo-qa-v1/`. The nine-patch fingerprint
is `468ca30f8c90bfe6b4c833295b15b1af8fa3e45b267fa9f05eb4257ad97b9fe9`;
upstream GGML is still pristine. No Nix derivations were rebuilt or memory
safety limits lowered.

## Rejected packed feed-forward experiment (2026-09-10)

Combining the two resident BF16 feed-forward projections into one wider
product preserved all 25 complete outputs on alternating dancer/rider inputs,
but regressed UBSan median from **110.649 to 114.746 ms**. All 25 profiled outputs
were exact as well. The matrix kernels saved only ~0.3 ms; strided consumers
lost the existing multiply-round fusion and increased downstream costs.
No release speedup is claimed and nothing was deployed. The prototype was
removed and archived after CPU sanitizer tests and trained Vulkan checks.
The restored source passes 48 CPU sanitizer tests, 112 Python tests and Go
race tests. See [the detailed evidence](experiments/packed-ffn.md).

The initial 6 GiB build was safely refused for headroom before execution;
the single-compiler retry used a 5 GiB cap, peaked at 4.17 GiB and recorded no
OOM events. The 10 GiB reserve was never reduced. The running validated demo
and unrelated services remained untouched. The next measured candidate is
activation/rounding fusion on the original contiguous tensors, not packing.

## Exact SiLU gate fusion accepted (2026-09-10)

The contiguous six-node gate now executes as one shader while retaining the
two BF16 RNE boundaries. Full-model traces confirm 32 actual fusions, not merely
proposed profiler labels. An explicit GGML widening-copy destination avoids
unsafe allocator overlap without bypassing real-source alias checks. See
[the investigation](experiments/bf16-silu-gate.md) for the failed proposals,
allocator evidence, guards and exact test coverage.

Release warm medians are **88.285 / 88.251 ms**, compared with **90.614 ms**
disabled in the same binary; UBSan is **108.509 ms**. These retain five warmups,
twenty timed alternating images and complete host outputs. All results remain
byte-identical, both 36-stage BF16 encoders and final policies pass, and all
646 strict F32 assertions/529 capture tensors and 25 public F32 results remain
unchanged. There are 408 GPU off/on cases, 48 CPU sanitizer tests, 48 normal
Vulkan-UBSan-build tests, 112 Python tests and passing Go race tests.

GPU timestamps total **69.138 ms**. Warm utilization averages 75–77%, peaking at
79%, not 90%. The updated UBSan demo passes actual headless Chrome upload,
render/overlay/history and exact GLB reload QA; geometry and hashes are
unchanged. The performance build uses `-O3 -g1` to reduce compiler memory,
without changing optimization, assertions or the separate sanitizer build.
The 10 GiB host reserve is unchanged. **The 80–85 ms goal remains open.**
