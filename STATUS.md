# Implementation status

Updated 2026-09-10. **Video:** offline file sampling, persistent sequences,
scrubbing and live webcam tracking now reuse the same native Body worker.
Live mode has a configurable Hz cap, one frame in flight and immediate result
presentation with a 25 ms blend (no delayed playback buffer). Remote cameras require HTTPS. This is
single-person frame-wise estimation, not a learned temporal model or automatic
identity tracker. See [video QA and limitations](reference/BODY_VIDEO.md).
The production live upload pipeline now overlaps worker encoding with inference,
retaining just one replaceable prepared frame. Direct camera-frame capture and
byte-exact typed image packing now reach **8.1 Hz**, **176 ms** median first render
submission and **202 ms** settled render in matched headless HTTPS tests.
These are not physical camera-to-display measurements. All 24-bit YCbCr colors,
alpha/stride/subimage packing checks, Go race tests and browser fallback,
cancellation, backpressure and exact settled-display checks pass.

The final scalar image-gather experiment measures 85.82–85.87 ms unsanitized,
with all full outputs exact. Its SSE2 variant adds no useful gain and is archived.
The optimization pass was stopped at the user's request; the 80–85 ms target is
not claimed achieved. The live demo now uses the production BF16 build with the
validated affine/norm fusions and scalar gather enabled: **85.79 ms** native
median, all 25 complete outputs byte-identical to the accepted baseline.
CPU/Vulkan correctness builds retain their sanitizers. Browser and server stage
timers now separate encoding, transport, preprocessing and native inference;
native latency is not live webcam throughput.
See [the final experiment](reference/experiments/image-gather.md) and
[live profiling](reference/BODY_LIVE_PERFORMANCE.md).

The scope remains **Body only**, following the user's
scope reduction. A real single-image Body web demo is now available; full trained
Body acceptance and optimized end-to-end performance parity remain incomplete. Objects work below
is retained as historical/deferred progress, not a prerequisite for this goal.

**BF16:** the current NVIDIA candidate passes all 36 complete encoder-stage
checks on each of two official images (dancer and rider), all 32 isolated trained
attention checks, and both final-body policies under frozen original-only limits.
Maximum vertex distances are 0.969 / 1.468 mm. One dancer hand-logit discrepancy
is reported but non-blocking. This uses BF16 image encoding, F32 decoder/MHR,
corrected fused attention for image queries and F32 math attention for the five
class/register queries. All 646 strict F32 checks remain green; all 529 captured
F32 tensors are unchanged. Real NVIDIA precision regression passes 250 cases,
including 192 binary BF16-round cases exercised with fusion both off and on.

The preceding SiLU-only demo-validated unsanitized latency was **88.25–88.29 ms** median across repeated runs (five
warmups, twenty timed alternating images each). A narrow scalar F32 GEMM tile
cuts precise-prefix work from ~5.1 to ~1.8 ms without changing output bits.
The latest 8×8 tile saves ~0.7 ms against the same-binary 16×8 control
(91.5 ms); increasing reduction depth did not improve it. See the
[tile follow-up](reference/experiments/f32-narrow-depth.md).
The preceding exact four-influence SSE2 skinning
saves another ~1.5 ms with unchanged accumulation order and bit-identical outputs;
96 added randomized/tail/duplicate-index cases pass. Bounded pinned-transfer batching
covers 82 of 83 graphs and caches device capabilities; the same-binary batching
off control is 100.4 ms. All outputs remain exact. The preceding selection of smaller
BF16 matrix tiles saves ~5.5 ms versus the original heuristic in the same
binary (104.87 ms). This follows the earlier exact SIMD finite scans,
resident patch/prefix assembly and linear-indexing shader improvements.
All 25 outputs remain byte-identical; all public F32 outputs
also match exactly. Upstream CUDA BF16 is 84.5 / 81.6 ms (eager / compiled
backbone); the **80–85 ms goal remains open**. The current UBSan candidate
measures **108.5 ms**, down from 110.8 ms. GPU kernel timestamps total
**69.1 ms**; significant host/transfer/dispatch work remains. A matrix-vector
row-group experiment gave no useful gain and is archived, not applied. The
profiler's matvec label had obscured the multi-head prefix's actual GEMM path.
Packing the two encoder feed-forward projections was also tested and rejected:
all trained outputs were exact, but UBSan full latency rose from 110.65 to
114.75 ms. It saved only ~0.3 ms of matrix work and increased downstream
layout/conversion costs. The prototype is archived, not active; see
[the experiment](reference/experiments/packed-ffn.md).
A now-deployed optimization fuses the SiLU gate while preserving both
BF16 rounding boundaries: repeated release timing is **88.25–88.29 ms** versus
90.61 ms disabled in the same binary. All full outputs, both encoder trajectories,
646 strict F32 checks, CPU sanitizers and 408 GPU off/on cases pass. Actual
kernel traces confirm 32 gate fusions per image; GPU time is 69.14 ms. The
updated UBSan demo passes actual headless Chrome upload/render/overlay/history/
export QA, including exact mesh reload. See [the fusion investigation](reference/experiments/bf16-silu-gate.md),
including allocator pitfalls and the fresh warm CPU profile. The target remains open.
Two exact SSE2 image-normalization variants were subsequently measured and
removed: neither established a useful gain over production. All results were
exact; the restored baseline remains about 88.4 ms. See the
[archived experiments](reference/experiments/rejected-image-simd.md).
A subsequent opt-in affine fusion measures **86.68–86.98 ms** release median,
versus 88.03 ms disabled in the same binary. All 1,248 exact device cases,
both encoder policies, both final-body gates, 646 strict F32 checks and 25
public-F32 outputs pass; profiling confirms 162 affine dispatches per image.
Both 48-test sanitizer-build suites, 112 Python tests and Go race tests also
pass. The deployed demo still uses the accepted SiLU build; affine demo
integration/QA is pending. See [the affine experiment](reference/experiments/bf16-affine.md).
The subsequent normalization/affine candidate reaches **86.39–86.55 ms**
versus 87.12 ms disabled in the same release binary. It preserves all full
outputs, both encoder/final-body policies, strict F32 and normal sanitizer
regressions. A first prototype was rejected for changing normalization
contraction; the corrected shader passes 720 exact device cases and executes
64 fusions/image. GPU time is 67.44 ms. See [the normalization experiment](reference/experiments/bf16-norm-affine.md).
The 80–85 ms target and deployment/Chrome QA of the new fusions remain pending.
Profiling mode needs
an extra upload synchronization because GGML's timestamp logger requires an
empty compute context; the normal path retains one final synchronization.
The failed diagnostic attempt and tested fix are recorded in the
[transfer experiment](reference/experiments/batched-transfers.md). The earlier
resident stem removed 7.24 MB of uploads, 5.24 MB of downloads and one graph
call per image. Both CPU ASan/UBSan/LSan
and Vulkan UBSan builds pass all 48 normal tests. The new transfer test covers
alternating inputs, tensor views, guards, rejections, recovery and oversized
synchronous fallback; actual NVIDIA execution passes with and without profiling.
Earlier request/result/crop C API
fuzz runs pass 100,000 cases each with ASan/UBSan/LSan, excluding GGUF loading.
The 250 device precision
cases plus 64 new linear-indexing cases and 96 alternative-layout F32 operation
cases pass, as do 64 new matrix tile/carrier cases; a faster prefix
vector-batching experiment nevertheless failed four complete-encoder gates
and is archived outside production. No tolerances were changed.
CPU ASan/UBSan/LSan also passes the held-out rider's 36 encoder stages and
final-body gate, with one non-blocking hand-box discrepancy. Yoga's original
controls exceed calibration ceilings and have no accepted policy.

Updated two-image visual comparisons and the actual BF16 demo workflow pass
real headless Chrome QA. The demo supports `--precision bf16`, retains the F32
default/option, and identifies the precision of each saved job. Upload,
cancellation/recovery, warm reuse, overlay, exact GLB export/reload and history/
mobile scrolling pass. The latest warm UBSan browser job is 0.253 seconds
including export (one QA request, not a benchmark median). No private GGML commits are used; changes are
reviewable build-copy patches.
See the [active goal](reference/BODY_BF16_GOAL.md),
[numerical evidence](reference/BODY_PRECISION.md) and
[visual and live-demo evidence](reference/BODY_BF16_VISUAL.md).

The [single-image demo](demo/README.md) now supports photo upload, explicit person
selection/camera settings, a mesh/skeleton viewer, original-upstream overlay,
GLB/OBJ downloads and persistent history. A real headless Chrome upload on
NVIDIA Vulkan passes end-to-end QA, including cancellation/recovery and mobile
scrolling. Its full result is byte-identical to the accepted native C API output;
all original final geometry/projection comparisons pass. Video was added later
as described above.
See [demo evidence](reference/BODY_DEMO.md). Normal regression is now 48 native
sanitizer tests, 112 Python tests and the Go demo tests.

Photo upload now attempts bounded local FFmpeg conversion before person selection
for unsupported formats or oversized files. The UI previews the converted pixels
and retains a conversion/error notice; original uploads allow 128 MiB, prepared
inference inputs remain limited to 20 MiB / 16 MP. Real Chrome tests cover BMP,
17.5 MP resizing, a file over 20 MiB, malformed-input failure and recovery.

The earlier strict-F32 [Vulkan profile](reference/BODY_PERFORMANCE.md) measures **0.393–0.404 s
warm inference** (395 ms median), **2.669 s cold / 0.462 s warm browser jobs**, and **69% mean
warm GPU utilization** (99% peak, not sustained 90%). The live demo now uses a
bounded persistent model worker, a GPU-resident 32-block backbone graph, and
shared immutable decoder/head/MHR parameter caches. Warm uploads fell from
611.25 MB to 100.11 MB; downloads from 110.34 MB to 42.94 MB. Arithmetic-preserving
skinning hoists and tiled transposes remove additional CPU work.
All 646 upstream comparisons pass; all 529 captured tensors and public results
remain byte-identical. All 44 native regressions, actual NVIDIA repeated-input
backbone tests, 96 Python tests and Go race tests pass. Real Chrome QA includes
cancellation/recovery, warm reuse, original overlay and exact GLB export.
Reusable decoder/geometry graphs, remaining CPU/GPU alternation and the 90% utilization
target remain open; full CPU model numerical gaps are not waived.

The earlier [Vulkan optimization pass](reference/BODY_PERFORMANCE.md) reduced
native inference from 39.9 s to about 3.25 s, and the updated live demo completes
the real Chrome sample job in 3.32 s including exports. It retains UBSan,
assertions and strict F32, removes redundant backbone weight scans, retains
immutable MHR weights on CPU/GPU, and reuses bounded backend scratch buffers.
All 646 upstream checks pass; all 19 public result fields and the exported GLB
are byte-identical to the accepted outputs. All 43 normal CPU tests pass with
ASan/UBSan/LSan, the optimized build passes the same 43 tests, and 96 Python
tests pass. Real Chrome upload/cancellation/recovery, original overlay, exports
and history/mobile QA pass. Full decoder residency and real-time or
optimized-upstream performance parity remain unfinished.

Current milestones: Body raw-image composition passes 435 checks per backend
with synthetic SAM state and real MHR; Objects default image/mask preparation
passes 114 original checks and has a tested opaque C API. Objects PointPatch
conditioning passes 196 synthetic original checks per backend, including the
full default 256/8/768 architecture. Objects SSI normalization additionally
passes 475 operation/final-field checks against each original CPU/CUDA reference
(the native normalizer runs on CPU). The pointmap-aware joint resize/crop/rembg
chain passes 129 original checks, including the 4096x2160 official image with
explicitly synthetic supplied XYZ. Composed PointMap preprocessing now passes
536 original boundary/independent-final-field checks with its own intermediates.
Post-encoder condition fusion passes 140 original synthetic checks per CPU/Vulkan
backend, including 768/1024-channel inputs. The explicit point-only composition
from raw inputs through preprocessing, PointPatch and fusion passes 108 small
comparisons and a nonfinite-configuration rejection per backend. Its full-size
CPU and Vulkan cases each pass another 54 comparisons (162 per backend), with
CPU sanitizers enabled. Normal tests are 42 native sanitizer
tests and 96 Python tests; the Objects image C API,
internal SSI, joint-transform and composed-preprocessing components each pass
100,000 sanitizer fuzz cases. Official Body/Objects access was confirmed on
2026-09-09. Body's checkpoint, MHR asset and configuration now pass local
size/SHA-256 verification; Objects' six principal inference checkpoints and seven
pipeline/component configs also pass. Auxiliary model closure and trained-model
parity remain pending.

Real Body checkpoint extraction and backbone GGUF conversion now pass exact
safe-tensor readback and native sanitizer-enabled archive validation. Stored
all-zero QKV bias masks exposed and fixed a factory-only validator assumption.
Trained backbone CPU/CUDA references are repeatable. All 36 trained backbone
stage checks now pass on CPU/Vulkan under a separately frozen original-control
F32 policy, after improving Vulkan dot-product accumulation. Original tighter
final-feature limits also pass. Legacy failures remain recorded. Actual
math-SDPA operand scaling has been traced and matched: all 704 trained
own-intermediate operation checks now pass on CPU and NVIDIA Vulkan under the
frozen original-control policy. Observer-neutrality and synthetic regressions
also pass; see [operation evidence](reference/TRAINED_OPERATIONS.md).
See [trained evidence and remaining diagnostics](reference/TRAINED_BODY.md).

The trained raw-RGB Body branch is now connected: its first Vulkan run passes
646 comparisons, including actual operations in all six decoder layers and all
final deformed vertices. This uses the real no-mask embedding and trained
hand-box heads. CPU passes 640/646 checks, including every decoder operation
and last-layer final field. Four internal centimetre-vertex checks and one
earlier vertex projection (checked twice) still narrowly fail. Original MHR
replay localizes part of that error to propagated pose-input rounding.
The 316-tensor decoder/head GGUF companion is now converted and independently
validated. An internal model session takes only RGB, box/intrinsics and three
GGUFs: its Vulkan result is byte-identical to the earlier inline-weight runner
and passes all 646 comparisons. Normal tests cover the companion's identity,
typed indices, malformed metadata and input extraction. The public opaque Body
pose-branch C API is now implemented: a real pure-C Vulkan call returns all 19
fields byte-identical to that accepted run, after freeing its input buffers
before inference and its model/request before reading results. It exposes
explicit capabilities, tensor shapes and coordinate metadata. A separately
configured C-only installed consumer passes sanitizer tests. Request/options
and result getters each pass 100,000 sanitizer fuzz cases (seeds 932/933);
GGUF loading is excluded. See the [API guide](docs/API.md). Hand refinement,
lower-level feature APIs and Objects inference remain unfinished.
See [trained branch evidence](reference/TRAINED_BODY_BRANCH.md).

Hand refinement is now audited through its actual full-method crop, separate
decoder, validity checks, body reprompt and IK/geometry merge. The first native
component—hand-box inverse mapping, left mirroring and padding-0.9 crop
preparation—passes 69 exact original comparisons, including trained-example
normalized pixels. A compact original regression and 100,000 sanitizer fuzz
cases pass. Wrist-frame conversion and both hand masks now pass 21 comparisons
against each original CPU/CUDA capture, with exact masks, plus 21 synthetic-only
comparisons, a normal regression without checkpoint buffers and 100,000
sanitizer fuzz cases (seed 936). The separate learned hand head is now connected
through complete MHR geometry: Vulkan passes 158/158 operation/final-field/mask
checks; CPU passes 156/158, with two internal centimetre-geometry limits still
open. Original replay attributes these to propagated global pose-input rounding,
not a large same-input MHR decoding error. All final fields pass, but full CPU
head acceptance is not claimed. The six-layer learned hand decoder now runs
with its own head/geometry feedback, shared sparse prompt encoder, separate
dense positional encoding and actual hand-specific weights. Feature-input
stress tests pass 328/593 comparisons on CPU and 383/593 on Vulkan; strict
hand-decoder parity is not achieved. Original CPU versus CUDA also fails
263 of the same 593 comparisons on identical inputs. These controls identify
reference sensitivity, not permission to waive failures. The learned hand-image
composition now includes original padding-0.9 preprocessing, shared DINO/no-mask
embedding and shared box heads. Right-hand Vulkan passes 590/646 generic checks,
including every independent final field; its original CPU/CUDA control passes
538/646. These use a fixed original-derived hand ROI, not native body-to-hand
selection. Intermediate numerical acceptance and final refinement remain open;
see [hand-image evidence](reference/HAND_IMAGES.md) and
[hand refinement evidence and next gates](reference/HAND_REFINEMENT.md).

The optimized sanitizer CPU preset now passes all 42 normal tests without
disabling assertions or ASan/UBSan. A global OOM killed the invoking Codex
process during overlapping verification; the affected run's partial artifacts
are not treated as clean-exit evidence. A serial cgroup-bounded runner now
verifies actual limits and preserves host headroom. Its intentional tiny OOM
test stayed inside the test cgroup, and subsequent lock recovery passed.
See [memory safeguards and evidence](reference/MEMORY_SAFETY.md).

See [Body evidence](reference/BODY_IMAGE_FLOW.md) and
[Objects image evidence](reference/OBJECTS_IMAGE.md) and
[point-window evidence](reference/OBJECTS_POINTPATCH.md) and
[SSI evidence](reference/OBJECTS_SSI.md) and
[joint-transform evidence](reference/OBJECTS_JOINT.md) and
[composed-preprocessing evidence](reference/OBJECTS_PREPROCESS.md) and
[condition-fusion evidence](reference/OBJECTS_FUSER.md) and
[point-conditioning composition](reference/OBJECTS_POINT_CONDITION.md) for the limits of these claims.

## Verification history (earlier counts and intermediate failures retained)

- Official Body and Objects source checkouts are at the manifest's exact pins
  and clean. Generated source-only preflight reports explicitly leave
  `reference_ready=false`.
- `reference/preflight.py` validates pinned Git state and selected artifact
  sizes/SHA-256s without checkpoint deserialization. It rejects escaping paths,
  missing bytes and dirty/mismatched source.
- `scripts/check_parity.py` checks ordered safetensors operation boundaries,
  complete tensor sets, shape/dtype, finite values, exact discrete outputs and
  both floating error metrics. Missing/incomplete captures cannot silently pass.
- Twelve weight-free tests pass, including negative CLI, incorrect layout,
  missing taps, near-zero reference, large integer index and source/hash cases.
  These establish tool behavior, **not SAM model parity**.
- Python data-tool dependencies are pinned in `pyproject.toml`/`uv.lock`.
  Checkpoints, source clones, generated reports and environments are ignored.
- The native C++23 Body crop-geometry component and opaque C API build with
  GCC/Clang. The C-only ownership/error/geometry test passes with ASan/UBSan and
  LeakSanitizer (outside the ptrace sandbox required by LeakSanitizer), including
  a separately compiled C consumer of the installed library/headers.
- Actual pinned upstream bbox functions were run in three fresh isolated
  containers without weights, network or GPU. All three captures are identical.
  Across 135 cases/675 tensors, native F32 center/padded/prior/final scale values
  match exactly. After the pixel-rounding fix below, affine F64 tensors also
  match byte-for-byte (previous direct-solve error was up to `1.82e-12`).
  This is bbox/affine operation parity only, not complete B1/R0.
- A Clang libFuzzer target covers the crop C API with sanitizers enabled;
  100,000 executions completed without a sanitizer finding (seed `336374588`).
- Body RGB U8 affine resampling, ToTensor scaling and CHW normalization are now
  native. All 24 tensors across 12 reference cases match byte-for-byte, including
  the official dancing photograph and rotated/off-image crops. Captures invoke
  original `GetBBoxCenterScale`, `TopdownAffine`, torchvision `ToTensor` and
  `BaseModel.data_preprocess`; no learned weights are used.
  Three fresh image captures are byte-identical. The expanded image/crop fuzzer
  completed another 100,000 sanitizer-enabled executions (seed `391`) without
  a finding. The rotated-pattern full-RGB checksum is now a normal CTest regression.
- Pixel-level comparison caught a real downstream issue missed by the earlier
  affine tolerance: the direct affine solve differed by tiny F64 amounts that
  crossed coordinate-rounding boundaries. Matching OpenCV's LU operation order
  fixed all 23 differing RGB channel values without relaxing pixel tolerances.
  C image tests cover strides, buffer bounds, size limits, normalized layout and
  owned-result lifetime. The full neural models and demo remain unimplemented.
- GGML 0.23.0 is pinned as an unmodified public Git submodule at
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`. CMake builds dynamically loaded
  CPU and optional Vulkan backends. The integration attaches the SPIR-V headers
  package interface for split-prefix SDKs without modifying upstream source.
- The first internal GGML graph performs F32 patch extraction, projection and
  bias addition in DINO token order. It explicitly avoids the convenience
  convolution helper's F16 im2col default. Every operation must be supported on
  the selected backend; a caller can require an exact device description.
- Original pinned DINOv3 `PatchEmbed` with stored synthetic weights passes all
  12 CPU checks, including the 512x512/1280-output Body shape. Three isolated
  upstream captures are byte-identical (SHA-256
  `43bef3a86c3f36d7e0a3d411f9ab4fb670fa7c9af0cf784356924332e4ce3117`).
  Intermediate unfold/bias-free-conv diagnostics are explicitly auxiliary
  PyTorch operations; final tokens come from the unmodified upstream forward.
  This is operation-contract testing, **not trained DINO or Body parity**.
- The installed NVIDIA 595.71.05 Vulkan ICD fails initialization with ASan loaded;
  its error is failure to obtain `vkCreateInstance` via `vk_icdGetInstanceProcAddr`.
  ASan/UBSan Vulkan smoke succeeds on AMD, and a separate UBSan-only build
  successfully runs the smoke test on the exact NVIDIA RTX 5070 Ti. CPU tests
  retain ASan/UBSan; neither workaround nor AMD discovery counts as NVIDIA parity.
- NVIDIA now passes all 12 patch-contract checks with the same frozen limits as
  CPU when Vulkan uses strict F32 arithmetic. Default backend math failed first
  at `case.0001.projection` (`9.69e-4` max error); patch extraction remained exact.
  Its result matched explicitly F16-rounded operands within `4.77e-7`.
  Disabling cooperative-matrix2 alone or all cooperative matrices was insufficient:
  the regular FP16 shader path also rounds operands. The fixture runner now sets
  `GGML_VK_DISABLE_F16=1`, `GGML_VK_DISABLE_COOPMAT=1` and
  `GGML_VK_DISABLE_COOPMAT2=1` in its child process and records backend settings.
  Strict-F32 candidate SHA-256 is
  `f4cb597ace0cf45733c36c989a2d72d49b5569c1bb62e848ec38e5575e3c0436`.
  Direct graph consumers must use these settings before backend initialization
  until per-session precision control is implemented. No tolerance was loosened.
- Six native tests now pass under ASan/UBSan/LSan, including rejection of the
  wrong backend, device index and required device description. A further 100,000
  crop/image C API fuzz runs passed with ASan/UBSan (seed `1953`).

- DINOv3's eval transformer block now has native affine LayerNorm (`eps=1e-5`),
  masked key bias, QKV/head layout, axial RoPE, attention, projection, LayerScale,
  residuals and SwiGLU. Training-only coordinate rescaling remains inactive.
  All 88 synthetic-weight block boundaries pass on ASan/UBSan CPU and strict-F32
  NVIDIA Vulkan, including the real Body block dimensions: 1029 tokens, 1280
  channels, 20 heads and 5120 hidden channels. Worst max errors: `6.68e-6` CPU,
  `2.82e-5` NVIDIA, both at the SwiGLU hidden activation. This is **block contract
  testing**, not trained backbone/model parity. Native uses its own intermediates.
- Original block/RoPE methods are observed without changing the uninstrumented
  output. Auxiliary logits/probabilities and layout/prefix views are explicitly
  distinguished from real module taps. Three fresh captures are byte-identical:
  `2cac9c767d06ec99fa0bb8edde54979e2bd0420a7c7e87772d3e87049c370e10`.
- A 93KB original-upstream synthetic fixture adds 22-boundary block regression
  coverage to normal CTest, without Python or downloaded weights. It covers
  two batches, multiple heads, five prefix tokens and a non-square grid.
  All seven native tests pass under ASan/UBSan/LSan, including invalid block
  parameter rejection. Provenance and frozen dual-error limits accompany it.

- Direct Vulkan versus original PyTorch CUDA block comparison also passes all
  88 checks on the RTX 5070 Ti, with TF32 disabled in the reference. Native uses
  the CUDA fixture's exact parameters: one generated RoPE period differs from
  CPU initialization in the two large cases. The CUDA reference is captured
  through the explicitly selected NVIDIA CDI device; generic Docker `--gpus all`
  hit an unrelated missing AMD CDI specification on this host. Reference SHA-256:
  `97b2aca42a288f175aa20cd459b6a77956518fe79c6fb9add37bf18a32fd0c79`.
  Three fresh CUDA captures are byte-identical. This is correctness evidence
  for a synthetic block, not a performance benchmark.

## GGUF foundation

- Added a safetensors-only F32 Body DINOv3 backbone converter, exact component
  metadata and a bounded native GGUF reader with per-read memory budgets.
  Conversion verifies input hashes, shapes, dtype, finite values and domain
  constraints, and never overwrites an existing output. No pickle on the host.
- Both compiled native and Python schemas match all 552 state tensors from the
  original H+ factory constructed on PyTorch meta: 840,633,600 learned parameters
  and 840,756,496 state elements. Reference report SHA-256:
  `fdebddef348bab05377dc1477db295adc30910ed01c55e9e2338a0d01e7a1b8f`.
  Its checked-in JSON contains shapes/provenance only, no weights.
- Converter-to-native interoperability passes for small synthetic data; GGML's
  own writer independently supplies the native round-trip test. The full schema
  verifier rejects the synthetic two-tensor file. All 18 Python and eight native
  ASan/UBSan/LSan tests pass, including deterministic malformed-input cases.
  GGUF loading remains excluded from fuzzing per the design.
- This is format/architecture validation, **not trained backbone parity**.
  Trusted extraction, learned conversion, complete models, inference C API and
  demo are still unfinished. See [reference/GGUF.md](reference/GGUF.md).

## Whole backbone composition

- Added complete F32 patch/prefix/block/final-normalization/feature-layout
  composition and a checked GGUF parameter-provider adapter. Each block runs
  with native-produced inputs, retaining only output activations and loading
  parameters sequentially. It is not yet a resident, performance-tuned graph.
- Original Body wrapper + DINO synthetic small cases pass 13 boundaries on
  ASan/UBSan CPU. A 64KB upstream-produced three-block fixture adds a normal
  CTest regression, including rejection cases. All nine native tests pass
  ASan/UBSan/LSan; all 18 Python tests pass.
- Full 32-block H+ shape, 512x512 input, passes all 36 boundaries on strict-F32
  NVIDIA Vulkan versus original PyTorch CUDA. Worst absolute error `3.30e-5`
  at block 24; final `[1,1280,32,32]` feature error `1.08e-5`, relative L2 `1.98e-6`.
  Limits remain `1e-4` and `2e-5`. Upstream instrumentation preserves the
  uninstrumented original output exactly. Reference SHA-256:
  `f971deeabe0be05119590a6319d6df3b134bf51c4bee1918775cc088b22994cf`;
  Vulkan candidate `4bf207e7d23c112ce4143806418a7cb6c681f5b24bf3c85e1c77cab4dc39f79f`.
  Three fresh upstream CUDA captures have identical parameter/input, rule and
  output hashes; output files were independently rehashed after capture.
- Full-size ASan/UBSan/LSan CPU execution also passes the same 36 checks against
  the exact CUDA reference inputs, with 12 CPU workers. Candidate SHA-256:
  `4087c43b0c85d2a02be867af9a0497ea57a2decb4215a6946923cc3ab448db4d`.
  The earlier single-worker full debug run was deliberately terminated for
  throughput, not a crash/OOM. Small cases are byte-identical at one/12 workers.
  Native fixture execution now emits live stage messages and hashes large
  inputs in bounded chunks; these correctness runs are not performance parity.
- This advances composition testing, **not trained Body parity**, and does not
  include the decoder, MHR, Objects or a web demo. Full-shape fixture files are
  generated/ignored, not distributed learned weights.

## Camera encoder component

- Implemented F32 ray antialias downsampling, 99-channel Fourier encoding,
  feature concatenation, 1x1 projection and explicit LayerNorm2d (`eps=1e-6`).
  Neural operations execute on the selected GGML backend; ray filtering and
  tensor-layout preparation are native C++. No PyTorch native dependency.
- All 32 original CameraEncoder boundaries pass ASan/UBSan CPU and strict-F32
  NVIDIA Vulkan, including the 512x512 ray/1280-channel feature shape and factors
  1/2/3/16. The reference hooks preserve unobserved outputs exactly. Frequencies
  are explicitly labelled auxiliary diagnostics, not original internal taps.
- Direct Vulkan/PyTorch-CUDA comparison also passes all 32 boundaries. Worst
  error `2.00e-5` in a four-channel normalization case; real-shape final-feature
  error `4.06e-6`, relative L2 `6.02e-7`. No tolerances were loosened. CPU reference
  SHA-256: `88d549dfb636ed3699617ffcdf04ab72f2219ec3b9e17b002ab3046410f7e1c4`;
  CUDA reference: `8dce0947f22716f02536ef85436c1869544901b2a61685981b243d567bc0285c`;
  Vulkan candidate: `169848d6ed68413369e1603a33249881780a42600606b72aecca63f36601c485`.
  Three fresh original CUDA captures have identical input, rule and output
  hashes; output files were independently rehashed after capture.
- The 40KB original synthetic camera fixture adds eight-boundary regression
  and malformed-input tests to normal CTest. All ten native sanitizer tests and
  18 Python tests pass. The extra reference image adds only hash-pinned einops
  0.8.1 to the existing PyTorch image; no Nix rebuild was performed. Image ID:
  `sha256:1dd15f53b07cb663477dc2345e6055051decf1adcd70ba12c2115162d00174d6`.
- These tests supply synthetic features/rays/parameters. Native construction of
  rays and CLIFF conditioning from crop/intrinsics, composed backbone-to-decoder
  execution, learned weights and final pose/mesh parity remain pending.

## Crop-to-camera geometry and C API

- Added opaque camera request/result handles, explicit intrinsics/image-size
  setters, both CLIFF-center conventions, owned ray grids/condition vectors and
  borrowed-buffer getters. Invalid setters preserve previous valid state;
  unsupported rotated or non-square camera crops are rejected. Metadata is
  rounded to F32 at upstream's actual batch-preparation boundary.
- All 60 original crop/batch/ray/CLIFF boundaries match byte-for-byte, including
  512² rays, off-image/fractional boxes, unequal focal lengths and both center
  conventions. Reference/native SHA-256:
  `0e7e553647f05a73f224ded93dcdad9d571994febf5f652ceebf53a3a1f33572`.
  Three fresh reference captures have identical inputs/rules/outputs; output
  files were independently rehashed. No upstream crop geometry is injected
  into native computation.
- The two original SAM3DBody method ASTs execute unchanged, using original
  preparation/transforms and BaseModel flattening. This avoids unrelated model
  imports/assets; it does not establish full-model entry-point parity. Upstream
  hardcodes CUDA for rays; native geometry itself has no GPU requirement.
- A 14KB original fixture adds 60-boundary regression to normal CTest. All 12
  native ASan/UBSan/LSan tests and 18 Python tests pass. Pure-C tests cover copied
  inputs, independent result ownership, error-buffer truncation, output resets,
  invalid requests, unequal focal lengths and unsupported crop rejection.
  The same pure-C test also passes against a separately installed header/library
  with ASan/UBSan/LSan, not just the in-tree build.
- Two 100,000-case sanitizer fuzz runs pass (seeds `8503`, `8504`); the second
  harness also explicitly promotes square/unrotated crops to reach valid ray
  paths. Peak RSS: 408/368 MB. An earlier direct-launch run reported 2109 MB at
  its initial empty input and hit libFuzzer's default 2048 MB guard. A genuinely
  forked child started at 33 MB and completed without changing that limit or
  disabling sanitizers, consistent with inherited launcher high-water accounting.
  The empty guard artifact was retained under ignored generated diagnostics.
- Composed crop/image/backbone/camera-to-decoder inference, learned Body weights,
  final geometry, Objects and the real web-demo workflow remain unfinished.

## Body decoder layer

- Added the F32 `TransformerDecoderLayer` contract: separate Q/K/V projections,
  LayerNorm eps=1e-6, repeated normalized positional embeddings and first-layer
  self-PE skip, self-attention, token-to-image attention, exact-erf GELU FFN,
  residuals and optional image-to-token attention. Masks preserve the upstream
  self diagonal and zero attention for fully masked reverse rows. Batch-one
  positional embeddings broadcast without replacing native intermediates.
- All 290 boundaries pass ASan/UBSan CPU and strict-F32 NVIDIA Vulkan against
  original PyTorch CPU; Vulkan also passes directly against original PyTorch
  CUDA. CPU/CUDA inputs/parameters are byte-identical. Seven cases cover mask
  edge cases, PE options, batch two, unequal token/context dimensions and a
  143-token/1024-image-token, 1024/1280-channel layer with a 4096-wide FFN. This
  large synthetic shape is not a claim of learned checkpoint configuration.
  Frozen limits remain max-absolute 1e-4 and relative-L2 2e-5; worst errors are
  6.20e-6 CPU and 1.34e-5 Vulkan/CUDA, both at large cross-attention logits.
- Original calls run unchanged with PyTorch's MATH SDPA backend and TF32 off;
  observer/non-observer outputs are exactly equal. Logits/probabilities are
  explicitly auxiliary expansions from original projected Q/K taps, not a
  replacement implementation used to produce reference final outputs.
  Three fresh CUDA captures have identical verified input/rule/output hashes.
  CPU reference SHA-256:
  `0edb82e516a82bf2a393e232ca61f830fbdfdabc7c5a878c88f9a91ff0c77161`;
  CUDA reference:
  `53ffae2b3f5bb3cfff97281a8438d72597c94b34c4c4e378a5294f6015a98ef7`;
  native CPU:
  `d655eac4b9c424dac4554760ff34a6d94381e493a3e227b3a47a308440825210`;
  native Vulkan:
  `913a552a4ab480ba92e5ace25eda8f01f549a95a93d2361f7f3562df33515474`.
- A 156KB original synthetic fixture adds 244-boundary, six-case regression to
  normal CTest, plus malformed inputs and native observer/no-observer equality.
  All 244 compact boundaries also pass NVIDIA Vulkan, including odd head width.
  All 13 native sanitizer tests and 18 Python tests pass. No Nix derivation or
  GGML-source patch was required.
- This is one decoder-layer variant, not learned Body inference. Prompt-token
  construction, complete layer-stack head/geometry feedback, learned conversion,
  Objects, the real demo and performance gates remain unfinished.

## Body prompt encoding

- Added original `PromptEncoder` and dense/pixel `PositionEmbeddingRandom`
  semantics: centered xy, stored Gaussian matrix projection, 2π angles,
  sine/cosine channel order, dense NCHW layout, separate invalid/not-a-point
  learned embeddings, valid joint embedding addition and F32 point masks.
  The native path needs no Python runtime. Mask-convolution variants and an
  absent point tensor are not implemented here; baseline Body supplies [0,0,-2].
- All 108 original-operation boundaries pass ASan/UBSan CPU versus PyTorch CPU
  and strict-F32 NVIDIA Vulkan versus PyTorch CUDA. Four cases include batch two,
  rectangular grids, coordinate endpoints, all 70 joint labels plus -2/-1, a
  32²/1280-channel dense grid and off-image pixel coordinates. Coordinate and mask
  checks are exact; other limits remain max-absolute 1e-4 and relative-L2 2e-5.
  Worst errors: 3.81e-6 CPU (dense angles), 7.84e-6 Vulkan/CUDA (pixel cosine).
- The first CUDA comparison failed five exact-coordinate checks. Inspection
  localized this to scalar division before any neural operation: CUDA's original
  scalar fast path multiplies by a rounded F32 reciprocal, while CPU divides.
  Added an explicit scalar-arithmetic mode and selected the matching reference
  convention in the diagnostic. All 108 checks then pass without changed rules.
  The initial mismatch report is retained in ignored generated diagnostics.
  Future model composition must preserve this selected arithmetic convention.
- Original `_pe_encoding` executes unchanged under a non-replacing ATen output
  observer; no reference intermediate is recreated by a substitute oracle.
  Original observed/unobserved outputs are exactly equal. CPU and CUDA capture
  inputs/state have identical hashes. Three fresh CUDA captures have identical
  independently verified inputs/rules/outputs.
  CPU reference SHA-256:
  `721f674193d767072a7082739bf8b6c7a7c39f6faa583350f18e36c6c453e5a0`;
  native CPU:
  `bec8930d25d4d98a58f5627ff00e375b9281b1e226c23b493a4051f34bea85db`;
  CUDA reference:
  `76a11f6ab93c7553e43d070db528c68452aaedd887f9785ec81eb55558bf01ad`;
  native Vulkan:
  `1c88d5b0d42e80102fa326d1371ae9c975af8231e3a05aeb8e257c5e01105616`.
- Two 50KB original fixtures add 162 boundary checks across CPU and CUDA scalar
  arithmetic to normal CPU-only CTest. Invalid coordinates, fractional/out-of-range
  labels, non-finite state, missing/extra/wrong-size parameters and malformed
  shapes are rejected. All 15 native sanitizer tests and 18 Python tests pass.
- This remains synthetic prompt-component parity, not complete conditioning,
  a learned decoder, final Body geometry, Objects or a working web demo.

## Conditioning through the first decoder layer

- Added native DINO/CLIFF Body input construction: learned initial pose/camera,
  explicit previous-estimate fallback or supplied previous estimate, GGML token
  projections, native-produced camera-conditioned features and prompt encoding,
  optional hand/3D-keypoint tokens, and the exact token/augmentation order. The
  decoder receives no mask even when prompt masks are zero, matching upstream.
- All 52 original-method composition boundaries pass ASan/UBSan CPU against
  PyTorch CPU and strict-F32 NVIDIA Vulkan against PyTorch CUDA. Three small
  cases cover batch two, explicit/default previous estimates, rectangular grids
  and optional tokens; a large case uses 512² rays, 1280-channel features and
  143 decoder tokens. CPU/CUDA raw input and parameter bytes are identical.
  Worst max errors: 4.65e-6 CPU (conditioned features), 2.43e-5 Vulkan (first-layer
  context); frozen limits remain max-absolute 1e-4 and relative-L2 2e-5.
  Three fresh CUDA captures have identical independently rehashed inputs, rules
  and outputs.
- The reference executes the unchanged `SAM3DBody.forward_decoder` AST with real
  original PromptEncoder, CameraEncoder, PromptableDecoder and first layer.
  An explicit hook terminates after that layer, before pose-head/geometry
  callbacks. Baseline/observed first-layer outputs are exactly equal. No fake
  final pose is returned. Native composition uses only its own intermediates;
  input features/rays/CLIFF are synthetic, not yet produced by the full image
  pipeline. This is not complete decoder or Body-model parity.
- Current MHR source defines 519 pose values (6+260+45+28+108+72), despite stale
  forward-decoder comments mentioning 404. The large synthetic case uses the
  current definition: 522 pose/camera values and 525 with CLIFF. Learned tensor
  compatibility still needs validation against authorized checkpoint bytes.
- CPU reference SHA-256:
  `c2f5b2876848178e34707d4aa9bc71e8c061e104a7a2b6a03b734f2408c518f1`;
  native CPU:
  `6316c4dfb9461cf2905f999563683805bc3c45055e054f20b559661bc308ddf7`;
  CUDA reference:
  `ee662d28cbef3029c9b69f1c1610876ae8619fea92f75c59e1bc2598cde1bc5f`;
  native Vulkan:
  `9765f14e2b18ec3fe16aed9e1d0fbba5eb97cbd7258edd5c20f3f573a1ae6ce0`.
- A 131KB original fixture adds 39 composed boundary assertions and rejection
  cases to normal CTest. All 16 native sanitizer tests and 18 Python tests pass.
  Full-layer-stack pose/geometry feedback, trained weights, image-to-pose output,
  Objects, demo and performance acceptance remain unfinished.

## Camera head and full-perspective projection

- Added the original PerspectiveHead contract entirely with F32 GGML operations:
  one-to-three-layer ReLU FFN, optional initial-camera residual, scale/vertical
  sign changes, crop-size/default-scale conversion, both center conventions,
  camera translation, depth normalization and full intrinsics application.
  Inputs are checked; zero-depth/non-finite projections are rejected without
  adding a clamp or changing the upstream formula for valid inputs.
- All 68 original operation boundaries pass ASan/UBSan CPU versus PyTorch CPU
  and strict-F32 NVIDIA Vulkan versus PyTorch CUDA. Five cases cover initial
  estimates, FFN depths, unequal focal lengths/off-diagonal intrinsics, both
  depth signs, both center conventions, a 1024-wide token and a synthetic
  18,439-point set. The point count is representative, not verified MHR topology.
  Raw input and parameter bytes match between CPU and CUDA reference captures.
- The first mismatch was exactly at intrinsics multiplication: an unnecessary
  transpose applied K in the wrong orientation. Every preceding operation
  matched. Removing that transpose fixes all ten affected projection/pixel taps;
  the initial failure evidence is retained. Tolerances were not changed during
  correction: 1e-3 for pixel/box quantities, 1e-4 for camera/latent quantities,
  relative-L2 2e-5 and exact focal lengths. Worst pixel errors are 6.10e-5 CPU
  and 1.22e-4 Vulkan.
- Original observed/unobserved head and projection outputs are exactly equal.
  Three fresh CUDA captures have identical independently verified input/rule/
  output hashes. CPU reference SHA-256:
  `8b49bbd67d55100eca14860cdfc55faa6d818ef3cd12549fac1565e65972e384`;
  native CPU:
  `d3ab20f990d0c25b91ed55dd0421848b52d42d5e7ce7ce00974ffefe9570c5b3`;
  CUDA reference:
  `950b4f441e0fb5b00a15aaf9a5b2fbbfd6d2d03cce02675f254c431279288945`;
  native Vulkan:
  `c040b8c38e569007445bafccb5467b6d8ca7c4517065033f54054e88a08f1ede`.
- A 16KB original fixture adds 42 boundary assertions plus input/error cases to
  normal CTest. All 17 native sanitizer tests and 18 Python tests pass. This
  component takes synthetic tokens/state/3D points; it does not generate MHR
  geometry or complete the decoder's pose/geometry feedback loop.

## Crop projection and keypoint feedback

- Added full-image-to-crop affine projection plus the original DINOv3-path
  2D/3D keypoint token updates. F32 GGML evaluates the affine/projection/MLP
  graphs; checked native CPU code performs bilinear feature sampling, indexing
  and token updates. This is a correctness milestone, not GPU-resident feedback
  or performance acceptance. Backend-resident composition remains required.
- All 77 original operation boundaries pass ASan/UBSan CPU versus PyTorch CPU
  and strict-F32 NVIDIA Vulkan versus PyTorch CUDA. Five synthetic cases cover
  batches, shuffled keypoint selection, non-square and single-pixel feature
  maps, rotated/sheared crop transforms, optional 3D tokens, final-layer identity
  and a 143-token/1024-wide case with 1280-channel 32x32 features and 70+70 points.
- Preserve zero-padding with `align_corners=False` (not border clamping), exact
  crop/depth validity predicates, and feature masking before linear projection:
  invalid samples still add the projection bias to their tokens. Positional
  updates are replaced, not added; 3D coordinates are hip-centered before
  keypoint selection. Other tokens and the final-layer state remain unchanged.
- Reference capture executes unchanged upstream method ASTs with the original
  FFNs and grid sampler. Non-replacing operation observers, module hooks and
  return-local capture leave original outputs exactly unchanged. Three fresh
  CUDA captures have independently verified identical input/rule/output hashes.
  CPU reference SHA-256:
  `54e192350213fb767847617d659f9040996f00bd513c091a1fe0ee0e2d035efb`;
  native CPU:
  `4581506ceaf601d38851d8f9c6655d515df0aadd2cf1bfc3c9f7d43793d103f7`;
  CUDA reference:
  `137b5caa479a811c6a8fca8154b84eb6f4f83f14e9617a2e3eba6dcacddf2b95`;
  native Vulkan:
  `d6e2df58f1ef9bbda776461ad97e50c2c909105d12eaaf2f16db380c840ce83b`.
- The 96KB original fixture adds 58 boundary assertions, masking/bias checks and
  malformed-input rejection to normal CTest. All 18 native sanitizer tests and
  18 Python tests pass. Supplied geometry is synthetic; this does not close the
  pose-head/MHR feedback loop or establish learned-model parity.

## MHR pose head and complete geometry-input assembly

- Implemented the body-mode MHRHead prefix through the actual MHR call inputs:
  519-value FFN plus optional initial estimate, global/body rotation decoding,
  shape/scale/hand/face slicing, body hand/jaw masks, scale and hand PCA, hand
  rotation decoding and indexed assembly of the 204-value model parameter vector.
  These dimensions follow the pinned source, not its stale dimension comments.
- Global rotations retain the distinct Gram-Schmidt → RoMa quaternion → intrinsic
  ZYX Euler path. Body/hand 3-DoF rotations use the original cross-product 6D
  construction and singular branch; 1/2-DoF rotations use sin/cos pairs. The
  reference uses a SHA-256-pinned official RoMa 1.6.1 wheel only inside the
  isolated container. BSD/SciPy notices are retained with the native adaptations.
- All 320 original boundaries pass ASan/UBSan CPU versus PyTorch CPU and
  strict-F32 NVIDIA Vulkan versus PyTorch CUDA. Eight synthetic cases include
  FFN depths 1–3, initial estimates, full 1024-wide tokens, all four quaternion
  branches, zero/tiny/collinear/gimbal-lock body rotations, nontrivial PCA and
  shuffled explicit hand-index buffers. Rules remain max-abs 1e-4, relative-L2
  2e-5, exact discrete/singular masks and disabled translation/face. Worst errors
  are 3.46e-6 CPU and 4.77e-6 Vulkan, both in angular quantities.
- Unchanged original method ASTs run until the source line immediately before
  `self.mhr`; no geometry is substituted. Additional operation/local/module
  observation leaves all original MHR inputs exactly unchanged. Three fresh
  CUDA captures have independently verified identical input/rule/output hashes;
  CPU and CUDA use identical synthetic input/parameter bytes.
  CPU reference SHA-256:
  `1cd95109d1fae61ffd272509f8b5e43f1dab5f636afe40f53c7f4a342c45554d`;
  native CPU:
  `6f3f50c916d1677b33770762443d8afa3d42ee78431f04671ccebe1f1d7878c2`;
  CUDA reference:
  `745bf81639a23938509e84a38dadc3a582013b596175c1187b057b3659415f75`;
  native Vulkan:
  `97fafe4a8061b44cfeeeddd8d5d676031c962933addd9348722aa969b38e7aa8`.
- The six-case original fixture adds 240 normal CTest boundary assertions,
  invalid tensor/index/parameter rejection and explicit masking tests. FFN/PCA
  runs in GGML; trigonometry and scatter currently run as checked CPU geometry
  for both backends. This is not GPU-resident whole-model or performance parity.
  Actual MHR asset inference, final geometry mapping, complete feedback-loop
  composition and trained image-to-mesh acceptance remain outstanding.
  All 19 native ASan/UBSan/LSan tests and 18 Python tests pass.

## Public MHR geometry reference and real GGUF conversion

- Verified Meta's independently public MHR v1.0.1 archive and its included
  Apache-2.0 license. Its `mhr_model.pt` SHA-256 exactly matches the SAM MHR
  companion identity. This is the original MHR publisher's separate public
  distribution, not a mirror of denied SAM neural weights. Archive/model/license
  hashes and source revision are recorded in `reference/sources.json`.
- Loaded only that hash-verified asset inside the reviewed offline container.
  Original TorchScript forward runs on CPU and NVIDIA CUDA, with/without pose
  correctives, using the unchanged official MHR demo input generator. It produces
  `[2,18439,3]` vertices in centimeters and `[2,127,8]` skeleton states. Safe state
  extraction includes exact tensor inventory, names, prefix groups and embedded
  code identities; the small inventory fixture is checked in, not model weights.
- Important source/asset differences were identified before native geometry
  implementation: the released asset uses separate identity/expression matrices,
  45-value parameter padding, sparse COO corrective projection and F64 prefix FK.
  Current Python's combined blendshape/padding and dense projection workaround
  must not silently replace the released computation. Details: `reference/MHR.md`.
- Three original CPU processes repeat exactly. Three CUDA processes repeat
  skeletons/non-corrective vertices exactly; corrected vertices vary at up to
  seven coordinates by 1.91e-6 cm. Non-replacing tracing localizes the first
  varying initialized operation to sparse COO `[3000,750]` × `[750,2]` matmul,
  max-abs 1.49e-8. Observed/unobserved vertex difference is at most 9.54e-7 cm,
  skeletons exact; this is bounded, not bit-exact, instrumentation equivalence.
  Deterministic allocation NaN scratch is separated from numerical boundaries.
- Converted actual safe state to `sam3d.mhr.lod1` GGUF: 18 F32/I32 tensors,
  173,275,904 elements, 693,111,264 bytes. The native sanitizer-enabled reader
  validates every real tensor. Typed indices, domains, names, asset identity,
  units, precision and caller byte budgets are checked. No GGUF fuzzing added.
  GGUF SHA-256:
  `d52ab772628fb6d851550381b428185a32da6398793f0ff70299bad1999ba8f8`;
  safe extracted state:
  `61b99bf082917648313c6712a8d5bb4b27941588bc8b177425d1393adeec851c`;
  original CPU geometry:
  `2381ff6f9fa002b6348208233db82f023d6695780472d7a92798c6432f1a817c`;
  first original CUDA geometry:
  `c478ba8722e64d3ba833c29c849d27dec844b0a90f098b31f82c8b5900ec92d6`.
- All 20 native ASan/UBSan/LSan tests and 23 Python tests pass, including new
  typed conversion/loading and rejection cases. Native geometry inference and
  operation parity still remain; upstream standalone geometry is not SAM
  image-to-mesh, demo or optimized performance acceptance.

## Native MHR parameter-to-skeleton parity

- Implemented the real GGUF `[889,249]` F32 parameter projection and 45-value
  zero padding, local translation/Euler-quaternion/prerotation/exp2-scale
  computation, and original four F64 prefix-composition passes before F32 output.
  All downstream inputs are native-produced. GGML runs the projection on the
  requested backend; small trigonometry and F64 geometry run on CPU explicitly.
- Original released TorchScript methods provide 46 observed operation boundaries.
  Observation disables profiling optimization, not mathematical source, and is
  checked against the uninstrumented full-model skeleton: CPU exact; CUDA
  max-abs 1.53e-5 / relative-L2 5.15e-8. The uninstrumented output is separately
  retained and compared to native, not merely inferred from component passes.
- CPU with ASan/UBSan/LSan and strict-F32 NVIDIA Vulkan with UBSan both pass
  46 operation checks plus the uninstrumented-skeleton check. Maximum errors:
  CPU 7.63e-6; Vulkan versus CUDA 1.53e-5 (cm for translation components).
  Gates remain max-abs 1e-4 and relative-L2 2e-5. Source, input, backend, runner
  and tensor identities are recorded in the generated reports and MHR guide.
- Parent topology and the complete stored prefix schedule are validated, including
  same-pass source uniqueness and ancestor-chain composition, not just ranges.
  The 484KB normal original local/FK fixture supplies 44 boundary assertions plus
  malformed shape, nonfinite, scale overflow/underflow, quaternion, parent and
  schedule rejection. It intentionally tests local/FK in isolation, not the
  parameter projection. All 21 native sanitizer tests and 26 Python tests pass.
- This does not yet include identity/expression blendshapes, learned correctives,
  skinning/final vertices, body output mapping or complete decoder feedback.
  It is not whole-model GPU residency, optimized performance or SAM image-to-body
  acceptance. Momentum/asset attribution and installed licenses are recorded.

## Complete standalone MHR geometry parity

- Implemented the original identity/base and expression blendshapes, corrective
  features, both corrective projections/ReLU, inverse-bind transforms and weighted
  skinning. Real GGUF parameters and native-produced intermediates generate all
  18,439 final vertices in cm and 127 skeleton states. Correctives off and on are
  both tested. No upstream local/geometry results are injected in this full run.
- All neural projections use F32 GGML on CPU or NVIDIA Vulkan. The first COO
  corrective matrix is materialized densely without changing its values, then
  compared against the actual released sparse PyTorch operation (CPU max-abs
  4.92e-7, relative-L2 3.23e-7). CPU trig/F64 state math/skinning and staged weight
  allocation remain explicit; whole-model GPU residency/performance is not claimed.
- The original complete forward is observed at 19 boundaries without correctives
  and 25 with them, plus independently retained uninstrumented final vertices and
  skeleton per mode. A cached CUDA fused plan initially hid operation taps and
  failed capture. A fresh unoptimized observation instance fixes that; mathematical
  source/weights remain unchanged. CPU observed/unobserved results are exact;
  CUDA vertices differ at max-abs 4.58e-5 cm / relative-L2 1.32e-7, within the
  previously fixed 1e-4 / 2e-5 gates. This is bounded CUDA observer equivalence.
- Both native backends pass all 48 operation/final-output checks on the two
  official MHR demo samples. CPU final vertex max-abs 4.58e-5 cm (both modes),
  relative-L2 at most 5.41e-8. Vulkan/CUDA final vertices: 6.10e-5 cm without
  correctives, 4.58e-5 cm with them, relative-L2 at most 1.15e-7. CPU retains
  ASan/UBSan/LSan; NVIDIA retains UBSan under the existing ASan/ICD exception.
- A 96KB normal original 16-vertex/46-influence fixture adds 12 skinning boundary
  checks and shape/index/finite/weight/state rejection. Restricting the original
  module's vertex buffers reproduces exactly the selected original full-mesh
  vertices on CPU. This isolated fixture is not the composed GGUF test. All 22
  native sanitizer tests and 29 Python tests pass. Provenance/commands/hashes:
  `reference/MHR.md` and `tests/fixtures/mhr-skin.json`.
- Standalone released-MHR geometry is now implemented and validated on these
  inputs. Body's subsequent output mapping, decoder/pose/geometry feedback,
  trained image inference, Objects, web demo/Chrome QA and optimized end-to-end
  performance remain outstanding. No claim of full SAM Body model parity.

## Composed pose head, real MHR and Body outputs

- Implemented Body's cm-to-meter mapping, vertices/joints concatenation, full
  308-row keypoint projection and 70-point selection. Position Y/Z flips are
  applied only at the original boundary; joint rotation matrices keep MHR axes.
  RoMa's XYZW conversion preserves the original non-normalizing formula. CPU
  direct scalar division and CUDA reciprocal arithmetic are explicit.
- Connected native pose FFN/PCA/parameter construction -> real MHR GGUF geometry
  -> output mapping. All intermediate geometry is native-produced. Mapping and
  neural projections use GGML; CPU geometry/diagnostic transfers remain explicit.
  No production model/session C API or whole-GPU performance claim yet.
- The reference now runs the original full `MHRHead` class and its original
  loader, with the verified released MHR. SAM head/PCA/hand index/keypoint mapping
  state is explicitly synthetic; trained checkpoint buffers are still unavailable.
  The two cases cover batch two without initial estimate and a 1024-wide, two-layer
  head with one. The original `do_pcblend=False` is verified to have no effect
  beyond bounded repeat noise because that flag is not forwarded to the asset.
- CPU and NVIDIA Vulkan/CUDA pass 70 comparisons each: 46 observed boundaries
  and 24 independently retained uninstrumented final tensor comparisons. Observer
  equivalence: CPU exact, CUDA max-abs 1.49e-8. Native final vertex max-abs is
  4.77e-7 m on CPU and 5.97e-7 m on Vulkan/CUDA; keypoints are within 2.39e-7 m
  and 3.58e-7 m respectively. Raw-intermediate maximum includes MHR centimeters:
  4.58e-5 CPU / 5.35e-5 Vulkan. Frozen 1e-4 / 2e-5 gates remain unchanged.
- The 268KB normal mapping fixture omits only zero synthetic mapping columns,
  retaining 16 vertices and all joints. It checks 12 original mapping boundaries,
  position axis signs, non-unit quaternion homogeneity and malformed inputs.
  This isolated test is not evidence for the composed path by itself. All 23
  native ASan/UBSan/LSan tests and 29 Python tests pass. Scope, commands and hashes
  are in `reference/BODY_OUTPUT.md`; normal provenance is adjacent to the fixture.
- This is real-geometry/synthetic-head composition parity, not learned SAM Body,
  semantic keypoint quality, full decoder feedback, image-to-mesh or demo acceptance.

## Complete promptable decoder composition

- Connected camera/prompt conditioning, every decoder layer, affine final
  normalization, pose head, real MHR, Body mapping, camera/keypoint/vertex
  projection and 2D/3D feedback in `src/body_flow.cpp`. All intermediates come from
  the native computation; no injected upstream mesh/camera/feedback is accepted.
- Original control flow retains fixed initial pose/camera residuals at every
  layer, updates raw tokens rather than normalized head tokens, and samples the
  original conditioned image rather than updated two-way context. Final-layer
  feedback is omitted; the optional hand-detection return slice is retained.
- The reference executes unchanged original Body methods, real original decoder
  and heads, and the released MHR asset. Two/six-layer trajectories cover batch
  two/one, optional hand tokens, a distinct previous estimate, repeated/nonrepeated
  PE, two-way context and both camera-center conventions. Instrumentation leaves
  original full tensor outputs exactly unchanged on CPU and CUDA.
- CPU ASan/UBSan/LSan and strict-F32 NVIDIA Vulkan/UBSan each pass 526 checks:
  372 boundary comparisons and 154 independent uninstrumented full-output checks.
  Vertex error across all layers is <=4.77e-7 m on CPU and <=7.16e-7 m on Vulkan;
  projected vertex error <=1.38e-4 / 2.45e-4 pixels respectively. Existing
  component tolerances were retained. Full evidence: `reference/BODY_FLOW.md`.
- Added eight original norm_final fixtures plus composition-contract rejection
  tests, and exact camera prediction/reused-camera projection regression.
  All 24 native sanitizer tests and 29 Python tests pass; no weight downloads in
  normal tests. Public C API is unchanged.
- Scope is deliberately small-width synthetic SAM parameters/features with
  real full MHR geometry. This is not learned semantic keypoint quality,
  full-width/model/image parity, hand refinement, demo or performance acceptance.

## Full-width decoder numerical investigation

Full-width decoder validation was added after the small-width composition gate.
Original CPU/CUDA captures use six layers, 1280 context channels, 1024 token
channels and real MHR. CPU passes 385/389, Vulkan 383/389; the only failed fields
are near-zero global Euler rotations, duplicated in diagnostic and full-output
checks. Their absolute differences are 1.1920928955078125e-7 radians, but relative
errors exceed 2e-5. **This full-width gate remains failed.**
Identical-input isolation traces locate CPU's first difference at `atan2` and
CUDA's at `hypot`, after bit-identical axes/matrices/quaternions. No inference
math or tolerance was changed; the new diagnostic calls the same native helper.
See `reference/BODY_FLOW.md` for evidence and the next numerical investigation.
An additional original CPU-versus-CUDA control on byte-identical rotation
vectors exceeds the same relative-angle criterion in five of six original
vectors, with the same 1.1921e-7 rad absolute difference. The first difference is
again `hypot`, not a native model operation. This is independent evidence of a
near-zero angular numerical-floor issue. At that stage criteria were unchanged
and the full-width gate was not marked passed; the resolution follows below.

## Angular-policy resolution and RGB composition

- Added a reference-supported global Euler metric: raw absolute angle error
  remains bounded by 1e-4 radians; raw relative-Euler error remains reported;
  relative L2 is bounded by 2e-5 on per-coordinate sin/cos pairs, whose norm does
  not vanish at identity. This is an explicit measurement change, not a numerical
  implementation fix. No other tensor limits changed.
- `reference/rotation-policy-v1.json` freezes original-only CPU/CUDA controls;
  native diagnostic inputs are excluded from calibration. Regression tests also
  reject meaningful angle errors, axis/unit/wrapping mistakes and malformed data.
- Fresh full-width six-layer CPU/Vulkan runs now each pass all 389 checks.
  Native output files are byte-identical to the initial failed runs; the old
  reports remain intact. Policy identity/hash and raw errors are in new reports.
- Added `sam3d_pipeline` / `body_from_rgb`: a native single-person RGB -> crop,
  normalized pixels, rays/CLIFF -> backbone -> complete decoder composition.
  It preserves F64 affine image sampling and F32 downstream affine metadata.
  Initially only its preparation/shape contract was tested, against original
  camera fixtures and malformed inputs. The full synthetic-state image chain
  has subsequently passed the composition test described below; public C API
  and trained-model sessions remain unchanged.
  All 25 native ASan/UBSan/LSan tests and 35 Python tests pass.

## Full RGB/backbone/decoder composition

- The official `dancing.jpg` image now runs through native preparation, all 32
  backbone blocks and all six decoder layers with their own geometry feedback.
  Parameters are synthetic SAM state plus real public MHR, not trained SAM.
- Independent original CPU-neural/CUDA captures execute the unchanged pose
  branch. Reference hooks only stream weights and observe outputs; original
  instrumented/uninstrumented outputs agree exactly. The original hardcoded
  CUDA ray computation is bridged even for the CPU-neural reference, so these
  captures are not pure-CPU or optimized performance baselines.
- CPU and NVIDIA Vulkan each pass all 435 boundary/full-output checks under
  existing limits. All nine preparation fields match exactly. Maximum mesh
  errors are <=7.16e-7 m CPU and <=5.97e-7 m Vulkan. The native input contains
  raw pixels, box/intrinsics and model state, not saved intermediate predictions.
- Reproducible commands, scope and artifact hashes are in
  `reference/BODY_IMAGE_FLOW.md`. All 25 native sanitizer and 35 Python normal
  tests pass. Trained inference, Objects and browser QA are not yet delivered.

## Objects default image/mask preparation

- Ported raw RGBA -> object/full RGB and mask tensors, retaining original
  exclusive mask bounds, truncation, off-image padding, background masking,
  bicubic antialiasing and nearest-mask semantics. No pixel clamping/extra norm.
- Original unchanged source captures cover five difficult small cases and the
  official kid_box photograph/mask at default 518x518 output. Three repeated
  uninstrumented runs and the observed run agree exactly. Native passes all
  114 boundary/full-output checks; max resize errors are <=8.345e-7 absolute and
  <=7.056e-7 relative-L2 under predeclared 1e-6 limits. 78 boundaries match exactly.
- Added opaque Objects preprocessing requests/results with constructors,
  setters/getters, borrowed owned buffers and checked C errors. All 27 native
  sanitizer tests and 37 Python tests pass. A separately installed pure-C
  consumer passes; no private C++ types are exposed in the header.
- The new C API fuzzer passes 100,000 cases, seed 60124, peak RSS 444 MB, with
  ASan/UBSan/LSan on. The first direct launch hit a 2048 MB guard while reporting
  3101 MB before its first input; a fresh forked child starts at 33 MB and passes
  without changing the limit, matching the earlier launcher high-water issue.
- Evidence and scope: `reference/OBJECTS_IMAGE.md`. This is CPU preprocessing
  (also used before device transfer), not Vulkan neural or complete O1 parity.
  Pointmaps/MoGe, learned conditioners and reconstruction are still unfinished.

## Current external dependency

The user confirmed requesting access to both official SAM repositories on
2026-09-09 around 04:45 UTC and expects approval in roughly an hour. Both pinned
configuration requests immediately before that message still returned HTTP 403.
Retry at or after approximately 05:45 UTC while continuing public-source work;
approval is not yet confirmed and no denied download has been bypassed.


Authenticated official HF CLI requests for both Meta model configurations
returned HTTP 403 and “not in the authorized list.” Body/Objects access must be
approved for the configured account before the official weights can be fetched.
The user has been asked to request/accept access. No mirrors were used to work
around the denial and no SAM neural checkpoint was loaded. This is an access prerequisite,
not evidence of an inference/runtime failure.
Both official configuration requests were rechecked after prompt implementation
and remain denied; no learned checkpoint was downloaded.
Rechecked again after the MHR-input prefix implementation: both official pinned
configuration requests still return HTTP 403, not authorized.
Latest recheck after standalone MHR mesh parity: the pinned Body configuration and
Objects pipeline requests still return HTTP 403 / not in the authorized list.
Neither SAM configuration nor neural weights were downloaded by these checks.
Rechecked again during full-width decoder validation: both pinned official
configuration requests still return HTTP 403 / not in the authorized list.
Rechecked after Objects image/mask preprocessing using revisions read directly
from `reference/sources.json`: both configuration requests remain HTTP 403,
not authorized. No neural weights were downloaded.

## Next work

Access is granted and principal weights are verified. Resolve the remaining
six CPU trained geometry/projection failures without weakening their gates;
fixed-token pose-head replay now separates incoming drift from head arithmetic.
The single-image upload/render/export/history demo now works through the public
C API and passes real headless browser QA against the original Body example.
Continue original hand-crop/refinement capture and native implementation, then
measure and improve matched end-to-end performance. Objects and video remain
deferred. The body-branch demo does not substitute for full Body acceptance.
