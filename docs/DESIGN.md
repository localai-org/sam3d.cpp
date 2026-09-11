# sam3d.cpp design and implementation process

This document defines architecture and the reference/parity process, not a live
task checklist. The [roadmap](ROADMAP.md) is authoritative for current scope and
remaining work; [development history](../reference/HISTORY.md) records experiments.

The supported implementation is the **SAM 3D Body pose branch**, with native
CPU/Vulkan inference and an image/offline-video/live-webcam demo. Video currently
uses independent frame estimates, not GEM temporal inference. Detailed hand
refinement, feature-only GEM integration and SAM 3D Objects are future work.
Sections describing those extensions are design, not claims of shipped support.
BF16 tolerances account for upstream floating-point variation, with isolated
hand metrics kept non-blocking for the supported Body branch. Full-model and
matched-workload performance acceptance remain separate gates.
Source and model identities are recorded
in [the audit](../reference/AUDIT.md) and [the source manifest](../reference/sources.json).

## 1. What this project provides

An independent C++23/GGML library for Meta's SAM 3D models:

- **Body (active):** recover human pose, shape, camera and ultimately an MHR mesh
  from an image. Expose the intermediate features that GEM-X actually consumes.
- **Objects (deferred):** reconstruct object geometry and appearance from an image
  and mask. It is a separate model family, not another head of Body.
- **gem-x.cpp is a consumer:** it owns GEM's learned temporal tracking model,
  SOMA decoding and downstream motion adapters. Neither library should require
  the other's demo, a Python runtime, or a physics engine.

GGML is the neural inference dependency, with CPU and Vulkan as required
backends. Small, audited image/geometry libraries are allowed; ONNX Runtime,
LibTorch, CUDA-only neural frameworks and llama.cpp are not runtime dependencies.
Python is allowed in the isolated reference and offline conversion tools.

Initial Body scope is one image/person with an explicit bounding box and camera
intrinsics. This avoids making a detector or field-of-view estimator a hidden
dependency. It is not advertised as parity with the entire automatic upstream
demo. Automatic detection, optional prompts and hand refinement get separate
acceptance cases. An image mask may be supplied by a caller: SAM 3 segmentation
is optional, not the same model as SAM 3D.

### Relationship to GEM-X and arm/wrist use

```text
image + box + intrinsics ── sam3d Body ── pose token + camera translation ─┐
video tracking + ViTPose + camera data ─────────────────────────────────┤
                                                    gem-x temporal model
                                                              │
                                                      SOMA motion + hands
                                                              │
                                          optional retargeting/controller
```

GEM's pinned extractor consumes a 1024-value decoder pose token and
`pred_cam_t`, not simply a DINO image embedding. Its configuration disables
hand-detection tokens for this feature path. That is **not** a requirement to
zero wrist rotations or remove hand articulation from GEM's motion output.
See [the exact integration source][gem-extractor].

Keep MHR model joints, MHR evaluation keypoints, SOMA joints and G1 actuator
coordinates explicitly distinct. Names, parents, rest transforms, units, axes,
rotation convention and camera/world spaces belong in versioned schemas.
Do not infer joint correspondence from matching array lengths.

Arm acceptance includes turns, reaching, raising/lowering arms and wrist
orientation. The existing upstream SONIC webcam bridge does not forward wrist
or finger estimates; that limitation must not silently enter a new adapter.
[Upstream documents the limitation][sonic-limitations]. Reconstructing a
box-lifting motion does not establish successful physical grasping: contacts,
force control, friction and robot reachability are separate evaluations.

## 2. Reference selection and reuse decisions

Use official Meta PyTorch as the numerical authority, not another C++ port.
Body's pinned revision matches GEM-X's submodule, and the published hashes of
GEM's Body checkpoint/MHR asset match Meta's release. This permits a common
starting artifact without assuming unrelated variants are interchangeable.

The selected Body configuration is `dinov3_vith16plus`, 512×512 input, with a
1024-dimensional, six-layer SAM decoder. The published configuration requests
BF16 mixed precision. That does not establish every checkpoint tensor's dtype;
the extraction inventory must do so. The first native numerical baseline is
F32, compared against a separately identified upstream F32 execution.

| Candidate | Decision for this project |
| --- | --- |
| [AmmarkoV/SAM3DBody-cpp][body-port] | Source reference for preprocessing, MHR decoding and export. Not our engine: major networks use ONNX Runtime; build/dependency and C API contracts differ. |
| [Asher-1/sam-3d-objects-ggml][objects-port] | Strong candidate for selective Objects graph reuse, including sparse stages and MoGe. Not accepted wholesale: raw-image workflow remains hybrid, mesh decoding/PBR incomplete, and no suitable flat C ABI. |
| Existing local GGML projects | Preferred infrastructure/operation references, subject to architecture-specific tests and attribution. Never treat their fixtures as SAM's oracle. |

These are compatibility and static-review findings, not claims of malicious
code or demonstrated exploits. Neither third-party C++ engine was configured,
built or executed. The [audit](../reference/AUDIT.md) explains the stopping points.

### Local reusable code

Sibling paths below are development references, not build/runtime dependencies.
Their audited revisions are in the source manifest. Copy only bounded,
reviewed portions and record the original repository, revision, files, license
and modifications in `NOTICE` when adoption happens.

| Source | Reuse candidates and qualification |
| --- | --- |
| [motion-bricks.cpp reference process](../../motion-bricks.cpp/reference/README.md), [implementation plan](../../motion-bricks.cpp/docs/IMPLEMENTATION.md) | Source/hash preflight, safe extraction, actual-upstream captures, repeated black-box baselines, stage and final-output checks. Historical status/tolerances are not SAM acceptance criteria. |
| [motion-bricks.cpp API](../../motion-bricks.cpp/docs/API.md), [error boundary](../../motion-bricks.cpp/src/error.hpp) | Opaque handles, independent inference/controller levels, fixed caller error buffers, exception containment, backend selection. |
| [SkinTokens GGML preparation](../../skin-tokens.cpp/cmake/PrepareGGML.cmake) | Pristine upstream submodule plus fingerprinted build-copy patching; adapt path validation and support the no-patch case. Do not copy unrestricted recursive-deletion targets. |
| [TRELLIS implementation](../../trellis2cpp/trellis2.cpp), [DINO test](../../trellis2cpp/tests/test_dino.cpp), [preprocessing test](../../trellis2cpp/tests/test_preprocess.cpp) | DINOv3 attention/RoPE/LayerScale, tensor taps and sparse-transformer patterns. Its ViT-L/16 is not Body H+ or GEM's custom ViTPose H; dimensions, FFN, positions and tokens must be ported from the selected model. |
| [Depth Anything backbone](../../depth-anything.cpp/src/dino_backbone.cpp), [ViT blocks](../../depth-anything.cpp/src/vit_block.cpp), [preprocessing](../../depth-anything.cpp/src/preprocess.cpp) | Patch layout, attention, position interpolation and image math. Exclude DA-specific camera tokens and multiview logic unless the target genuinely uses them. |
| [FreeSplatter image code](../../free-splatter.cpp/src/image.cpp), [pose code](../../free-splatter.cpp/src/pose.cpp), [tap comparison](../../free-splatter.cpp/scripts/compare_taps.py) | Camera/image utilities, Gaussian representation and inspection patterns; verify coordinate conventions and attribute activation/order before reuse. |

Use the [SkinTokens parity postmortem](../../SkinTokens-parity-postmortem.md) as a
methodological constraint: agreement with a rewritten oracle can reproduce the
same layout bug twice. Final-output checks cannot be replaced by decoder checks
that inject official intermediate tensors.

## 3. Establish the executable reference safely

The present source inventory is the first part of this phase. Reference execution
is a subsequent task, and is not yet complete.

1. **Freeze the complete dependency closure.** Pin source/submodule commits,
   container base digest, Python packages and wheels, model/config revisions,
   auxiliary DINO/MoGe assets, MHR/SOMA data and example input hashes. GEM's
   ViTPose dynamically fetches DINO code through `torch.hub`; replace that fetch
   with an explicitly pinned local source dependency, documenting the change.
2. **Download deliberately.** Use `hf download --revision <recorded revision>`
   for selected official files after required access/terms are resolved. Do not
   fetch entire GEM repositories containing unrelated SMPL/data/ONNX artifacts.
   Verify bytes against the published SHA-256 before loading; a model-card size
   is not a tensor inventory. The manifest currently distinguishes published
   hashes from locally verified bytes.
3. **Isolate legacy loading.** Official Body calls `torch.load` with
   `weights_only=False`; normal host conversion must never do that. Run reviewed
   official loaders and any necessary legacy/TorchScript extraction in a pinned
   reference container, non-root, with read-only source/weights, a dedicated
   writable output directory, resource limits, no runtime network, no secrets,
   no Docker socket and no privileged mode. Limit GPU exposure to the selected
   device. GPU-enabled containers are not a perfect security boundary; use a
   disposable stronger environment if trust cannot be established.
4. **Preflight before inference.** Check exact source state, package versions,
   asset hashes, config resolution and GPU visibility. Report unresolved or
   missing weights, unexpected state-dict keys and unsupported operations. Never
   substitute a fallback model/feature vector and call the run a success.
5. **Run the official public entry point.** Use the official example first,
   then a pinned single-person GEM video. Save complete uninstrumented results
   from three fresh processes before adding tensor captures. Record the actual
   precision, TF32/autocast settings, device and random streams.
6. **Validate instrumentation.** Hooks must leave public outputs unchanged.
   Staged loading, altered precision and GEM's token-exposure wrapper each need
   a comparison against their named unmodified baseline. A helper that calls
   upstream layers but reconstructs the pipeline is not automatically an oracle.
7. **Extract safe artifacts.** Produce safetensors plus data-only JSON with
   names, shapes, dtypes, counts, layouts and hashes. Inventory learned tensors,
   inference constants and geometry assets separately. Exclude training state.
   A restricted pickle allowlist is an extra check, not a security sandbox.

No third-party quickstart/setup scripts, build-time download helpers or
converted weights are needed to establish the official oracle. Reconsider
executing a candidate engine only for a concrete unanswered question after
source review, in a fresh restricted environment with downloads disabled.

## 4. Conversion, runtime and packaging

The ordinary converter accepts only verified safetensors and schema-checked
JSON. Reject unknown architecture variants, missing/duplicate tensors,
unexpected tensor shapes, non-finite constants and incompatible component IDs.
Each GGUF records architecture/schema version, source and config hashes,
original precision, converted precision, preprocessing, tensor layouts and
required companion identities. Geometry assets must be checked data, not
executable TorchScript at native runtime.

Plan separate Body and Objects model/session components with a small shared
runtime. Loading Body must not allocate Objects weights. GEM can retain just
its feature sequence and unload Body before temporal inference. Avoid embedding
two incompatible GGML copies when linking sam3d and gem-x into one application;
offer a supported shared GGML target/version contract.

Use a publicly fetchable upstream GGML **git submodule**, with reviewed patches
under `patches/ggml/` applied to a build-directory copy. Pin the version when the
implementation starts; neither an old local pin nor a candidate's patched
version is automatically the right choice. Verify clean clones and source
archives; never require unpublished submodule commits. Patching is incremental
and must not regenerate heavyweight dependencies on each normal rebuild.

Generic Linux CMake configure/build/install instructions come first. Nix is an
optional development environment. Provide debug, release, ASan/UBSan and
libFuzzer presets, CPU and dynamic Vulkan backends, installable headers/libraries
and CMake package metadata. No implicit model or Python-environment downloads
in the build or library. Runtime paths, devices, threads and memory budgets are
explicit options, never personal machine defaults.

## 5. API levels

Names are proposed contracts, not implemented declarations. Use opaque
`s3d_runtime_options`, `s3d_body_model`, `s3d_body_request`, `s3d_body_result`,
and corresponding Objects handles. Constructors, setters, getters and free
functions carry fixed-width scalar values and pointers/counts; no public
struct layout, STL object or C++ exception crosses the C ABI.

| Level | Responsibility | Consumer controls |
| --- | --- | --- |
| Body feature inference | Explicit normalized crops/conditioning → image features, pose token, camera outputs | Preprocessing/tracker; GEM-compatible mode and feature schema |
| Body image inference | RGB bytes + box + intrinsics + optional prompts → predictions and optional MHR geometry | Person selection, refinement mode, downstream animation/controller |
| Objects component inference | Explicit conditions/noise → chosen flow/decoder results | Scheduling experiments and diagnostic boundary injection |
| Objects image pipeline | Image/mask + options → raw Gaussians/mesh once native preprocessing is complete | Segmentation provider, export/postprocessing choices |
| Demo/application layer | Web demo with uploads, gallery and 3D visualization; frame-wise offline/live video with capped inference and display interpolation | Separate executable, optional to build/run for library consumers; not part of inference dependency closure; GEM temporal integration remains separate |

Expose semantic stage interfaces rather than arbitrary internal GGML pointers.
GEM must be able to consume Body features without decoding/rendering a full
mesh. Image buffers need width, height, row stride, pixel format and explicit
capacity. Results provide named tensor descriptors via getters (dtype, rank,
dimensions, element count and read-only data), coordinate metadata and
capability flags; no diagnostic directory is required as an application input.

Every fallible call returns a fixed-width status and accepts a caller-owned
error buffer/capacity. Success clears it; failure truncates safely and
NUL-terminates when capacity is nonzero. No global `last_error`. Results own
their storage; borrowed views survive until that result is freed. Setters copy
input or specify a clearly bounded borrowed lifetime. Null/free, overflow,
allocation failure, cancellation and concurrent-call rules must be documented
and tested. Valid live pointers and truthful capacities remain caller duties.

## 6. Implementation milestones and acceptance

Each row defines an acceptance gate, not an assertion that it has passed or an
immediate task assignment. The supported Body branch has scoped evidence for
reference, inference and demo work; B4 hand refinement, G1 and Objects are future
extensions. Consult the roadmap for remaining numerical/release work. Body/GEM
and Objects retain separate manifests and progress reporting.

Layer-by-layer parity is the default development method for **every** component,
not just an optional debugging aid. Establish an actual-upstream capture, compare
CPU F32 and Vulkan at each layer/operation boundary, and resolve the first
divergence before accepting the next boundary. Apply the same method to
preprocessing, sampling, geometry and export even where the steps are not neural
layers. Final numerical and browser end-to-end tests remain additional gates.

| Gate | Implement/capture | Required evidence before advancing |
| --- | --- | --- |
| R0 | Locked upstream environment, official examples, safe tensor/asset inventory | Three black-box repeats; instrumentation equivalence; no unresolved fallback |
| B1 | Body crop/normalization, DINOv3 H+ embeddings and blocks | Exact crop metadata; pixel/tensor comparisons against actual upstream on CPU and Vulkan |
| B2 | Prompt/CLiFF conditioning, decoder, MHR/camera heads, GEM feature mode | Per-operation taps, token order and 1024-value pose-token/camera parity, including null prompts |
| B3 | Native MHR decode, skeletal transforms, shape/scale/pose correctives and skinning | All selected keypoints and final deformed vertices, camera projections and coordinate transforms match; not merely pose parameters |
| B4 | Standalone optional hand refinement and prompts | Official refined/unrefined cases, both wrists and hand geometry; detector/FOV outside scope unless separately ported |
| D1 | Body web demo and official-example visual comparison | Real upload → native inference → rendered body → download/history workflow passes headless Chrome QA and the numerical Body gates |
| G1 | gem-x consumes native Body + native ViTPose features | Actual upstream GEM feature/temporal/SOMA outputs over a whole pinned clip; no injected official features in final acceptance |
| O1 | Objects image/mask/pointmap conditioning; DINOv2/PointPatch and MoGe as selected by config | Preprocessing and all conditioner outputs; supplied-pointmap and raw-image paths labeled separately |
| O2 | Sparse-structure flow, solver/CFG, decoder and pose/scale decoding | Stored-noise step/trajectory tests; occupancy decisions, support ordering and transforms verified |
| O3 | Structured-latent flow and Gaussian decoder | Native O2 coordinates feed O3; full trajectories and all raw Gaussian attributes match |
| O4 | Native mesh decoder and export | Topology, indices, vertices and attributes; export round-trip, axis/winding/material tests |
| O5 | Optional official-equivalent mesh repair, UVs and texture baking | Separate postprocessing/render acceptance; raw mesh export is not textured-PBR parity |
| D2 | Objects demo mode and official-example visual comparison | Real image/mask upload → native inference → rendered object → export/history passes headless Chrome QA; Gaussian, raw-mesh and textured modes only advertised after their respective gates |
| P1 | Optimized upstream/native CPU performance comparison for Body | Correctness-preserving, matched-workload CPU performance parity and reproducible profiling report |
| P2 | Optimized upstream CUDA versus native Vulkan for Body | Correctness-preserving performance parity on the same NVIDIA GPU, plus full demo regression QA |

G1 is integration work in gem-x.cpp, not a requirement to embed GEM in this
library. GEM's ViTPose uses a custom DINOv3 network and a SOMA77 heatmap head;
the shared backbone machinery needs a separate architecture identity and tests.
Camera-motion recovery and retargeting are subsequent explicit components.

### What “parity” must mean

- Capture real upstream outputs at preprocessing, projections, positional
  encoding, Q/K/V head splitting, normalization, attention, residual/FFN,
  decoding and final geometry. Record shapes and axis meanings, not just names.
- For floating tensors report finite checks, maximum absolute error and
  relative L2 with a defined zero-reference denominator; compare every required
  element. Freeze per-boundary tolerances after baseline repeatability study,
  before native tuning. Do not invent one tolerance covering pixels, logits,
  rotations and world metres. Production-precision and strict-F32 oracles are
  separate; F32 conversion does not restore precision absent in source weights.
- Global Euler radians have a separately frozen, reference-supported policy:
  retain maximum absolute error on the raw angles and report their raw relative
  L2, but gate relative L2 on each coordinate's `(sin(angle), cos(angle)) pair.
  This has a nonzero reference norm at identity. Original CPU/CUDA controls
  demonstrate that raw-angle relative error alone can reject ordinary F32
  elementary-function rounding near zero. This does not permit angle wrapping,
  axis/unit changes or loosening limits for other tensors. Evidence and negative
  tests are in `reference/rotation-policy-v1.json` and `tests/test_rotation_parity.py`.
- Compare discrete crop/index/permutation/topology decisions exactly when
  deterministic. Inspect threshold margins and near ties. A mismatch that
  changes sparse support is an investigation, not permission to feed reference
  coordinates into the supposedly end-to-end native result.
- Capture initial noise and all stochastic inputs. Matching seeds across
  frameworks is insufficient. If stochastic equivalence is needed, use shared
  draws for arithmetic tests and repeated distribution tests for the sampler.
- The final acceptance starts with the same image/mask or clip, and uses each
  pipeline's **own** intermediate outputs. Diagnostic cross-injection narrows
  faults but never substitutes for this test.
- Compare final MHR vertices/keypoints/cameras and GEM trajectories, including
  global root placement, local/global rotation geodesic error, left/right
  wrists and temporal continuity. For Objects compare raw Gaussian parameters,
  sparse coordinates, object transforms and mesh results, not only a render.
- QA an official exemplar first, then a small difficult suite: occluded wrists,
  crossing arms, turning, non-square crops, image borders, tiny/empty masks and
  disconnected object support. Browser overlays are supplementary evidence.

## 7. Web demo and final visual end-to-end QA

The web demo is a **required project deliverable**, similar to the previous
GGML ports, while remaining optional for library users. Deliver Body mode at
D1. Objects mode at D2 is deferred to a later goal; Objects-specific controls
and QA below describe that future extension, not the current demo deliverable.
Do not make Body's demo wait for Objects or GEM video integration.

### Demo workflow

- Reuse reviewed Go-server/Three.js UI patterns from kimodo.cpp, skin-tokens.cpp
  and trellis2cpp. Use LocalAI styling and the actual logo at the top right,
  a full-window viewer and a scrollable controls/history sidebar. Check asset
  licenses/attribution when copying; do not depend on sibling checkouts at runtime.
- Upload an image and preview exactly what will be processed. Body mode allows
  explicit person-box selection and camera inputs; Objects mode allows mask
  upload/editing. Show the selected model, actual input, settings and supported
  refinement options. Automatic detection/segmentation is not silently required.
- Use the public native API through one bounded inference worker. Display
  queued/running stage, elapsed time, completion, cancellation and actionable
  errors. Failed jobs must not leave an old result looking like the new output.
- Provide orbit/pan/zoom/reset controls, sensible front-oblique framing, body
  mesh/skeleton toggles and supported object mesh/Gaussian views. Offer source
  image projection overlays and synchronized upstream/native comparison views.
  No motion playback controls for a static reconstruction.
- Persist input/settings/result history; selecting a prior result restores its
  inputs and settings for a new run. Provide explicit download buttons for
  supported geometry exports and metadata, and verify exported files can be
  reloaded. Store generated artifacts outside source control.
- Parameterize bind address, port, model/backend and data directory. Default to
  loopback; remote access is an explicit deployment choice. Bound upload size,
  decoded image dimensions, queue depth and storage; reject invalid inputs.

### Official-example comparison

Choose at least one reproducible **Body** example first and one **Objects**
example when that mode is delivered. Prefer original input images/masks and
visualizations shipped in the pinned upstream repositories/notebooks. Starting
points are Body's [human demo][body-human-demo] and
[dancing image][body-dancing-image], and Objects'
[single-object demo][objects-single-demo]. These are candidate fixtures, not
already captured or tested results. Record the selected image/mask bytes and
hashes, upstream commit/checkpoint/config, camera/crop, prompts, refinement,
seed/noise and reference outputs in a fixture manifest; check image reuse terms.

Use the [Meta gallery][meta-gallery] or a clearly identified figure from the
official paper for the presentation target: source photograph beside a 3D
reconstruction, with a comparable view and appearance. Match camera/projection,
framing, axes, lighting, background and render mode where known. Do not silently
fit or recenter each output independently in a way that hides root/scale errors.
Compare the body silhouette, pose, wrists and surface, or the object's silhouette,
shape and supported appearance. Capture a fixed front view and side/turntable
views where reference geometry is available, not just a flattering screenshot.

A website/paper image alone is a **qualitative** reference: its checkpoint,
camera or postprocessing may be unknown. If its original input/settings are
unavailable, use a repository example run through the pinned official pipeline
as the reproducible oracle and retain the gallery/paper comparison as context.
Do not make CI depend on the live Meta service or claim pixel parity with an
unversioned screenshot. Exact fixture selection and gallery-view inspection
remain implementation tasks; no gallery result has been reproduced yet.

For the decisive test, upload the original image through our actual demo UI,
set the recorded box/mask/camera/options, run native inference, load the generated
asset and render it. No mock inference, injected reference features or prebuilt
mesh may satisfy this gate. Associate the displayed/downloaded artifact hash
with the job and its numerical comparison report. Render official and native
geometry through the same controlled viewer as an additional comparison to
separate reconstruction errors from lighting/renderer differences. Upstream
reference assets are labeled and cannot be mistaken for native output.

### Mandatory headless Chrome QA

Automate the real demo with headless Chrome/Chromium (for example Playwright),
starting a fresh server with an isolated test data directory and known build.
Exercise upload, selection/settings, generate/progress, completion, camera and
overlay controls, export/reload, history selection and persistence after reload.
Test desktop and narrow layouts, sidebar scrolling, malformed/oversized inputs,
empty masks, unavailable models, failed/cancelled jobs and worker recovery.

Fail on uncaught page exceptions, unexpected console/network errors, missing
assets, non-finite geometry/camera transforms, blank or clipped results and
stale input/result associations. Wait for explicit job/render readiness, not
arbitrary sleeps. Capture screenshots of key states, a synchronized reference
comparison and browser/server logs; visually inspect the actual captured images
as well as automated assertions. A server health check or successful HTTP
response is not browser QA. Record browser version, viewport, device-pixel ratio
and WebGL renderer; software WebGL may test presentation but does not substitute
for native Vulkan inference coverage.

Save an end-to-end report linking the upstream input, native job, CPU/Vulkan
numerical results, downloaded artifacts, screenshots and any explained visual
differences. Fix unexplained differences before marking D1/D2 complete.
Deterministic same-renderer screenshot/silhouette checks use tolerances established
from repeated reference renders; cross-renderer/gallery similarity additionally
requires visual review. Neither visually similar output nor numerical model
parity alone passes the combined demo gate. Lightweight mocked UI tests may run
without weights, but release acceptance requires the real model-backed browser
workflow on CPU and Vulkan for each shipped mode.

## 8. Safety, performance and release gates

Normal tests must not download weights. Add synthetic operation/layout tests,
converter and metadata rejection tests, image/geometry bounds tests, C ABI
ownership/error tests and C-only/PureGo smoke consumers. Fuzz request setters,
tensor descriptors and image/geometry parsers with ASan/UBSan; do not fuzz GGUF
loading. A model-backed fuzzer may load one verified model before fuzzing inputs.
Use sanitizer builds for development whenever they fit memory; disabling them
for performance must be an explicit recorded choice. CPU sanitizers do not
validate GPU shader memory accesses, so run Vulkan validation/debug checks too.

CPU and Vulkan model-backed parity are mandatory release checks, not optional
quality claims. Missing hardware/assets may explicitly skip a developer run,
but cannot produce a successful release report. Test fresh generic-Linux
checkouts, backend selection, unavailable-device errors and parameterized paths.

Profile load, decode/crop, host↔device transfers, per-block compute, waits,
sparse indexing, geometry decode and export with wall-clock and device timings.
Record peak RAM/VRAM, tensor dtypes and backend/driver. Do not infer a stall from
low GPU utilization alone. Keep weights/activations resident within a stage,
batch compatible crops/frames and stage large models sequentially. Bound caches
and queues, expose progress/cancellation, and never silently change precision,
input resolution or model variant to fit memory. Quantization follows F32
acceptance and has separate quality/performance gates.

### Performance parity is a completion requirement

After Body and its real browser workflow work correctly, profile the official
Body PyTorch pipeline and the native Body pipeline. P1/P2 are required parts of
the goal, not optional benchmark follow-ups. Compare GGML CPU with PyTorch CPU,
and GGML Vulkan with PyTorch CUDA on the same NVIDIA GPU on the test system.
Record the discovered hardware/driver in benchmark artifacts; do not hardcode
this machine into application defaults.

Use a well-configured upstream baseline: evaluation/inference mode, sensible
CPU thread settings, supported optimized attention and compilation where it
helps (for example `torch.compile` with appropriate shape specialization).
Measure eager and compiled candidates, including graph breaks/recompilation,
and use the best validated configuration for the primary comparison. Validate
that compilation/precision changes preserve the reference outputs; do not
compare against an intentionally slow or incorrect upstream configuration.
If an upstream stage has no supported CPU implementation, document and resolve
that reference gap rather than presenting CUDA timings as CPU evidence.

Freeze the official-image benchmark suite and workload contracts before native
performance tuning. Match model/checkpoint, image resolution, batch size,
prompts/masks/camera, refinement, steps/noise, output representation and accepted
numerical quality. Report strict-F32 comparisons separately from validated
production-precision comparisons; lower precision is not a free parity claim.
Both sides may use equivalent memory-bounded scheduling, with the actual
residency/caching policy stated explicitly.

Profile stage wall time, CPU/GPU compute, transfers, synchronization/locking,
allocations, preprocessing, sparse support construction, decoding and export.
Synchronize asynchronous work at measurement boundaries. Report model load and
first-run/compilation cost separately from warmed inference, and additionally
measure full request-to-artifact latency without excluding the slow stages on
only one side. Browser transport/render time is a separate end-to-end metric.
Record peak RAM/VRAM and repeated-run median, tail latency and throughput.

The performance target is native median complete-pipeline latency no higher
than the optimized upstream baseline for each model/backend at the declared
workload, without worse sustained throughput or unexplained tail regressions.
Define repetition counts, noise bands and statistical decision rules from
baseline measurements before tuning; overlap/insufficient evidence is not a
speedup claim. Run on otherwise idle hardware with no competing inference
jobs. Do not satisfy the goal by dropping outputs/stages, reducing quality or
timing only favorable transformer kernels. Re-run numerical parity and the
model-backed headless Chrome suite after optimizations, and publish failures
or remaining gaps explicitly rather than marking P1/P2 complete.

Source repositories contain no checkpoints, generated meshes/videos, caches,
private paths or machine-specific configuration. Reviewed fixtures can be
small committed data; large artifacts use hash-addressed external storage.
Publication requires a source/model/asset license inventory first: Meta's
[SAM License][sam-license] applies to SAM materials, and a third-party wrapper's
MIT label does not relicense the model. Record attribution for copied code and
get any unresolved redistribution conditions cleared before uploading GGUFs.
If publishing through our HF organization, use `LocalAI-io`, upstream-linked
model cards and a generic download workflow. No HF repository is authorized or
created by this design task.

## 9. Current work

Maintain the current task list in [ROADMAP.md](ROADMAP.md), not here. It separates
completed Body capabilities, remaining release actions, known numerical limits
and future features. Detailed verification remains in the component reference
reports and [development history](../reference/HISTORY.md).

[gem-extractor]: https://github.com/NVlabs/GEM-X/blob/32992550dba114c62243fb55e361311972dce8f9/gem/utils/sam3db_extractor.py
[sonic-limitations]: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/live_camera_teleop.html#limitations
[body-port]: https://github.com/AmmarkoV/SAM3DBody-cpp/tree/198002ede17892648d27d5c0a0085860a9e63bcd
[objects-port]: https://github.com/Asher-1/sam-3d-objects-ggml/tree/1c14b7c3c3e8d9109b943ddc83a0a39c73744246/cpp_ggml
[sam-license]: https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/LICENSE
[meta-gallery]: https://aidemos.meta.com/segment-anything
[body-human-demo]: https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/notebook/demo_human.ipynb
[body-dancing-image]: https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/notebook/images/dancing.jpg
[objects-single-demo]: https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/notebook/demo_single_object.ipynb
