# sam3d.cpp

A C++/GGML port of Meta's SAM 3D Body: recover human pose and
body geometry from photographs. SAM 3D Objects is deferred to a later phase;
its existing experimental components are retained.

Original code is Apache-2.0, with SAM/DINO and other third-party exceptions:
see [licensing](LICENSING.md). The models are **not** Apache-relicensed.

The [body web demo](demo/README.md) provides
photo upload, person-box selection, mesh/skeleton viewing, original-upstream
comparison, saved history and static GLB/OBJ downloads. It also supports offline
video sequences and live webcams with a configurable inference-rate cap and
interpolated playback. Video uses independent single-person body estimates,
not a temporal model. Remote webcams require HTTPS.
The trained F32 Body
image-to-mesh branch now runs through an opaque public C API:
646 Vulkan comparisons pass, while six strict CPU geometry/projection checks
remain open. The API includes preprocessing and a GGUF-only body pose-branch
session. Detailed hand refinement and performance optimization
remain unfinished. Follow [STATUS.md](STATUS.md),
[TODO.md](TODO.md) and the full [design](DESIGN.md).

## AI use

This project was produced largely automatically by GPT-6 Astra based on a playbook
we have been figuring out to quickly create PyTorch to GGML conversions and a small
demo app to showcase the model.

Because the complexity of these projects is bounded to performing a well defined
computation that can be fuzzed, I believe the risk to be within acceptable limits.

## Build on Linux

Install a C++23 compiler (GCC or Clang), CMake 3.24+ and Ninja. From this directory:

```sh
git submodule update --init --recursive
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
cmake --install build/debug --prefix /your/install/prefix
```

Debug enables ASan/UBSan by default. For larger CPU verification runs,
`optimized-sanitizers` uses `-O2` with debug symbols, assertions and both
sanitizers retained, in a separate build directory:

```sh
cmake --preset optimized-sanitizers
cmake --build --preset optimized-sanitizers
ctest --preset optimized-sanitizers
```

`release` disables sanitizers for later performance measurements.
Tests must run outside ptrace-based sandboxes for LeakSanitizer;
do not disable sanitizers to hide a sandbox limitation.

Full-model verification should run one job at a time with a hard RAM cap and
headroom for other applications. The optional Linux/systemd
[bounded runner](reference/MEMORY_SAFETY.md) enforces this for native jobs;
Docker references require their own container memory caps.

To fuzz the current C API with Clang and libFuzzer:

```sh
cmake --preset fuzz
cmake --build --preset fuzz
./build/fuzz/bin/sam3d-crop-fuzz -runs=100000 -max_len=128
./build/fuzz/bin/sam3d-objects-image-fuzz -runs=100000 -max_len=128
./build/fuzz/bin/sam3d-body-model-fuzz -runs=100000 -max_len=512
./build/fuzz/bin/sam3d-body-result-fuzz -runs=100000 -max_len=512
./build/fuzz/bin/sam3d-hand-crop-fuzz -runs=100000 -max_len=256
./build/fuzz/bin/sam3d-hand-frame-fuzz -runs=100000 -max_len=512
```

No weights, Python runtime, network access or sibling projects are needed to
build/run the current native tests after fetching the public GGML submodule.
GGML 0.23.0 is pinned to a publicly fetchable commit, without private patches.
The CPU backend is built as a dynamically loaded module. To also build Vulkan,
install a Vulkan SDK (loader, headers, `glslc` and SPIR-V headers), then configure
with `cmake --preset debug -DSAM3D_VULKAN=ON`. Nonstandard SDK locations can be
provided through standard CMake package paths; none are hardcoded in the project.
Diagnostics accept an explicit trusted backend module and device index and
never silently substitute CPU for a requested Vulkan device.
Nix development tooling is optional and has not been added yet.

Some NVIDIA drivers fail Vulkan ICD initialization when the ASan runtime is
loaded. Only for that diagnosed case, `vulkan-ubsan` provides a separate debug
build with UBSan retained and ASan disabled. Keep the ordinary `debug` build
for ASan/UBSan CPU validation; a successful AMD run is not NVIDIA validation.

For Vulkan F32 correctness testing, use the optimized, UBSan-enabled preset
(assertions and strict F32 retained; no fast-math):

```sh
cmake --preset vulkan-optimized
cmake --build --preset vulkan-optimized -j2
ctest --preset vulkan-optimized
```

The [measured optimization passes](reference/BODY_PERFORMANCE.md) reduce the
sample browser job from about 40 s to **2.67 s cold / 0.46 s warm**, including
exports. The demo reuses a bounded native model session; warm inference itself
takes about 0.395 s (393–404 ms measured). Mean warm GPU utilization is about
69%, not yet the 90% target or real-time/optimized-upstream performance parity. All 646 upstream
checks still pass with unchanged outputs.

A faster BF16-backbone mode is available through the
[C API](docs/API.md#vulkan-precision), native CLI (`--bf16`) and
[demo](demo/README.md#bf16-image-encoding-on-nvidia-vulkan) (`--precision bf16`).
The patched NVIDIA Vulkan path passes complete encoder-stage and final-body
policies on two official images, with a reported non-blocking hand-logit
discrepancy. The live demo should use `vulkan-bf16-production` with
`--precision bf16`; this optimized build disables sanitizers, while the separate
CPU/Vulkan correctness builds retain them. Warm native inference is **85.8 ms**;
upstream CUDA BF16 is **84.5/81.6 ms** (eager/compiled
backbone). These are body-only measurements, not full detector/refinement or
upload/export latency. Real BF16 browser upload/render/GLB-export QA passes,
including an exact comparison with the accepted native output. F32 remains the
default build/runtime option. The 80–85 ms performance goal is still open; see
[measurement details](reference/BODY_PRECISION.md) and the separate
[live pipeline profile](reference/BODY_LIVE_PERFORMANCE.md).

## Reference validation

For interactive image reconstruction or video tracking, follow the
[demo build and launch instructions](demo/README.md). The optional Go server
uses the native C API; neither Python nor a sibling project is needed for inference.

The [reference guide](reference/README.md) covers pinned official sources, safe
conversion and original-upstream captures. Tests progress from operation/layer
comparisons to composed paths using native-produced intermediates. Normal tests
include original fixtures and malformed-input checks without downloading weights.

Current CPU/Vulkan milestones include the complete 32-block DINOv3 backbone at
512×512, Body conditioning/decoder components, and the real public MHR geometry
asset converted to checked F32/I32 GGUF. The complete Body decoder feedback loop
also passes 526 original boundary/full-output checks across two/six-layer
trajectories. These use small-width **synthetic SAM state** and real MHR geometry:
they are not trained model or image-to-3D parity. See
[decoder validation](reference/BODY_FLOW.md), [MHR validation](reference/MHR.md)
and the [conversion guide](reference/GGUF.md) for precise boundaries and commands.

The full-width six-layer decoder also passes 389 checks per backend, with an
explicit reference-supported angular metric; raw errors remain recorded.
The full raw-image/backbone/decoder composition now also passes 435 checks per
backend on an official photograph, still with synthetic SAM state. See the
[image composition evidence](reference/BODY_IMAGE_FLOW.md).
The single-image [demo](demo/README.md) has real model-backed headless Chrome QA
and an original-upstream comparison. Remaining gates include full hand refinement,
the outstanding CPU numerical checks and optimized end-to-end performance comparison. Current
geometry/sampling run on CPU between GGML graphs; no performance parity is claimed.

Deferred Objects work: default RGBA image/mask preprocessing passes 114 original comparisons,
including an official photograph and mask. Its C API has ownership/error tests
and sanitizer fuzz coverage. [Details](reference/OBJECTS_IMAGE.md) explain the
boundary: full pointmap processing and object generation remain unfinished.
The point-window transformer separately passes 196 synthetic-state checks per
CPU/Vulkan backend, including its full default size; this is not trained model
acceptance. See [point-window validation](reference/OBJECTS_POINTPATCH.md).
Native pointmap normalization passes 475 comparisons against each original
CPU/CUDA reference; [normalization validation](reference/OBJECTS_SSI.md) documents
the CPU preprocessing scope and invalid-point behavior.
The [joint RGB/XYZ transform tests](reference/OBJECTS_JOINT.md) also cover soft
alpha and aligned cropping on the original full-resolution Objects photograph,
with explicitly synthetic supplied XYZ rather than an estimated pointmap.
The [composed preprocessing tests](reference/OBJECTS_PREPROCESS.md) connect
normalization, joint and full-image transforms, passing 536 original comparisons
without supplying reference intermediate tensors. Learned generation is still pending.
The [condition-fusion graph](reference/OBJECTS_FUSER.md) passes 140 synthetic
operation/output comparisons on each CPU/Vulkan backend; encoder integration and
trained checkpoint validation remain unfinished.
The [point-only conditioning composition](reference/OBJECTS_POINT_CONDITION.md)
now consumes native preprocessing outputs through PointPatch and fusion, with
small and full-size CPU/Vulkan reference checks. This is not yet the
published complete image/point conditioning pipeline.

Body and Objects weights require access approval through their official
Hugging Face repositories. Development access was granted on 2026-09-09;
normal conversion will consume verified safe intermediates, not load pickle on
the host. No converted model downloads are published by this project yet.
The [model card and publication tooling](distribution/README.md) are prepared
for a separate, explicitly authorized HF release; the uploader defaults to a
local checksum-verifying dry run.

The Body backbone, pose branch and MHR geometry are converted to checked GGUFs.
The internal RGB-to-body Vulkan path passes 646 trained comparisons; six small
CPU numerical failures remain. Hand refinement is still unfinished;
the image/video Body demo is available and trained Objects inference is deferred. See
[trained-model progress](reference/TRAINED_BODY_BRANCH.md).

## API and licensing

[sam3d.h](include/sam3d.h) exposes constructors, setters, getters and owned result
handles without public struct layouts. The implemented crop API computes a
box's center, expanded scales and original-image-to-crop affine transform.
The image API resamples RGB U8 into a black-bordered crop and emits normalized
planar CHW F32 data. The camera API takes that crop geometry, original-image size
and explicit `fx,fy,cx,cy` intrinsics, and exposes owned ray-grid/decoder-condition
results. It supports axis-aligned square crops up to 512×512; unsupported rotated
or rectangular camera crops are rejected. These APIs do not estimate a pose or
run a neural graph. [Pure-C tests](tests/test_body_camera.c) demonstrate ownership,
getters and error handling.

[sam3d_model.h](include/sam3d_model.h) adds the GGUF-only Body image inference
API, explicit CPU/Vulkan selection, copied requests and independently owned
mesh/pose results. See the [API guide](docs/API.md) for abstraction levels,
ownership, tensor shapes, coordinate conventions, installation and current
limitations. A pure-C shared-library run matches all 19 output tensors from the
accepted Vulkan capture byte-for-byte.

See [third-party notices](THIRD_PARTY_NOTICES.md) and the included
[SAM License](LICENSES/SAM.txt) for adapted Meta material. No third-party C++
inference engine has been built or executed during this work.
