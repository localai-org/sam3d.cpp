# sam3d.cpp

A C++23/GGML port of Meta's SAM 3D Body and an experimental SAM 3D Objects
runtime. Native inference runs on CPU or Vulkan without Python, PyTorch, CUDA
or llama.cpp.

- One person's image and bounding box → MHR mesh, joints, pose and camera.
- One scene image and painted object mask → vertex-coloured geometry GLB.
- Opaque C API for inference and lower-level image/camera preparation.
- Optional Go/WebGL demo with photo uploads, history, static GLB/OBJ export,
  offline video, live webcam recording and skeleton animation GLB export.

Video uses independent frame estimates, not a learned temporal model. Detailed
hand refinement remains outside the current supported scope. Objects now has
an accepted raw-geometry baseline; mesh repair and baked PBR textures remain
outside its current scope.

## Build on Linux

Install a C++23 compiler (GCC or Clang), CMake 3.24+ and Ninja. From the source root:

```sh
git submodule update --init --recursive
cmake --preset debug
cmake --build --preset debug -j2
ctest --preset debug
cmake --install build/debug --prefix /your/install/prefix
```

Debug enables ASan/UBSan. Use the `release` preset for optimized CPU deployment,
or `optimized-sanitizers` for optimized CPU correctness checks. No weights,
Python packages or network access are needed for the native build/tests after
fetching GGML's pinned public submodule.

For Vulkan, also install a Vulkan SDK (loader, headers, `glslc` and SPIR-V
headers). The optimized correctness build retains UBSan:

```sh
cmake --preset vulkan-optimized
cmake --build --preset vulkan-optimized -j2
ctest --preset vulkan-optimized
```

On supported NVIDIA GPUs, `vulkan-bf16-production` enables the validated BF16
encoder optimizations and disables sanitizers for deployment. It applies bundled
GGML patches to a build-directory copy, leaving the public submodule unchanged.
The decoder and geometry remain F32. See the [demo configuration](demo/README.md#bf16-image-encoding-on-nvidia-vulkan)
for the required precision/backend flags and the
[development guide](docs/DEVELOPMENT.md) for sanitizer and fuzzing builds.

GGML backends are dynamically loaded and selected explicitly; a failed Vulkan
request is not silently run on CPU. Nonstandard SDK paths use standard CMake
package variables. Nix is optional; no Nix-specific setup is required or bundled.

## Models and demo

The Body runtime needs three compatible local files:

- `body-dinov3-f32.gguf` — image encoder.
- `body-pose-branch-f32.gguf` — decoder and pose heads.
- `mhr-lod1-f32.gguf` — MHR geometry.

The same archives support the runtime BF16 encoder mode. These are F32/F32-I32
format conversions, not low-bit quants. No weights are checked into this repository.

Objects uses converted MoGe, SS generator/decoder, SLat generator and mesh
decoder GGUFs under `generated/models/objects-gguf`. The demo accepts a scene
image and exact painted mask, runs this native path, renders the indexed
FlexiCubes mesh and preserves the same vertex-coloured GLB for download.

The [model card and uploader](distribution/README.md) are prepared, but converted
HF downloads have not been published yet. Until publication, see the
[reference setup](reference/README.md) and [conversion guide](reference/GGUF.md)
for official model access and safe conversion.

Follow the [demo build and launch guide](demo/README.md) for the optional Go
server. It uses the native C API and does not need Python for inference.
Remote webcam access requires HTTPS. The server has no authentication and must
only be exposed through a trusted deployment.

## Library API

Use [sam3d_model.h](include/sam3d_model.h) for model loading and body inference;
[sam3d.h](include/sam3d.h) exposes lower-level crop, image and camera operations.
Handles are opaque, with constructors/setters/getters and explicit ownership,
so callers do not need to reproduce C struct layouts. See the
[API guide](docs/API.md) for linking, examples, output conventions and limitations.

## Validation and performance

Tests compare individual operations/layers and complete outputs with pinned
official PyTorch references. The Body pose branch has passed 646 trained F32
Vulkan comparisons; six strict CPU geometry/projection comparisons remain open.
BF16 uses separate numerical policies, with a recorded non-blocking hand-logit
difference. Headless Chrome checks cover real upload, inference, rendering,
history and export against the official-image reference.

On the profiled NVIDIA GPU, warm BF16 native inference measured **85.8 ms**,
versus **81.6–84.5 ms** for compiled/eager upstream CUDA at the compared scope.
The live pipeline measured **8.1 Hz** and **176 ms** median first render
submission; that is not physical camera-to-display latency. These are measured
workloads, not general hardware or full hand-refined performance guarantees.

For Objects, the corrected Vulkan MoGe point map is within `1.16e-4` relative
L2 of the pinned PyTorch result, and matched first-step SS shape velocity is
within `1.25e-4`. The geometry decoder reproduces upstream FlexiCubes exactly
when supplied the same raw tensor. On the complete kids-room exemplar, native
and upstream meshes have better than 0.9999 silhouette IoU in three fixed views.
See [Objects runtime evidence](reference/OBJECTS_RUNTIME.md) and the
[geometry parity report](reference/OBJECTS_MESH_PARITY.md).

Optional [inference scheduling/crop optimizations](reference/BODY_FAST_INFERENCE.md)
follow **Timing Yang and the Fast SAM 3D Body researchers**. We credit their
contribution in [NOTICE](NOTICE); our native implementation and measurements
have their own numerical and performance limits.

See [precision evidence](reference/BODY_PRECISION.md),
[live profiling](reference/BODY_LIVE_PERFORMANCE.md) and the
[current roadmap](docs/ROADMAP.md). Detailed historical results are retained
in [reference/HISTORY.md](reference/HISTORY.md), not as an active checklist.

## Development and licensing

- [Development](docs/DEVELOPMENT.md): tests, fuzzing and optional Python tooling.
- [Design](docs/DESIGN.md): architecture and layer-by-layer acceptance process.
- [Roadmap](docs/ROADMAP.md): remaining release work, limitations and future features.
- [Reference](reference/README.md): reproducible upstream captures and conversion.

Original contributions use [Apache-2.0](LICENSE), with SAM/DINO and other
third-party exceptions. The weights are **not** Apache-relicensed. See
[licensing](docs/LICENSING.md), the consolidated [NOTICE](NOTICE) and
the individual license texts in [LICENSES](LICENSES).

## AI use

This project was produced largely automatically by GPT-6 Astra based on a playbook
we have been figuring out to quickly create PyTorch to GGML conversions and a small
demo app to showcase the model.

Because the complexity of these projects is bounded to performing a well defined
computation that can be fuzzed, I believe the risk to be within acceptable limits.
