# Library APIs

The library is usable without the demo, Python, Torch or a reference fixture
directory. It loads local GGUFs and an explicitly selected GGML backend module.
All public handles are opaque: constructors, getters and setters use fixed-width
scalars and pointer/count pairs. There are no public struct layouts, C++ types
or exceptions in the C ABI, so FFI consumers need not reproduce a struct layout.

The interface is experimental. Full Body refinement, Objects model inference,
cancellation, configurable graph memory budgets and feature-only/component
inference APIs remain unfinished. The existing CPU numerical failures are not
waived by exposing the model through an API.

## Choose a level

| API | Input → output | Status |
| --- | --- | --- |
| Body crop/image/camera, `sam3d.h` | Box/RGB/intrinsics → crop geometry, normalized image, camera conditions | Implemented; no neural inference |
| Objects image preparation, `sam3d.h` | RGBA/options → original image/mask tensors | Implemented; no 3D reconstruction |
| Body model, `sam3d_model.h` | RGB + person box + intrinsics + three GGUFs → body mesh, joints, pose and camera | Implemented one-person, no-mask **body pose branch only** |
| Body feature inference | Normalized crop/conditioning → GEM-compatible features without mesh decoding | Planned |
| Objects component/pipeline inference | Conditions/noise or image/mask → geometry | Planned |
| Demo/export | Single-image upload/history/visualization/static GLB or OBJ download | Optional Go server and native C-API CLI; see [demo guide](../demo/README.md) |

## Build and link on Linux

Follow the root README for dependencies and CMake build options. A release
installation can be consumed from C or C++:

```sh
cmake --preset release
cmake --build --preset release -j
cmake --install build/release --prefix /path/to/install
```

```cmake
find_package(sam3d CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE sam3d::sam3d)
```

Supply `/path/to/install` in the consumer's `CMAKE_PREFIX_PATH`. No model is
downloaded by configure, build, install or inference. Headers and GGML dynamic
backend modules are installed with the libraries. For sanitizer builds, the
consumer must link the same sanitizer runtimes; normal applications should use
a release library. Nix is optional, not an application runtime dependency.

## Body model lifecycle

1. Create `s3d_runtime_options`. Select CPU or Vulkan, the explicit backend module
   path, device index and CPU thread count. Optionally require an exact device
   description. Set the backbone, pose-branch and MHR GGUF paths individually.
2. Load `s3d_body_model`. It validates the pinned component contracts, companion
   identities, hand-index partitions and shared mesh topology. Options can now
   be freed. Files and backend modules must remain trusted and immutable;
   metadata validation does not authenticate arbitrary externally supplied bytes.
3. Create `s3d_body_request`. Set RGB U8 pixels with byte capacity and row stride;
   pixels are copied into owned, tightly packed storage. Set an XYXY person box
   and `fx,fy,cx,cy` intrinsics in original-image pixels. The caller supplies
   person selection and camera calibration; neither is guessed by this API.
4. Call synchronous `s3d_body_model_infer`. The model runs preprocessing,
   backbone, six decoder/geometry feedback layers and hand-box prediction using
   its own intermediates. No tensors from a reference capture are required.
5. Enumerate result tensors with `get_count`, `get_tensor`, `get_dimension`.
   Discover semantics with `get_metadata`. Prefer names over numeric indices.
   The result owns its buffers independently of the model and request.
6. Free result, request and model handles when finished. Freeing NULL is safe.

Callers must supply valid live pointers and truthful buffer capacities. A freed
or forged handle is not a recoverable input. Setters preserve prior values on
failure. All fallible calls return `s3d_status` and accept a caller-owned error
buffer: success clears it, failure truncates and NUL-terminates it. `NULL,0`
discards diagnostics; `NULL,nonzero` is invalid. There is no global `last_error`.
Failure clears returned handles/views/counts. There is no cancellation yet.

One model serializes its inference calls. Separate model instances may execute
concurrently subject to device memory; their backend discovery is serialized.
Do not mutate or free a request/options object while a call reads it, or free
a model during inference. Immutable results support concurrent reads until freed.

Reuse a model handle across images to amortize loading. Vulkan models retain
the full transformer stack on device and reuse its graph/workspace; normal
inference omits unused parity readbacks. Decoder/head and MHR projection weights
are also cached as immutable snapshots with shared device storage.
The model owns these resources until freed, independently of requests/results.
It currently uses about 4.4 GiB of device memory; individual model instances do
not share their buffers. Decoder/geometry activation transfer and host-work optimization
is still incomplete. The demo's process-level cancellation and idle unloading
are wrappers, not new cancellation or eviction functions in the C API.

## Body inference scheduling and crop size

Configure inference before loading the model with
`s3d_runtime_options_set_body_inference(options, crop_size, intermediate_mask,
correctives, slim, error, capacity)`. Defaults are `(512, 31, 1, 0)`.
Crop size accepts 384, 448 or 512; mask bits 0–4 select intermediate body
predictions. Every transformer layer and the final prediction still execute.
Mask `7` selects predictions after layers 0, 1 and 2, with later feedback reusing
the latest prediction. Mask `0` skips all intermediate predictions and feedback.
The boolean flags control MHR pose correctives and omission of unused intermediate
outputs. The final result retains all 20 fields, including skeleton transforms.

The options and loaded-model getters expose the same four values. Invalid setters
leave the previous configuration intact. Changing options cannot mutate a loaded
model. These modes use the existing GGUFs and combine with either precision mode.

The CLI and persistent worker accept trailing `--body-crop-size=448`,
`--body-intermediates=0,1,2` (or `none`), `--no-body-correctives` and
`--slim-body-intermediates`. Intermediate indices must be unique and within 0–4.
Slim preserves the computed feedback; the other controls change estimates.
See [measured native differences](../reference/BODY_FAST_INFERENCE.md) before
choosing an approximate mode.

## Vulkan precision

F32 acceptance currently requires these process-wide GGML initialization flags:

```sh
export GGML_VK_DISABLE_F16=1
export GGML_VK_DISABLE_COOPMAT=1
export GGML_VK_DISABLE_COOPMAT2=1
```

Set them **before the first Vulkan backend initialization**. The library neither
changes environment variables nor silently switches to CPU. It rejects known
earlier non-strict initialization, or external Vulkan initialization whose
precision it cannot verify. Changing environment variables after initialization
does not repair it; use a fresh process. Select the backend/device explicitly rather than assuming device zero
is the desired GPU. No CUDA inference runtime is required.

Backbone precision is an explicit model option, defaulting to `S3D_BACKBONE_F32`:

```c
s3d_runtime_options_set_backbone_precision(options, S3D_BACKBONE_BF16,
                                         error, sizeof error);
```

Check the returned status, as for every setter. The options getter and
`s3d_body_model_get_backbone_precision` expose the configured/loaded mode. Invalid
values leave the previous setting unchanged; changing options never mutates an
already loaded model. No new public struct layout is involved.

BF16 matches
upstream's precision scope: only the DINO image encoder is rounded to BF16;
the pose decoder and MHR keep their existing precision, and results remain F32.
The same trusted F32 GGUFs work in both modes; weights are converted at loading,
with finite/overflow checks. Native GGML runs without Python or PyTorch.
The CLI and persistent worker accept a trailing `--bf16` flag.

The unpatched scalar Vulkan path uses the strict flags above, including for BF16
operands. The faster NVIDIA cooperative-matrix-2 path instead requires the
`vulkan-bf16-cm2` build-copy patch preset and these flags **before initialization**:

```sh
unset GGML_VK_DISABLE_COOPMAT GGML_VK_DISABLE_COOPMAT2
export GGML_VK_DISABLE_F16=1
export SAM3D_BF16_COOPMAT2=1
export SAM3D_BF16_FLASH_ATTENTION=1
export SAM3D_BF16_PRECISE_PREFIX=1
export GGML_VK_FUSE_BF16_ROUND=1
export GGML_VK_FUSE_BF16_BINARY=1
export GGML_VK_BF16_BINARY_LINEAR=1
export GGML_VK_BF16_MATMUL_TILE=small
export SAM3D_BATCHED_TRANSFERS=1
export SAM3D_SIMD_SKINNING=1
export GGML_VK_F32_NARROW_MATMUL=1
export GGML_VK_F32_NARROW_TILE=tiny32
export GGML_VK_FUSE_BF16_SILU_GATE=1
```

The patched backend preserves genuine F32 arithmetic for the decoder and
advertises a capability handshake; an incompatible module is rejected, not
silently substituted. The precise-prefix path keeps all image keys/values and
uses F32 math attention for the five class/register queries. This configuration
passes all encoder stages and final-body policies for two official images, with
one non-blocking dancer hand-logit discrepancy. It is not universal-image or
full hand-refined estimator acceptance. CPU BF16 uses math attention and needs
none of these Vulkan flags. The demo sets the appropriate environment in its
child worker when `--precision bf16` is selected. See
[precision benchmarks and remaining gates](../reference/BODY_PRECISION.md).

The last five settings select the measured runtime optimizations (linear
indexing, smaller BF16 tiles, bounded pinned transfers, exact CPU SIMD skinning
and a narrow F32 GEMM tile); they do not lower decoder precision or remove model
outputs. Use the corresponding current build-copy patches. Leave diagnostic
tracing unset for timing. The scalar fallbacks remain available.

## Result schema and coordinates

`get_metadata("schema")` returns `sam3d.body.pose_branch.v1`; the capability bit
is `S3D_CAP_BODY_POSE_BRANCH`, **not full Body/refined-hand support**.
Metadata keys also include `scope`, `geometry_units`, `coordinates` and
`joint_rotation_coordinates` and `joint_transform_coordinates`.

All tensors are contiguous row-major with the logical dimensions below. Batch
dimension is retained even for the sole supported sample. All are F32 except
`faces`, which is signed I32. Existing shapes and names are stable for this schema. The additive
`joint_transforms` field brings the current count to 20; enumerate by name rather
than assuming a fixed count or index. Rebuild strict CLI-format consumers with
the updated runner.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `vertices`, `joints`, `keypoints` | `[1,18439,3]`, `[1,127,3]`, `[1,70,3]` | Body coordinates, metres |
| `faces` | `[36874,3]` | Original zero-based vertex indices |
| `joint_transforms` | `[1,127,8]` | Original MHR global state: centimetre XYZ translation, XYZW quaternion, uniform scale |
| `joint_rotations` | `[1,127,3,3]` | Original MHR global matrices; **not** parent-local or axis-flipped |
| `vertices_pixels`, `keypoints_pixels` | `[1,18439,2]`, `[1,70,2]` | Original-image projected pixel coordinates |
| `camera_translation`, `camera_parameters` | `[1,3]` each | Metric translation; original raw head camera vector respectively |
| `pose_raw`, `global_rotation`, `body_pose` | `[1,266]`, `[1,3]`, `[1,133]` | Original Body/MHR pose parameter conventions, not interchangeable rotation layouts |
| `shape`, `scale`, `hand`, `face` | `[1,45]`, `[1,28]`, `[1,108]`, `[1,72]` | Original coefficients; face output is disabled/zero in this mode |
| `mhr_model_parameters` | `[1,204]` | Original assembled MHR parameters |
| `hand_boxes`, `hand_logits` | `[1,2,4]`, `[1,2,2]` | Normalized crop-relative CXCYWH boxes and class logits, not refined hand meshes |

Body vertices/joints/keypoints are the original MHR geometry converted from
centimetres to metres and multiplied by `diag(1,-1,-1)`, matching upstream.
They do **not** include camera translation. Add `camera_translation` before
projecting with the supplied intrinsics. The original `joint_rotations` are
deliberately preserved as upstream returns them; do not attach them directly
as local rotations to the axis-flipped joint positions. A renderer/exporter
must handle coordinate conversion explicitly. No viewer-specific axis change,
normalization or recentering is hidden in the library output.

## Evidence and remaining limits

The pure-C consumer in `tests/body_api_capture.c` invokes the actual shared
library. It discards the caller's pixels/geometry before inference, then frees
the model/request before reading all results. The original real Vulkan validation returned
all 19 then-existing fields byte-identical to the accepted GGUF-only native capture, whose
646 trained upstream comparisons pass. Topology matches the verified original
safe state. See `scripts/run_body_api.py`, `scripts/check_body_api.py` and
[trained branch evidence](../reference/TRAINED_BODY_BRANCH.md).

Normal tests need no weights: they cover C input ownership, error paths, stride
and overflow bounds, result ownership/schema and malformed output parsing.
The request/options and result getter fuzzers run with ASan/UBSan/LeakSanitizer.
GGUF loading is intentionally excluded from fuzzing and instead has deterministic
rejection tests. Real Vulkan runs use UBSan because of the documented NVIDIA
ICD/ASan initialization conflict. CPU full-model geometry/projection gates,
hand refinement and Objects inference are still open. The single-image demo's
real Vulkan upload, render, export/reload and history flow now passes headless
Chrome QA against the original Body-branch example; see [demo evidence](../reference/BODY_DEMO.md).
