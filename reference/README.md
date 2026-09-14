# Reference setup

This directory contains pinned upstream capture/conversion procedures and
numerical evidence. The supported Body pose branch has runnable original/native
comparisons; this is not a claim of complete hand-refined or Objects parity.
The [Fast-SAM-inspired inference report](BODY_FAST_INFERENCE.md) measures optional
native scheduling, corrective and crop changes against our previous implementation.
See [the roadmap](../docs/ROADMAP.md) for current scope, [the design](../docs/DESIGN.md)
for the acceptance process and [the source audit](AUDIT.md) for reuse decisions.
Historical component notes below describe their original, narrower milestones.
The optional Python environment is isolated in [python/](python/); it is not
needed to build or run the native library.

The independently public official MHR geometry asset is now verified and runnable
as a standalone upstream CPU/CUDA reference, with safe extraction and GGUF
conversion. Its native parameter-to-skeleton path now passes operation and
uninstrumented-skeleton checks on CPU and Vulkan/CUDA. Complete native MHR mesh
decoding also passes original operation and final-vertex checks on the official
demo parameters. See [MHR.md](MHR.md) for its exact scope, commands and repeatability
evidence. This does not supply gated SAM neural weights or complete SAM inference.

The [raw-image composition guide](BODY_IMAGE_FLOW.md) covers original-image
preparation through the full backbone and decoder using native-produced
intermediates, with explicitly synthetic SAM state and real MHR geometry.
The [full decoder guide](BODY_FLOW.md) covers the complete per-layer pose, real
MHR, camera and keypoint-feedback composition, with 526 checks per backend.
Its small-width synthetic SAM state is not trained image-to-body parity.

The [Body output guide](BODY_OUTPUT.md) covers the composed original/native pose
head through real MHR and output mapping, with synthetic SAM head state. Its 70
checks per backend do not establish trained Body or image-to-mesh parity.

The [Objects image/mask guide](OBJECTS_IMAGE.md) documents original default RGBA
preprocessing, 114 boundary/full-output checks on an official example and edge
cases, its opaque C API and sanitizer fuzzing. It does not establish pointmap,
learned-conditioner or full Objects inference parity.

The [Objects geometry parity report](OBJECTS_MESH_PARITY.md) records the pinned
CUDA/spconv oracle, native mesh-decoder stage gates, exact FlexiCubes extraction
check, end-to-end structural and topology comparison, and three-view rendered
comparison on the official kids-room exemplar. The earlier
[Objects runtime report](OBJECTS_RUNTIME.md) retains the separate Gaussian-path
evidence and its still-open official Gaussian-render gate.

## Model access

Development access to both official model repositories was confirmed on
2026-09-09. New users should request access at
[SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3) and
[SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects), then authenticate
the official HF CLI. Do not work around repository access restrictions with
unreviewed mirrors. Credential values must never appear in logs/manifests or
be mounted into the checkpoint-loading container.

After access is granted, download selected files at the revisions in
`sources.json`, not an unpinned whole repository. For example:

```sh
uv tool run --from huggingface-hub==0.36.0 hf download \
  facebook/sam-3d-body-dinov3 model.ckpt model_config.yaml assets/mhr_model.pt \
  --revision 11aaa346c7204874a1cbafe3d39a979080b2c55a --local-dir models/body
```

No host-side program should deserialize these legacy checkpoints. The separate
isolated Body extractor now emits byte-verified safe tensors; see
[TRAINED_BODY.md](TRAINED_BODY.md) for commands, the real-state inventory and the
currently failing intermediate trained-backbone gates. Full-model reference
integration remains unfinished.

## Source/artifact preflight

Commands below run from the project root. `preflight.py` uses only the Python
standard library and Git. It never imports upstream code or opens pickle data.

```sh
python3 reference/preflight.py --component body \
  --source reference/upstream/sam-3d-body --source-only \
  --report generated/preflight/body-source.json

python3 reference/preflight.py --component body \
  --source reference/upstream/sam-3d-body --model-directory models/body \
  --report generated/preflight/body-artifacts.json
```

A successful source-only check asserts only the source boundary. Even the
artifact check covers only the listed files, not auxiliary dependencies or the
Python/container environment. Reports explicitly leave `reference_ready=false`.
The source must be at the exact pin with no tracked or untracked changes.
Body's checkpoint, MHR asset and configuration were downloaded from the pinned
official repository and locally size/SHA-256 verified on 2026-09-09. Objects
six principal inference checkpoints and seven pipeline/component configurations
also pass local size/SHA-256 verification. Configuration Git blob identities
match the pinned official repository. All identities are recorded in the manifest.
This is artifact verification, not trained-model parity.

## Layer/operation comparison

Install the small data-only tools (no PyTorch):

```sh
uv sync --project reference/python --frozen
uv run --project reference/python --frozen python -m unittest discover -s tests -v
```

`scripts/check_parity.py` compares safe tensor captures in a specified operation
order. Every tensor must have an explicit rule, and both files must contain
exactly that tensor set. Floating checks require **both** max-absolute and
relative-L2 limits; integer/bool outputs require exact equality. Shapes, dtypes,
non-finite/empty arrays and missing taps cannot silently pass.

Example rule schema (values illustrate syntax only; not SAM tolerances):

```json
{
  "schema_version": 1,
  "boundary": "synthetic projection test, not SAM model parity",
  "tensors": [
    {"name": "block.0.qkv", "mode": "float", "max_abs": 0.00001,
     "relative_l2": 0.00001, "zero_reference_floor": 0.000000000001},
    {"name": "selected_indices", "mode": "exact"}
  ]
}
```

```sh
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/upstream.safetensors \
  --candidate generated/fixtures/native.safetensors \
  --rules generated/fixtures/rules.json --report generated/parity/report.json
```

For model captures, freeze rules based on actual upstream repeatability before
native tuning. Capture floating taps in a common explicit dtype (normally F32)
while recording the original execution precision separately. Reports include
artifact/rule hashes and the first failing boundary in declared execution order.
These hashes provide identity, not proof that a fixture came from upstream;
validated capture provenance and final pipeline/browser gates remain necessary.

## Actual-upstream crop operation fixtures

This first native boundary needs no weights. The reviewed standalone upstream
`bbox_utils.py` is verified by SHA-256 before import; it is not replaced with a
handwritten Python oracle. `capture_body_crop.py` invokes its center/scale,
aspect expansion and affine functions. It does not yet validate the full
`TopdownAffine` wrapper, image resampling, normalization or learned model.

Build the reviewed image, then run captures with no network, credentials or GPU:

```sh
docker build -f reference/Dockerfile.crop -t sam3d-reference-crop reference
mkdir -p generated/fixtures/body-crop
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --memory 2g --pids-limit 128 \
  --tmpfs /tmp:rw,nosuid,nodev,size=128m \
  -v "$PWD:/work:ro" -v "$PWD/generated/fixtures/body-crop:/output:rw" \
  sam3d-reference-crop --upstream /work/reference/upstream/sam-3d-body --output /output

./build/debug/bin/sam3d-crop-capture generated/fixtures/body-crop/cases.txt \
  generated/fixtures/body-crop/native.txt
uv run --project reference/python --frozen python scripts/pack_crop_capture.py \
  --input generated/fixtures/body-crop/native.txt \
  --output generated/fixtures/body-crop/native.safetensors
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/body-crop/upstream.safetensors \
  --candidate generated/fixtures/body-crop/native.safetensors \
  --rules generated/fixtures/body-crop/rules.json \
  --report generated/fixtures/body-crop/parity.json
```

Record the resolved image ID and run three fresh captures into separate output
directories before native comparison. Initial three captures were byte-identical
with Python 3.11.12, NumPy 2.2.6 and OpenCV 4.11.0. Four F32 center/scale tensors
per case compare exactly; affine F64 comparison allows only double-solve roundoff.
The native debug binary retains ASan/UBSan/LeakSanitizer. Under a ptrace-based
sandbox LeakSanitizer cannot run; use a normal unsandboxed test invocation rather
than silently disabling it. Reports are generated artifacts, not model parity.

## Actual-upstream RGB image preparation

`Dockerfile.image` adds the dependencies needed to import the original SAM
transforms and `BaseModel.data_preprocess`. `capture_body_image.py` uses package
namespace paths to avoid unrelated model/detector initialization; it does not
replace transform functions. It invokes normalization as the actual unbound
method with explicit mean/std inputs. No model weights or neural forward run.

```sh
docker build -f reference/Dockerfile.image -t sam3d-reference-image reference
mkdir -p generated/fixtures/body-image
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --memory 4g --pids-limit 128 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  -v "$PWD:/work:ro" -v "$PWD/generated/fixtures/body-image:/output:rw" \
  sam3d-reference-image --upstream /work/reference/upstream/sam-3d-body --output /output
./build/debug/bin/sam3d-image-capture generated/fixtures/body-image generated/fixtures/body-image-native
uv run --project reference/python --frozen python scripts/pack_image_capture.py \
  --input generated/fixtures/body-image-native \
  --output generated/fixtures/body-image-native/native.safetensors
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/body-image/upstream.safetensors \
  --candidate generated/fixtures/body-image-native/native.safetensors \
  --rules generated/fixtures/body-image/rules.json \
  --report generated/fixtures/body-image-native/parity.json
```

Run three fresh upstream captures to verify repeatability. Cases include
patterns, noise, black/near-black/white images and the official dancing photo,
with rotations, off-image boxes and padded input strides. Native uses its own
box/affine calculation, not injected reference matrices. All cropped U8 and
normalized F32 tensors currently match byte-for-byte. The JPEG is decoded by
the reference into shared RGB input; native JPEG/PNG decoding is **not** covered
by this boundary. Full model and browser end-to-end acceptance remain pending.

The earlier affine-only tolerance missed a pixel-level difference: a direct
affine solve changed last bits of F64 matrix entries, crossing fixed-point
sampling thresholds in rotated images. Native now follows OpenCV's small-system
LU operation order. A full rotated-pattern RGB checksum from actual upstream
is included in the normal CTest suite, with provenance under `tests/fixtures/`.

## DINOv3 patch graph contract (synthetic weights)

Clone the official DINO source into `reference/upstream/dinov3` at
`6876159a11b4df116f30f667f8c9888617df0751` (also in `sources.json`).
`capture_dino_patch.py` checks the original layer's hash before importing it.
It assigns stored synthetic F32 weights and calls the unmodified `PatchEmbed`
forward. Auxiliary PyTorch unfold and bias-free convolution diagnostics locate
intermediate graph errors; they are not claimed as captured upstream internals.
This tests the operation contract, **not trained DINO/Body parity**.

```sh
mkdir -p generated/fixtures/dino-patch
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --memory 4g --pids-limit 128 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  -v "$PWD:/work:ro" -v "$PWD/generated/fixtures/dino-patch:/output:rw" \
  --entrypoint python sam3d-reference-image /work/reference/capture_dino_patch.py \
  --upstream /work/reference/upstream/dinov3 --output /output

uv run --project reference/python --frozen python scripts/run_patch_capture.py \
  --reference generated/fixtures/dino-patch --output generated/fixtures/dino-patch-cpu \
  --binary build/debug/bin/sam3d-patch-capture --module build/debug/bin/libggml-cpu.so \
  --backend CPU
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/dino-patch/upstream.safetensors \
  --candidate generated/fixtures/dino-patch-cpu/native.safetensors \
  --rules generated/fixtures/dino-patch/rules.json \
  --report generated/fixtures/dino-patch-cpu/parity.json
```

Repeat fresh upstream captures three times. For Vulkan, build with
`SAM3D_VULKAN=ON`, select `--module build/debug/bin/libggml-vulkan.so`,
`--backend Vulkan --device INDEX --expect-device "EXACT DEVICE DESCRIPTION"`,
and use a separate output/report directory.
The diagnostic prints the actual device name/description and checks every graph
operation is supported on it. Backend module paths load executable code and must
be caller-trusted: never take them from uploaded files or model metadata.
No implicit discovery or fallback is used. The reference/native input protocol
is bounded diagnostic data, not the future GGUF model format.

If a diagnosed NVIDIA driver/ASan incompatibility prevents ICD initialization,
use the separate `vulkan-ubsan` preset and its binary/module paths for NVIDIA
only; retain the default ASan/UBSan build for CPU. Do not disable sanitizer
findings, accept a different GPU, or turn this workaround into the default.

The graph requests F32 im2col and F32 matmul precision explicitly: GGML's
convenience convolution currently converts F32 input patches to F16 by default.
Taps cover patch extraction, bias-free projection and final token layout on
batch/non-square/non-divisible cases plus the 512x512/1280-output Body shape.

Vulkan F32 tensor storage and `GGML_PREC_F32` accumulation alone do not prevent
the current backend from rounding operands to F16. `run_patch_capture.py`
therefore sets `GGML_VK_DISABLE_F16=1`, `GGML_VK_DISABLE_COOPMAT=1` and
`GGML_VK_DISABLE_COOPMAT2=1` in the child process by default. Direct diagnostic
invocations must set those before loading the Vulkan module. The recorded child
backend environment is part of the parity evidence. `--vulkan-math ggml-default`
is an explicit investigative override, not an accepted F32 configuration.
Both CPU and this strict NVIDIA configuration pass the same frozen 12-tensor
rules. No complete transformer, trained model or performance claim follows.

## DINOv3 transformer block contract

`capture_dino_block.py` imports the original eval block, attention, LayerScale,
SwiGLU and RoPE modules from the pinned DINO source, verifying hashes first.
It uses nontrivial synthetic parameters, including nonzero key bias masked by
the actual model mask. Observers must preserve the original uninstrumented
output exactly. Native receives input tokens and parameter state, including
persistent RoPE periods, but no upstream-produced intermediate activations.

```sh
mkdir -p generated/fixtures/dino-block
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --memory 8g --pids-limit 128 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  -v "$PWD:/work:ro" -v "$PWD/generated/fixtures/dino-block:/output:rw" \
  --entrypoint python sam3d-reference-image /work/reference/capture_dino_block.py \
  --upstream /work/reference/upstream/dinov3 --output /output
uv run --project reference/python --frozen python scripts/run_patch_capture.py \
  --reference generated/fixtures/dino-block --output generated/fixtures/dino-block-cpu \
  --binary build/debug/bin/sam3d-block-capture --module build/debug/bin/libggml-cpu.so \
  --backend CPU
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/dino-block/upstream.safetensors \
  --candidate generated/fixtures/dino-block-cpu/native.safetensors \
  --rules generated/fixtures/dino-block/rules.json \
  --report generated/fixtures/dino-block-cpu/parity.json
```

Use the same Vulkan runner options/strict-F32 setup described above. Twenty-two
boundaries per case include normalization, masked QKV, head layout, RoPE,
attention, projection, scales, residuals and SwiGLU. Auxiliary logits/probabilities
are distinguished from actual module taps; the real SDPA output is observed at
the attention projection's input. All 88 checks pass on CPU and NVIDIA, including
the 1029-token/1280-channel/20-head Body block dimensions. Three fresh reference
runs are byte-identical. These are **synthetic-weight block contracts**, not a
trained 32-block backbone or final Body model result.

`--small-regression` generates one compact fixture with two batches, multiple
heads, five prefix tokens and a non-square grid. Its `regression.txt` is retained
as `tests/fixtures/dino-block.txt`, with provenance in the adjacent JSON file.
Normal CTest validates all 22 taps with the same frozen two-metric limits and
rejects missing/wrong-sized tensors, nonpositive periods and invalid key masks.

For a direct CUDA reference, run the same container command with `--device cuda`
as a capture argument and `-e CUBLAS_WORKSPACE_CONFIG=:4096:8` plus the appropriate
Docker GPU selection. On CDI-enabled systems, `--device=nvidia.com/gpu=all`
selects NVIDIA explicitly; other installations commonly use `--gpus all`.
The capture disables TF32 and records the actual CUDA device. Use a separate
fixture directory and run native against **that directory's input files**:
CPU and CUDA initialization of persistent RoPE periods can differ by a last bit.
This direct Vulkan/PyTorch-CUDA block comparison also passes all 88 checks.
It establishes neither trained-model nor performance parity.

## Complete DINOv3 backbone contract

`capture_dino_backbone.py` invokes the original Body wrapper's `forward` method
with the original DINO encoder. Its normal cases exercise multiple batches,
non-square images, three/two consecutive blocks, and present/absent storage tokens.
`--full-shape` uses the unchanged original H+ factory: 32 blocks, 512×512 RGB,
1280 channels and four storage tokens. **Parameters and images are synthetic**.
Nontrivial LayerScale values stress propagation of numerical error across blocks.
Instrumentation must exactly preserve the unobserved original output.

Use the isolated reference command above, substituting the capture script and
an independent output directory, and add:

```sh
--body-upstream /work/reference/upstream/sam-3d-body --full-shape --device cuda
```

The full-size input fixture is approximately 3.36 GB; it is generated under the
ignored output tree and never checked into Git. Reference output includes the
patch projection, concatenated prefix/image tokens, every block output, final
normalization and the Body wrapper's `[B,1280,32,32]` patch feature map.
Every tap is an actual original method/module output, with layout-only patch
flattening explicitly reflecting token order. No reference intermediate is fed
into a native stage.

Run the existing native runner with `--binary build/PRESET/bin/sam3d-backbone-capture`
and compare using `check_parity.py` and the generated rules. The strict Vulkan
environment and NVIDIA/ASan limitation described above still apply. All 36
full-size boundaries pass ASan/UBSan CPU and NVIDIA versus PyTorch CUDA with the unchanged `1e-4`
maximum-absolute and `2e-5` relative-L2 limits. This is not optimized timing or
learned-model validation.

The native runner streams diagnostic stage messages as they finish and hashes
large input files in bounded chunks. `--threads N` selects CPU workers explicitly
and is recorded in the command manifest; the default remains one worker. Keep
ASan/UBSan enabled for CPU correctness checks, including parallel runs.

`--small-regression` (CPU, excluding `--full-shape`) emits a compact text fixture
for normal native CTest. It covers three blocks, two batches and a non-square
grid. Its final feature tensor is generated through the same original Body
wrapper. The adjacent manifest records source hashes and artifact identities.

The internal `dino_backbone` accepts a checked parameter provider; `body_backbone`
connects it to a validated complete H+ GGUF archive at the official fixed input
shape. Blocks currently stream sequentially, retaining only native activations
between them. That is a bounded-memory correctness implementation, not the final
resident/reused-graph performance design. GGUF-to-learned-features acceptance
awaits official weight access and safe extraction.

## Body CameraEncoder contract

The camera reference adds the pinned `einops` wheel to the existing image:

```sh
docker build -f reference/Dockerfile.camera -t sam3d-reference-camera reference
```

Its default parent is `sam3d-reference-image`; override `BASE_REFERENCE` with a
verified local image tag if needed. The Dockerfile-specific ignore file excludes
upstream/model/generated trees from the build context. No Nix build is involved.
Wheel installation uses `--require-hashes`, `--only-binary` and `--no-deps`.
Runtime reference execution remains offline, non-root and read-only.

Run the isolated capture command with this image and
`/work/reference/capture_camera_encoder.py --upstream /work/reference/upstream/sam-3d-body --output /output`.
Use `--device cuda` with the NVIDIA/CUBLAS settings documented above for a CUDA
reference. Native execution uses `sam3d-camera-capture` with the existing runner
and generated `rules.json`. Normal cases include batch two, non-square images,
factors 1/2/3/16, and the real 512x512 ray/1280-channel feature dimensions.

Eight boundaries per case cover antialiased rays, three-component ray positions,
frequencies, 99-channel Fourier encoding, feature concatenation, projection,
normalization and final NCHW output. Frequencies are a labelled auxiliary
`torch.linspace` diagnostic; other taps observe original module inputs/outputs
with only explicit layout views. Hooks must preserve the unobserved result exactly.
The channel norm uses `eps=1e-6` and explicit centered-variance/divide-sqrt order,
not the DINO backbone's `eps=1e-5` contract. Antialiasing is not ordinary bilinear
sampling: the filter widens for downsampling and normalizes truncated edge support.

`--small-regression` emits the 40KB text fixture retained in
`tests/fixtures/camera-encoder.txt`. Normal CTest checks all eight boundaries and
invalid-input rejection with ASan/UBSan/LSan. Synthetic feature/ray/parameter
inputs establish this component only; they do not validate camera-coordinate
construction, learned weights, decoder behavior or the image-to-pose pipeline.

## Crop/intrinsics to camera geometry

The public C API now constructs camera rays and CLIFF decoder conditioning from
its own crop result, explicit `fx,fy,cx,cy` intrinsics and original-image size.
It preserves `prepare_batch`'s F64-to-F32 affine metadata boundary, then the exact
upstream division/subtraction order. CLIFF uses `fx` for all three components;
the ray grid uses `fx`/`fy` separately. Both image-center and intrinsics-center
CLIFF conventions are exposed. Only axis-aligned square camera crops up to 512²
are accepted; this matches the first selected Body path, not arbitrary rotated
or rectangular crop behavior.

`capture_body_camera.py` invokes the original transforms and `prepare_batch`,
then the unchanged hash-verified AST bodies of `get_ray_condition` and
`_get_decoder_condition`, with original BaseModel person flattening. It avoids
full-model imports/assets without replacing expressions or numerical operations.
The original ray method hardcodes `.cuda()`, so this reference uses the isolated
NVIDIA container even though native camera geometry needs no GPU.

Use the camera reference image and the documented CUDA/offline container flags,
substituting `/work/reference/capture_body_camera.py`. Native comparison:

```sh
uv run --project reference/python --frozen python scripts/run_body_camera.py \
  --reference generated/fixtures/body-camera --output generated/fixtures/body-camera-native \
  --binary build/debug/bin/sam3d-body-camera-capture
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/body-camera/upstream.safetensors \
  --candidate generated/fixtures/body-camera-native/native.safetensors \
  --rules generated/fixtures/body-camera/rules.json \
  --report generated/fixtures/body-camera-native/parity.json
```

Twelve cases/60 boundaries cover original-image sizes, fractional/off-image boxes,
multiple padding factors, unequal focal lengths, off-center principal points,
both CLIFF conventions and ray grids up to 512². Center, expanded scale, F32 affine,
rays and final CLIFF vectors match byte-for-byte. Native receives only raw case
inputs, not upstream geometry. `--small-regression` emits the 14KB original text
fixture used by normal CTest. This is geometry-method parity, not a neural-model
or complete Body entry-point result.

The camera C API is included in the sanitizer-enabled crop/image fuzzer; GGUF
loading remains excluded. If a launcher reports a large peak RSS before the
first fuzz input, verify actual child RSS before weakening the memory limit.
On this host a fresh forked child avoided inherited high-water accounting;
the same sanitizer-enabled binary completed 100,000 cases below the unchanged
default limit. No memory-access finding was suppressed.

## Body decoder-layer contracts

`capture_body_decoder.py` imports the hash-verified original
`TransformerDecoderLayer`, without modifying its forward or SDPA implementation.
It runs evaluation F32, origin/GELU FFN, no LayerScale, LayerNorm eps=1e-6 and
explicit PyTorch MATH SDPA with TF32 off. Seven synthetic cases exercise separate
Q/K/V projections, positional repeat/skip and batch broadcasting, masks including
fully invalid reverse-attention rows, unequal token/context widths, and a large
1024/1280-channel case. Original observed and unobserved outputs must be equal.
Q/K/V inputs/projections, normalization, attention projection inputs/outputs,
FFN and residuals are original module taps. Logit/probability expansions are
separately labelled auxiliary diagnostics; they do not replace reference SDPA.

Use the camera reference image/offline container instructions, substituting
`/work/reference/capture_body_decoder.py`. `--device cuda` uses the documented
NVIDIA CDI/CUBLAS flags. `--small-regression` emits the original six-case text
fixture used in normal CTest. Native CPU comparison:

```sh
uv run --project reference/python --frozen python scripts/run_patch_capture.py \
  --reference generated/fixtures/body-decoder-cpu \
  --output generated/fixtures/body-decoder-native-cpu \
  --binary build/debug/bin/sam3d-decoder-capture \
  --module build/debug/bin/libggml-cpu.so --backend CPU --threads 12
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/body-decoder-cpu/upstream.safetensors \
  --candidate generated/fixtures/body-decoder-native-cpu/native.safetensors \
  --rules generated/fixtures/body-decoder-cpu/rules.json \
  --report generated/fixtures/body-decoder-native-cpu/parity.json
```

For Vulkan substitute its build/module/backend and require the expected device
description; the runner defaults to the strict-F32 configuration described above.
All 290 boundaries pass CPU and NVIDIA Vulkan, including direct PyTorch CUDA
comparison with verified identical input bytes. Three CUDA captures repeat
exactly. These component tests neither use trained weights nor execute
`PromptableDecoder`'s intermediate pose-head/geometry/keypoint feedback; that
full loop must pass its own original-upstream test before claiming decoder parity.

## Body prompt and positional encoding

`capture_body_prompt.py` invokes the original `PromptEncoder`, dense-grid
`PositionEmbeddingRandom.forward` and `forward_with_coords`. A scoped observer
calls the unchanged bound `_pe_encoding` method and records the results of its
original ATen operations without replacing them. It checks exact equality with
uninstrumented outputs. There is no rewritten numerical oracle.

The four synthetic cases cover both special negative labels, every one of 70
joint labels, batch two, non-square grids, normalized endpoints, pixel coordinates
outside the image and a 32²/1280-channel dense grid. Mask-convolution variants and
an absent point tensor are outside this component; initial Body inference uses
the explicit invalid [0,0,-2] point. All 108 boundaries pass CPU and NVIDIA Vulkan
against their respective original PyTorch devices. Three CUDA captures repeat
exactly. The Gaussian matrix and label embeddings are stored model state, not
injected reference activations.

Use the same offline camera reference image, substituting
`/work/reference/capture_body_prompt.py`; use `--device cuda` and the documented
NVIDIA/CUBLAS container flags for CUDA. `--small-regression` produces either of
the two 50KB original fixtures used in normal CTest. Native comparison:

```sh
uv run --project reference/python --frozen python scripts/run_patch_capture.py \
  --reference generated/fixtures/body-prompt-cpu \
  --output generated/fixtures/body-prompt-native-cpu \
  --binary build/debug/bin/sam3d-prompt-capture \
  --module build/debug/bin/libggml-cpu.so --backend CPU
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/body-prompt-cpu/upstream.safetensors \
  --candidate generated/fixtures/body-prompt-native-cpu/native.safetensors \
  --rules generated/fixtures/body-prompt-cpu/rules.json \
  --report generated/fixtures/body-prompt-native-cpu/parity.json
```

For Vulkan use the CUDA capture and corresponding Vulkan binary/module/backend,
with strict F32 and the expected NVIDIA device guard. The diagnostic explicitly
selects direct scalar division on CPU and reciprocal multiplication for Vulkan's
CUDA-equivalent reference. PyTorch's original CUDA scalar division uses the latter
fast path; applying CPU division first failed exact coordinate assertions despite
small final errors. Both modes are now tested on CPU without needing a GPU in
normal CTest. Coordinate and mask checks remain exact; neural checks remain
max-absolute 1e-4 and relative-L2 2e-5. Do not compare different scalar conventions
as if their coordinates were byte-identical, or loosen rules to hide the difference.
Model composition must select/preserve the appropriate arithmetic convention.

## Conditioning into the first decoder layer

`capture_body_condition.py` executes the unchanged `SAM3DBody.forward_decoder`
method AST with original prompt/camera/decoder modules. It stops explicitly in
a hook immediately after the first real decoder layer, before any pose-head or
geometry feedback. No fake final outputs or substitute callbacks are used.
Additional observers must leave those baseline first-layer outputs exactly
unchanged. This is a partial-method composition boundary, not the complete
decoder loop. The later full-model test must run without this termination hook.

Native `body_condition` builds the initial pose/camera plus CLIFF input, selects
the initial or supplied previous estimate, projects pose/previous/prompt tokens,
uses native prompt and camera outputs, and assembles hand/keypoint/3D-keypoint
tokens and positional augmentation. The diagnostic converts its own image
features to BNC and feeds its own tokens/positions to `body_decoder_layer`.
It deliberately does not pass the prompt mask: original code sets the decoder
mask to `None`, including the invalid initial prompt. Synthetic inputs include
features, rays and CLIFF; upstream-produced conditioning or layer activations are
never injected into native execution.

Use the same offline camera container with
`/work/reference/capture_body_condition.py`; `--device cuda` selects the original
CUDA reference and `--small-regression` emits the normal CTest fixture. Run:

```sh
uv run --project reference/python --frozen python scripts/run_patch_capture.py \
  --reference generated/fixtures/body-condition-cpu \
  --output generated/fixtures/body-condition-native-cpu \
  --binary build/debug/bin/sam3d-condition-capture \
  --module build/debug/bin/libggml-cpu.so --backend CPU --threads 12
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/body-condition-cpu/upstream.safetensors \
  --candidate generated/fixtures/body-condition-native-cpu/native.safetensors \
  --rules generated/fixtures/body-condition-cpu/rules.json \
  --report generated/fixtures/body-condition-native-cpu/parity.json
```

Use the CUDA fixture/Vulkan binary and module for NVIDIA comparison, with the
strict-F32 and expected-device flags documented above. All 52 boundaries pass
their matched CPU/CUDA reference. The large synthetic case uses the current
MHR source's 519 pose values (not the stale 404-value comment), 512² rays and
143 tokens. This does not establish learned checkpoint shape compatibility;
checked conversion must verify that independently when access is authorized.

## Camera prediction and full-perspective projection

`capture_camera_head.py` invokes the original `PerspectiveHead.forward` and
`PerspectiveHead.perspective_projection`, including the original geometry helper.
It observes actual FFN/ReLU outputs, camera-function return locals and the actual
ATen depth-division result without replacing any original numerical operation.
Instrumented and uninstrumented head/projection outputs must match exactly.

Five synthetic cases cover one-to-three-layer FFNs, optional initial estimates,
both camera-center conventions, unequal focal lengths and off-diagonal camera
intrinsics, positive/negative camera-space depth, a 1024-wide token and 18,439
points. The last point count is a representative mesh-sized workload, not a
claim about learned MHR topology. Native computation uses its own predicted
camera through to final pixels; reference intermediate cameras are not injected.

Use the offline camera reference image with `/work/reference/capture_camera_head.py`;
`--device cuda` and the documented NVIDIA/CUBLAS flags select CUDA.
`--small-regression` emits the 16KB original CTest fixture. Native CPU comparison:

```sh
uv run --project reference/python --frozen python scripts/run_patch_capture.py \
  --reference generated/fixtures/camera-head-cpu \
  --output generated/fixtures/camera-head-native-cpu \
  --binary build/debug/bin/sam3d-camera-head-capture \
  --module build/debug/bin/libggml-cpu.so --backend CPU --threads 12
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/camera-head-cpu/upstream.safetensors \
  --candidate generated/fixtures/camera-head-native-cpu/native.safetensors \
  --rules generated/fixtures/camera-head-cpu/rules.json \
  --report generated/fixtures/camera-head-native-cpu/parity.json
```

Use the CUDA fixture and strict-F32 Vulkan backend for NVIDIA comparison. All 68
boundaries pass, and three CUDA captures repeat exactly. Rules distinguish pixel/
box units (max-absolute 1e-3) from camera/latent units (1e-4), require relative-L2
2e-5, and compare focal lengths exactly. The first failing operation identified
an extra native K transpose; correcting it passed without changing the rules.
The native API rejects undefined zero-depth projections instead of silently
clamping or exposing non-finite buffers. Pose-head/MHR geometry and full decoder
feedback composition are still separate, unfinished requirements.

## Crop projection and keypoint feedback

`capture_body_feedback.py` executes the unchanged original `_full_to_crop`,
`keypoint_token_update_fn` and `keypoint3d_token_update_fn` method ASTs, with the
original flatten helper, FFNs and `grid_sample`. Operation/module/return-local
observers must leave the uninstrumented result exactly unchanged. The selected
backbone convention is DINOv3; alternate ViT and hand callbacks are not covered.

Five synthetic cases exercise crop transforms, boundaries, depth validity,
single-pixel/non-square images, 2D/3D index ordering and last-layer identity,
including 70+70 keypoints at the model's feature/token width. Predicted geometry
is supplied as synthetic input. Native execution constructs its own crop points,
sampled features and projected token updates; no reference intermediate is
injected. The complete model must eventually supply these inputs from native
pose-head/MHR geometry, not from fixtures.

Use the documented offline camera container with
`/work/reference/capture_body_feedback.py`; `--device cuda` selects CUDA and
`--small-regression` emits the 96KB original CTest fixture. CPU comparison:

```sh
uv run --project reference/python --frozen python scripts/run_patch_capture.py \
  --reference generated/fixtures/body-feedback-cpu \
  --output generated/fixtures/body-feedback-native-cpu \
  --binary build/debug/bin/sam3d-feedback-capture \
  --module build/debug/bin/libggml-cpu.so --backend CPU
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/body-feedback-cpu/upstream.safetensors \
  --candidate generated/fixtures/body-feedback-native-cpu/native.safetensors \
  --rules generated/fixtures/body-feedback-cpu/rules.json \
  --report generated/fixtures/body-feedback-native-cpu/parity.json
```

Use the CUDA reference, Vulkan diagnostic/module and strict-F32 expected-device
settings above for NVIDIA comparison. All 77 boundaries pass; three independent
CUDA captures repeat exactly. Homogeneous inputs, selected depths, validity masks
and final-layer unchanged tokens/positions are exact; floating rules use max-abs
1e-3 for crop pixels, 1e-4 elsewhere and relative-L2 2e-5.

The native bilinear sampler is CPU code between GGML graphs even in Vulkan
sessions. This deliberately records current backend coverage: the check is not
proof of a GPU-resident full decoder or performance parity. Preserve upstream's
feature mask *before* linear projection (invalid samples still add bias), its
separately masked 2D positional update, and hip centering before 3D selection.

## MHR pose head through geometry inputs

`capture_body_pose.py` executes unchanged original `MHRHead.forward`,
`replace_hands_in_pose` and `mhr_forward` ASTs. It uses source-defined scalar
configuration and synthetic FFN/buffers without calling the asset-loading
constructor. Execution stops on the source line **before** `self.mhr(...)`;
no geometry callback is replaced. This is a method-prefix contract, not complete
head/model execution. The MHR inputs with/without extra observation must be exact.

The prefix includes the 519-value FFN/residual; global Gram-Schmidt → quaternion
→ intrinsic ZYX Euler; body/hand cross-product 6D and sin/cos decoding; hand/jaw/
face masking; scale/hand PCA and indexed assembly of the 204 MHR parameters.
Eight synthetic cases include all four quaternion branches, body singularities,
tiny/zero/collinear vectors, optional initial estimates, one-to-three-layer FFNs,
nontrivial PCA/index buffers and full 1024-wide tokens. Trained hand index buffers
must still be loaded and validated, not inferred from these test indices.

The additional upstream dependency is the official data-only wheel
`roma-1.6.1-py3-none-any.whl`, SHA-256
`79c3a07ab94c0e784e8de2ecc878645a5cf0aff312f7b1e281ca1be7c5f0b7e8`:

```sh
mkdir -p generated/wheels
curl --fail --location --output generated/wheels/roma-1.6.1-py3-none-any.whl \
  https://files.pythonhosted.org/packages/8f/e5/484ff091291cfb9ba82977717ca817456f9a22d519906ec9d16629036360/roma-1.6.1-py3-none-any.whl
sha256sum generated/wheels/roma-1.6.1-py3-none-any.whl
```

Do not execute it on the host. Run the reviewed camera container offline with
`/work/reference/capture_body_pose.py`, adding
`--roma-wheel /work/generated/wheels/roma-1.6.1-py3-none-any.whl`. The script verifies
the wheel hash and imports it directly inside the isolated container, so no new
image/install step is needed. It records individual Python source hashes too.
`--device cuda` selects the original CUDA reference; `--small-regression` creates
the six-case normal CTest fixture. Native CPU comparison:

```sh
uv run --project reference/python --frozen python scripts/run_patch_capture.py \
  --reference generated/fixtures/body-pose-cpu \
  --output generated/fixtures/body-pose-native-cpu \
  --binary build/debug/bin/sam3d-pose-capture \
  --module build/debug/bin/libggml-cpu.so --backend CPU
uv run --project reference/python --frozen python scripts/check_parity.py \
  --reference generated/fixtures/body-pose-cpu/upstream.safetensors \
  --candidate generated/fixtures/body-pose-native-cpu/native.safetensors \
  --rules generated/fixtures/body-pose-cpu/rules.json \
  --report generated/fixtures/body-pose-native-cpu/parity.json
```

Use CUDA fixtures and the strict-F32 Vulkan diagnostic/module for NVIDIA
comparison. All 320 boundaries pass at max-absolute 1e-4 and relative-L2 2e-5;
branch selections, singular masks and disabled translation/face are exact.
Native FFN/PCA uses GGML; rotation math and parameter scatter currently use CPU
geometry routines in either session. This is not GPU-resident whole-model or
performance acceptance. MHR asset inference, geometry output mapping and the
complete image-to-mesh test remain required.
