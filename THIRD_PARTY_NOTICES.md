# Third-party sources

Original project contributions use Apache-2.0; the adaptations below retain
their applicable upstream terms. See [LICENSING.md](LICENSING.md) and [NOTICE](NOTICE).
The original-project grant does not relicense these materials.

## Optional single-image web demo

`demo/web/vendor/three.{core,module}.min.js` are the already-vendored Three.js
distribution copied unchanged from motion-bricks.cpp's demo. The original MIT
notice is retained at `demo/web/vendor/THREE-LICENSE.txt`. They are frontend
rendering dependencies, not native inference dependencies. `demo/web/localai.png`
is the existing LocalAI branding asset reused from kimodo.cpp. LocalAI names and
logos identify the project; they do not imply that Meta endorses this port.

`scripts/devtools.py` adapts the reviewed standard-library DevTools transport
from trellis2cpp's `scripts/headless_smoke.py`, adding local-only connection
validation, bounded/fragmented frames and correct ping replies. The QA workflow,
Go server and viewer code are new, following those sibling applications' UI
patterns without depending on their running servers or source trees.
The helper's upstream MIT notice (Copyright (c) 2026 rms80) is retained in
[LICENSES/Trellis2cpp-MIT.txt](LICENSES/Trellis2cpp-MIT.txt).
`cmake/PrepareGGML.cmake` adapts skin-tokens.cpp's Apache-2.0 build-copy patch
phase, adding content-addressed copies; it does not require that sibling repo.

`demo/web/skeleton.json` contains MHR v1.0.1's 127 joint names and parent indices,
extracted from the verified public asset (SHA-256 recorded in the file). It is
topology, not learned weights; the existing MHR/Momentum notices below apply.
Official example images and original predicted geometry are not bundled. The
optional preparation script retains source/capture hashes in local artifacts.

## GGML runtime and DINOv3 patch embedding

`ggml/` is an unmodified public upstream submodule at
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` (GGML 0.23.0), under its
[MIT license](ggml/LICENSE). Default builds require no private commits or patches.
Experimental BF16 Vulkan builds optionally apply the reviewable patches in
`patches/ggml/` to a build-directory copy, leaving the upstream submodule pristine.
These adapt the pinned GGML Vulkan pipelines and shaders: scalar F32 matrix
products alongside BF16 cooperative-matrix-2 execution, and an exact BF16 cast
roundtrip fusion based on `contig_copy.comp`, and an accumulator-precision
BF16 flash-attention denominator using the existing shader reduction machinery.
The denominator follows the precision split in FlashAttention's original softmax
algorithm (reference linked in the patch notes; no FlashAttention code copied).
The BF16 probability-residual patch additionally adapts the existing GGML
cooperative-matrix P×V shader to accumulate a second product for the probability
rounding residual. It retains BF16 Q/K/V and the F32 denominator.
The opt-in binary BF16-round fusion adapts the same pinned `add.comp`,
`mul.comp`, binary pipeline dispatch and integer BF16 conversion functions.
It retains an F32 operation followed by the existing BF16 rounding boundary.
The optional linear-indexing variant additionally specializes contiguous
same-shape, repeated-row and scalar operands, using these same MIT-licensed
shader and dispatch helpers without changing the arithmetic or rounding.
The diagnostic BF16 matrix-tile selector adapts the same Vulkan pipeline
heuristic to select existing small/medium/large cooperative-matrix kernels.
Adapted GGML code remains under
its MIT license. See [patch scope and testing](patches/ggml/README.md).

`src/neural.cpp` expresses the F32 convolution and token layout contract of
Meta's [DINOv3 PatchEmbed](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/patch_embed.py)
using GGML operations. Copyright (c) Meta Platforms, Inc. and affiliates.
DINO materials and derivatives are subject to the included
[DINOv3 License](LICENSES/DINOv3.md). Changes: C++ graph, explicit backend,
checked shape/data bounds and intermediate diagnostic taps; optional patch norm
is not implemented. No DINO weights are included. The reference script imports
the hash-verified original class with stored synthetic inputs/weights; auxiliary
PyTorch unfold/bias-free convolution diagnostics are identified separately from
the original module's final output. These tests do not establish learned-model parity.

`src/dino_block.cpp` additionally adapts the eval-mode block, attention, RoPE,
LayerScale and SwiGLU semantics from `dinov3/layers/{block,attention,
rope_position_encoding,layer_scale,ffn_layers}.py` at the same DINOv3 revision,
under the same license. It uses explicit GGML operations, validated parameters,
F32 attention and diagnostic taps; training/drop-path and unsupported variants
are omitted. No third-party C++ engine was copied. The reference capture invokes
original classes and checks observer/non-observer output equality.
`tests/fixtures/dino-block.txt` contains synthetic inputs/parameters and original
upstream test outputs, not learned weights.

The Body-backbone schemas in `scripts/gguf_schema.py` and
`src/tensor_archive.cpp` follow the same original DINOv3 H+ factory and state
structure. `reference/capture_dino_schema.py` invokes the unchanged factory on
PyTorch's meta device; `tests/fixtures/dino-schema.json` records shapes and source
hashes, not learned weights. Conversion and checked archive parsing are new
project code using the standard GGUF format and GGML's metadata reader.

`src/dino_backbone.cpp` follows DINOv3 `prepare_tokens_with_masks` and
`get_intermediate_layers` in `models/vision_transformer.py`, together with
SAM 3D Body's `models/backbones/dinov3.py` wrapper at the pinned revisions above.
Changes: F32-only C++/GGML execution, checked parameters, sequential block
streaming and optional observation callbacks; no training, masking or extra
embeddings. Original reference methods generate the synthetic regression data
in `tests/fixtures/dino-backbone.txt` (provenance in its adjacent JSON file).
No learned parameters are contained in that fixture.

## Body camera conditioning

`src/camera_encoder.cpp` adapts Meta's `CameraEncoder`, Fourier feature layout
and `LayerNorm2d` from the pinned Body revision:

- [camera_embed.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/modules/camera_embed.py)
- [transformer.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/modules/transformer.py)

Changes: bounded F32 C++/GGML inputs and outputs, CPU ray resampling, GGML Fourier
features, 1x1 projection and channel normalization, with diagnostic taps. The
original classes generate `tests/fixtures/camera-encoder.txt` from synthetic
inputs/parameters, not learned weights; its adjacent JSON records provenance.

The F32 separable antialias filter and edge normalization follow
[PyTorch v2.7.0 UpSampleKernel.cpp](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp),
source SHA-256 `e3b7bf13fecd1af33e1c6d7dfb441d97590b71075194d040c5ad325814fdac1f`.
Copyright and redistribution terms are retained in [LICENSES/PyTorch.txt](LICENSES/PyTorch.txt).
Only integer-factor F32 downsampling with half-pixel centers, antialias enabled,
and horizontal-then-vertical evaluation is adapted; no PyTorch runtime is linked.
The camera reference image additionally uses hash-pinned `einops` 0.8.1, solely
as an upstream Python dependency. No einops source is copied into the native port.

`src/body_camera.cpp` adapts `SAM3DBody.get_ray_condition` and
`_get_decoder_condition` from the pinned Body `models/meta_arch/sam3d_body.py`.
It preserves the F32 metadata boundary in `data/utils/prepare_batch.py`, diagonal
ray transform operation order and CLIFF's use of horizontal focal length for all
three condition values. Unsupported rotated/non-square camera crops are rejected;
the square axis-aligned path uses portable native C++ instead of upstream's
hardcoded `.cuda()` call. The public interface adds opaque request/result handles
and validated caller-owned input/error buffers.

`reference/capture_body_camera.py` executes the original crop/batch preparation
and unchanged AST bodies of the two methods, with the original BaseModel person
flattening method. AST isolation avoids unrelated full-model imports and does
not rewrite expressions or replace numerical operations. Its scope is explicitly
geometry-method parity, not full-model execution. `tests/fixtures/body-camera.txt`
contains small original reference outputs and input cases, with adjacent provenance.

## SAM 3D Body decoder layer

`src/body_decoder.cpp` adapts `TransformerDecoderLayer`, `Attention`, `FFN` and
`LayerNorm32` from the pinned Body `models/modules/transformer.py` linked above,
under the SAM License. Changes: checked F32 C++/GGML execution, separate Q/K/V
projections, positional and mask handling, exact-erf GELU and operation taps;
evaluation only, eps=1e-6, no LayerScale/SwiGLU/training variants. A fully masked
reverse-attention row uses a safe dummy softmax followed by zero probabilities
to reproduce SDPA's zero-attention result without NaNs. No third-party C++
engine was copied. The original layer generates the synthetic fixture
`tests/fixtures/body-decoder.txt`; its adjacent JSON records provenance. Auxiliary
logit/probability expansions are labelled separately from original module taps.
This is not the full promptable decoder's pose/geometry feedback loop.

## SAM 3D Body complete decoder composition

`src/body_flow.cpp` additionally adapts complete intermediate-prediction control
flow from `SAM3DBody.forward_decoder` and
[PromptableDecoder.forward](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/decoders/promptable_decoder.py),
under the SAM License. Changes: checked native component composition, explicit
fixed residuals and image/context lifetimes, GGML affine final normalization,
per-layer output capture and configuration rejection. Original heads, real MHR
and unchanged method ASTs generate the synthetic-state complete-loop reference in
`reference/capture_body_flow.py`. The small `decoder-norm.txt` fixture isolates
original per-layer normalization and retains adjacent full-capture provenance.
See [full decoder validation](reference/BODY_FLOW.md) for scope and source hashes.

## SAM 3D Body prompt encoder

`src/body_prompt.cpp` adapts `PromptEncoder` and `PositionEmbeddingRandom` from
[prompt_encoder.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/decoders/prompt_encoder.py),
under the SAM License. Changes: checked F32 GGML Gaussian projection and
trigonometric operations, native dense/pixel coordinate construction and exact
label-table lookup/addition; no mask-convolution variants or absent-point input.
`tests/fixtures/body-prompt{,-cuda}.txt` contain synthetic state and original
operation outputs, with adjacent provenance JSON. The reference observer calls
the unchanged original bound method and returns every original ATen result
unchanged; instrumented and uninstrumented final outputs must be equal.

Scalar division preserves the direct CPU path or the reciprocal-multiply CUDA
path explicitly, following
[PyTorch v2.7.0 BinaryDivTrueKernel.cu](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu).
The included [PyTorch license](LICENSES/PyTorch.txt) applies to adapted PyTorch
material. No Python/PyTorch runtime is linked into the native library.

## SAM 3D Body decoder input construction

`src/body_condition.cpp` adapts the DINO/CLIFF input-construction portion of
`SAM3DBody.forward_decoder` from the pinned Body source. Changes: checked native
composition of camera and prompt encoding, F32 GGML token projections, explicit
previous-estimate fallback and token/augmentation assembly. It retains the
original absence of a decoder attention mask for invalid prompts. Alternate
ViT crops, external initial-estimate overrides and pose/geometry callbacks are
not implemented by this component.

`reference/capture_body_condition.py` executes the unchanged original method AST
with original camera/prompt/decoder modules and synthetic state. A hook terminates
after the first real decoder layer, before any pose-head/geometry feedback;
it does not invent a pose result or claim complete method execution. The adjacent
manifest records this boundary and source hashes for the original fixture
`tests/fixtures/body-condition.txt`. The method and its adaptations remain under
the [SAM License](LICENSES/SAM.txt).

## SAM 3D Body camera head and projection

`src/body_camera_head.cpp` adapts the original `PerspectiveHead` and
`geometry_utils.perspective_projection` from the pinned Body revision:

- [camera_head.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/heads/camera_head.py)
- [geometry_utils.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/modules/geometry_utils.py)

Changes: bounded F32 GGML evaluation, validated pinhole intrinsics and inputs,
one-to-three-layer ReLU FFN, optional initial-camera residual, explicit sign and
crop-to-camera conversion, depth normalization and intrinsics application, with
intermediate taps. Undefined/non-finite projections are rejected, not clamped.
The original modules/functions generate `tests/fixtures/camera-head.txt` from
synthetic state and points; adjacent JSON records provenance. Observation uses
module hooks, unchanged function return locals and a non-replacing ATen output
observer. This is not MHR geometry generation or complete Body inference.
These adaptations remain under the [SAM License](LICENSES/SAM.txt).

## Official MHR geometry asset and reference

The independently public [MHR v1.0.1 release](https://github.com/facebookresearch/MHR/releases/tag/v1.0.1)
contains `assets/mhr_model.pt` and `assets/LICENSE.txt`. The latter is retained in
[LICENSES/MHR-Apache-2.0.txt](LICENSES/MHR-Apache-2.0.txt). The asset hash matches
the published MHR companion identity used by SAM 3D Body. This does not grant or
claim access to the separate gated SAM neural checkpoints.

`reference/capture_mhr.py` loads only this exact hash-verified TorchScript inside
the isolated reference container. Input generation uses the unchanged official
MHR demo function at revision `e412e12c9d7287a598f00edf19242b476b440211`.
`scripts/mhr_schema.py` and `convert_mhr_gguf.py` select and rename the required
geometry tensors, preserve F32 and losslessly narrow bounded indices to I32.
Unused solver-limit state is omitted. No MHR or third-party engine runs in the
native loader. Converted assets are ignored local artifacts, not checked-in or
published model downloads. Standalone native MHR mesh parity now passes on the
original captured demo inputs; SAM Body model/output mapping is still separate.

`src/mhr_skeleton.cpp` adapts the parameter transform, local skeleton conversion,
quaternion math and F64 prefix composition from the Momentum functions embedded
in that exact released asset. Momentum copyright: Meta Platforms, Inc. and
affiliates; [MIT license](LICENSES/Momentum.txt), obtained from the
[official repository](https://github.com/facebookincubator/momentum/blob/main/LICENSE).
The released geometry asset's Apache-2.0 terms are also retained. No Momentum
runtime or third-party C++ engine is linked or executed by the native library.
The normal `mhr-local.txt` regression contains small original geometry constants
and captured local/FK values under the asset license, not neural weight matrices.
The original TorchScript functions remain the reference, not current Momentum
source or a Python reimplementation. Operation observations are validated against
an uninstrumented full-model skeleton. See [reference/MHR.md](reference/MHR.md).

`src/mhr_geometry.cpp` additionally adapts the released model's separate identity/
expression projections, corrective feature construction and Momentum's inverse-bind
and weighted skinning operations under those same terms. Projection implementation
uses GGML; the first COO layer is materialized densely natively but validated
against the released sparse operation. `capture_mhr_geometry.py` observes the
unchanged original full forward and verifies both intermediate and final outputs.
`mhr-skin.txt`/`.json` contain a small original 16-vertex LBS regression and its
provenance, including asset constants under the MHR asset license. The restricted
original module is checked against the same vertices in the full original mesh.
It is not a synthetic or replacement full-model oracle.

## SAM 3D Body MHR pose and parameter assembly

`src/body_output.cpp` adapts the output mapping in the same pinned `MHRHead` under
the [SAM License](LICENSES/SAM.txt), and RoMa's `unitquat_to_rotmat` under the
existing NAVER/SciPy [BSD attribution](LICENSES/RoMa.txt). The native composition
connects the pose head to the real MHR asset, then output mapping. Its reference
executes the original complete class with explicitly synthetic SAM head/PCA/index/
mapping buffers and the byte-verified actual MHR. The small `body-output.txt`
fixture includes generated geometry under the MHR asset license and synthetic
mapping state, not a trained SAM checkpoint. Full scope and source/tensor hashes
are documented in [reference/BODY_OUTPUT.md](reference/BODY_OUTPUT.md).

`src/body_pose.cpp` adapts the prefix of `MHRHead.forward`, `mhr_forward` and
`replace_hands_in_pose`, plus compact body/hand rotation decoding, from:

- [mhr_head.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/heads/mhr_head.py)
- [mhr_utils.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/modules/mhr_utils.py)
- [geometry_utils.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/modules/geometry_utils.py)

Changes: checked F32 GGML FFN/PCA projections, CPU rotation math and indexed
parameter assembly. Supports the body-mode 519-value prediction and 204-value
MHR parameter input; not the wrist-centric hand model, asset inference, optional
offsets, output keypoint mapping or mesh decoding. Original synthetic method
fixtures are in `tests/fixtures/body-pose.txt`, with adjacent provenance JSON.
The reference executes unchanged method ASTs and stops before the actual
`self.mhr` call; it does not substitute generated geometry. SAM adaptations and
fixtures remain subject to the [SAM License](LICENSES/SAM.txt).

Global matrix-to-quaternion and intrinsic ZYX Euler conversion adapt
`rotmat_to_unitquat` and `unitquat_to_euler` from
[RoMa 1.6.1](https://pypi.org/project/roma/1.6.1/), Copyright (c) 2020 NAVER Corp.
These functions include SciPy-derived algorithms. The original BSD-3-Clause
license and SciPy notice are retained in [LICENSES/RoMa.txt](LICENSES/RoMa.txt).
Changes specialize the convention, replace tensor batches with bounded scalar
F32 loops and retain original branch/normalization behavior. RoMa/PyTorch are
reference-only Python dependencies, not native runtime dependencies.

## SAM 3D Body crop/keypoint feedback

`src/body_feedback.cpp` adapts `_full_to_crop`, `keypoint_token_update_fn` and
`keypoint3d_token_update_fn` from pinned
[sam3d_body.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/models/meta_arch/sam3d_body.py).
Changes: bounded F32 GGML affine and learned projection graphs, native bilinear
sampling with zero padding and `align_corners=False`, checked indexing/masking
and token-range updates. This supports the DINOv3 coordinate convention; it does
not implement the alternate ViT horizontal-coordinate rescaling or hand-specific
callbacks. Native sampling currently runs on CPU, including in Vulkan sessions.

`reference/capture_body_feedback.py` executes unchanged original method ASTs,
including `BaseModel._flatten_person`, with original FFNs and PyTorch's grid
sampler. Non-replacing observers verify that captured and unobserved outputs
agree exactly. `tests/fixtures/body-feedback.txt` contains the resulting synthetic
regression; its adjacent JSON records provenance. Geometry is supplied, not
generated by MHR. These adaptations remain under the [SAM License](LICENSES/SAM.txt).

## SAM 3D Body crop geometry

`src/body_hand_crop.cpp` additionally adapts `_get_hand_box` and the mirrored
left-hand crop preparation from the pinned `SAM3DBody.run_inference` method.
Changes are checked one-person C++ array operations, bounded image storage,
explicit row-stride support and reuse of native crop/camera preparation with
the original hand padding. `reference/capture_hand_crop.py` executes the
original box method and selected original flip/crop statements, then original
transforms. The small `tests/fixtures/hand-crop.txt` contains synthetic image
inputs and fingerprints of their original outputs; full direct comparisons,
including the official photograph's trained hand boxes, remain generated data.
This is preprocessing, not the hand decoder or full refinement merge. Copyright
(c) Meta Platforms, Inc. and affiliates; [SAM License](LICENSES/SAM.txt).
See [hand refinement evidence](reference/HAND_REFINEMENT.md).

`src/body_hand_frame.cpp` adapts the wrist-frame and mask blocks of
`MHRHead.mhr_forward` from the same pinned SAM 3D Body source, under the
SAM license. Its quaternion composition, axis-rotvec conversion and matrix
expansion follow RoMa 1.6.1 (NAVER Corp., BSD-3-Clause); matrix-to-Euler
conversion shares the RoMa/SciPy adaptation already attributed for
`src/body_pose.cpp`. See `LICENSES/RoMa.txt` and the rotation notices above.
Changes: checked bounded C++ F32 batches, diagnostic taps, explicit parameter
index validation and no Python runtime. `reference/capture_hand_frame.py`
executes the original selected blocks and original RoMa calls. The normal
`tests/fixtures/hand-frame.txt` contains only synthetic inputs/buffers and
their original outputs; checkpoint-derived captures remain generated data.
The optional internal hand configurations in `src/body_pose.cpp` and
`src/body_output.cpp` compose those same original transfer/mask operations in
`MHRHead.forward`/`mhr_forward` order, including the original translation scale,
returned-versus-deformation global rotation distinction and output axes. The
trained capture `reference/capture_hand_head.py` executes the original class
and geometry asset, with all head buffers verified against safe source bytes;
it does not rewrite the neural head or geometry as an oracle.

The hand branch in `src/body_flow.cpp`, `src/body_condition.cpp` and
`src/body_prompt.cpp` additionally adapts original `forward_decoder_hand`,
`camera_project_hand` and its 2D/3D keypoint feedback callbacks from the same
pinned SAM 3D Body source (Meta Platforms, Inc. and affiliates; SAM License).
It preserves separate hand neural modules, shared sparse prompt weights,
independent dense hand positional encoding and hand camera scale factor 10.
Changes: checked C++/GGML composition, explicit buffer/index contracts and
operation diagnostics. The reference executes unchanged original methods with
verified safe trained state. No third-party C++ inference engine was copied.
Hand-image composition in `src/body_pipeline.cpp` and the reviewed
`reference/body_image_pipeline.py` follows the same original
`forward_pose_branch` with the hand batch selected: padding-0.9 crops, shared
no-mask embedding and shared post-decoder box heads. This is an explicit-ROI
component, not a substitute implementation of the complete refinement merge.

`src/body_crop.cpp` adapts the semantics and float-rounding boundaries of
`bbox_xyxy2cs`, `fix_aspect_ratio`, `get_warp_matrix` and `TopdownAffine` from
Meta's SAM 3D Body, revision `b5c765a0d89d789985e186d396315e7590887b94`:

- [bbox_utils.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/data/transforms/bbox_utils.py)
- [common.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/data/transforms/common.py)

Copyright (c) Meta Platforms, Inc. and affiliates. SAM materials and derivatives
are subject to the [SAM License](LICENSES/SAM.txt). Changes: C++23 port,
opaque C interface, checked inputs/outputs and no native OpenCV dependency.
The original direct affine solve was replaced with the operation ordering of
OpenCV's small-system LU solve after pixel-level tests exposed rounding differences.
Current supported path uses the standard non-UDP crop and two-stage aspect
expansion; it does not implement every optional upstream transform mode.

`reference/capture_body_crop.py` loads the hash-verified original upstream
`bbox_utils.py` directly for operation fixtures. It does not substitute a
rewritten Python oracle. Imported upstream source remains outside source control.

The third-party C++ engines listed in `reference/AUDIT.md` have not been copied,
built or executed. Existing MotionBricks and SkinTokens projects informed the
validation/error-boundary process; their code is not copied into this crop port.

## OpenCV affine sampling conventions

`src/body_image.cpp` follows the fixed-point coordinate/interpolation conventions
of [OpenCV 4.11.0 imgwarp.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/imgwarp.cpp).
The file's copyright and license are retained in
[LICENSES/OpenCV-imgwarp.txt](LICENSES/OpenCV-imgwarp.txt).
The adaptation supports RGB U8, bilinear interpolation and a constant black
border; SIMD, OpenCL, IPP and other interpolation/border modes are not copied.
OpenCV is used by the Python reference only, not linked by the native library.

`src/body_crop.cpp` adapts the 6x6 partial-pivot elimination order of
[OpenCV 4.11.0 matrix_decomp.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/matrix_decomp.cpp)
and the affine system assembly in `imgwarp.cpp`. The LU source is governed by
the included [OpenCV Apache 2.0 license](LICENSES/OpenCV-Apache-2.0.txt).
This preserves floating-point behavior at downstream pixel-coordinate rounding
boundaries; it does not import OpenCV's backend dispatch or general matrix API.

The Body image scaling/normalization contract follows Meta's
`BaseModel.data_preprocess` and the estimator's torchvision `ToTensor` path,
at the same pinned SAM revision above. The reference calls those original
methods; mean/std constants identify the selected DINOv3 Body configuration.

`src/body_pipeline.cpp` composes these native stages according to the original
`SAM3DBody.forward_pose_branch` at that revision. The reference helper
`reference/body_image_pipeline.py` executes its unchanged method AST, original
transforms, backbone wrapper and DINOv3 factory. Reference-only hooks stream
original backbone parameters and observe outputs; a device bridge accommodates
the original hardcoded CUDA ray calculation. These do not replace neural
operations. The official `dancing.jpg` example is hash-identified but remains
outside source control. [BODY_IMAGE_FLOW.md](reference/BODY_IMAGE_FLOW.md)
documents the synthetic-weight scope and the CPU-reference device limitation.
SAM-derived material remains governed by the included SAM License; original
DINOv3 factory/layers retain their separately documented attribution.

## SAM 3D Objects image/mask preparation

`src/objects_image.cpp` adapts `InferencePipeline.preprocess_image`, the default
preprocessor, mask-bound/crop/background and centered-pad semantics from Meta's
SAM 3D Objects revision `f91db411c50efee93d8db7aeb323885650f6f722`:

- [inference_pipeline.py](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/inference_pipeline.py)
- [preprocess_utils.py](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/preprocess_utils.py)
- [img_and_mask_transforms.py](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py)
- [img_processing.py](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/data/dataset/tdfy/img_processing.py)

Copyright (c) Meta Platforms, Inc. and affiliates. These adaptations remain under
the included [SAM License](LICENSES/SAM.txt). Changes: checked C++23 array
operations, bounded allocations, friendly degenerate-mask errors and an opaque
C API. Pointmap and learned image conditioning are not implemented by this stage.

The float antialiased bicubic filters follow the index/weight/accumulation
conventions in [PyTorch v2.7.0 UpSampleKernel.cpp](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp),
under the included [PyTorch license](LICENSES/PyTorch.txt). No PyTorch runtime,
general tensor iterator, SIMD dispatch or third-party C++ inference engine is
linked. `reference/capture_objects_image.py` calls hash-verified original ASTs
and PyTorch/torchvision transforms in isolation; it is not a rewritten oracle.
Original small outputs and capture provenance are retained in
`tests/fixtures/objects-image.{txt,json}`; original photograph/mask stay outside
source control. See [the validation scope](reference/OBJECTS_IMAGE.md).

The same native file additionally adapts the pointmap-aware
`resize_all_to_same_size`, `crop_around_mask_with_padding` and `rembg` from that
original transforms file. Changes: checked CHW/RGBA inputs, shared rectangular
bilinear-AA/nearest kernels, explicit NaN restoration and diagnostic outputs.
It preserves soft alpha, zero-versus-NaN padding distinctions and the original
joint-transform order. `reference/capture_objects_joint.py` executes the
original `PreProcessor._preprocess_image_mask_pointmap` with an explicitly
configured triple-transform sequence; this is not a verified published
checkpoint configuration. See [OBJECTS_JOINT.md](reference/OBJECTS_JOINT.md).

`src/objects_preprocess.cpp` additionally adapts the original
`PreProcessor._process_image_mask_pointmap_mess`, `_normalize_pointmap` and
`InferencePipelinePointMap.preprocess_image` at that same revision, under the
SAM License. The latter source is
[`pipeline/inference_pipeline_pointmap.py`](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/inference_pipeline_pointmap.py).
Changes: checked CPU span inputs, typed explicit transform options, native
composition, synchronous diagnostic callbacks and independently owned output
buffers. The original source methods produce the reference fixtures unchanged;
this is not a claimed full-model port. See
[OBJECTS_PREPROCESS.md](reference/OBJECTS_PREPROCESS.md).

## SAM 3D Objects point-window conditioning

`src/objects_fuser.cpp` separately adapts Meta's `EmbedderFuser` post-encoder
eval path and its Llama-style `FeedForward`, from
[`embedder_fuser.py`](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py)
and [`llama3/ff.py`](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/model/layers/llama3/ff.py),
under the SAM License. Changes: F32 GGML graphs, checked explicit shapes and
parameters, immutable caller embedding buffers, integer position-group IDs,
deterministic eval-only modality drop and optional operation diagnostics.
`reference/capture_objects_fuser.py` executes the unchanged original classes
behind explicitly supplied clone-only embedding boundaries. It does not claim
that these boundaries implement the original neural encoders. Provenance and
limitations: [OBJECTS_FUSER.md](reference/OBJECTS_FUSER.md).

`src/objects_point_condition.cpp` composes those adapted preprocessing,
PointPatch and fuser implementations using native-produced intermediate buffers.
`reference/capture_objects_point_condition.py` runs the unchanged original neural
classes on hash-verified original preprocessing outputs; it never supplies those
outputs to native inference. Source licenses and pins are unchanged. Its
point-only synthetic configuration and the retained original remapping-domain
failure are documented in [OBJECTS_POINT_CONDITION.md](reference/OBJECTS_POINT_CONDITION.md).

`src/objects_pointpatch.cpp` adapts Meta's `PointPatchEmbed` and `PointRemapper`
from the same pinned Objects revision, in
[`model/backbone/dit/embedder/pointmap.py`](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/model/backbone/dit/embedder/pointmap.py)
and its adjacent `point_remapper.py`. Copyright (c) Meta Platforms, Inc. and
affiliates; the included [SAM License](LICENSES/SAM.txt) applies.

The window-transformer semantics additionally follow the original `Block`,
`Attention` and `Mlp` in [timm 0.9.16](https://github.com/huggingface/pytorch-image-models/tree/v0.9.16),
the exact dependency pinned by Objects. Copyright 2019 Ross Wightman;
the original [Apache-2.0 license and attribution](LICENSES/timm-Apache-2.0.txt)
are retained. Changes: C++/GGML F32 graph, bounded chunked window execution,
checked parameter/mask/remapping contracts and streamed diagnostics. Coordinate
remapping and layout assembly are native CPU operations. Training/stochastic
dropout is not implemented; deterministic evaluation/forced dropout is.
No third-party C++ inference engine was copied.

`reference/original_pointpatch.py` hash-verifies the official timm wheel and
executes unchanged selected original class/function ASTs only inside the
isolated reference container. `capture_objects_pointpatch.py` records original
operations and independently runs the uninstrumented module. The checked-in
fixture has synthetic state, not learned weights. See
[OBJECTS_POINTPATCH.md](reference/OBJECTS_POINTPATCH.md) for validation scope.

## SAM 3D Objects scale/shift normalization

`src/objects_ssi.cpp` adapts `SSIPointmapNormalizer`, `ObjectCentricSSI`,
`ObjectApparentSizeSSI`, `NormalizedDisparitySpaceSSI`, `_apply_metric_to_ssi`
and the normalization methods of `ScaleShiftInvariant` from the pinned Objects
`data/dataset/tdfy/{img_and_mask_transforms,pose_target}.py`. Its disparity
remapping follows the same `PointRemapper` attributed above. Copyright (c)
Meta Platforms, Inc. and affiliates; [SAM License](LICENSES/SAM.txt).

The homogeneous row-vector transform convention and scale/translation/inverse
composition follow original [PyTorch3D transform3d.py](https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/pytorch3d/transforms/transform3d.py),
at the exact revision pinned in Objects' `requirements.p3d.txt`. Copyright (c)
Meta Platforms, Inc. and affiliates. All rights reserved. Its full
[BSD license](LICENSES/PyTorch3D.txt) is retained and installed.

Changes: bounded C++ arrays, native median/quantile/statistic calculations,
checked scale/shift/domain/overflow errors, original per-mode override and
clipping behavior, and optional diagnostics. There is no PyTorch3D runtime or
general rotation/pose engine in the native library. The reference loader executes
unchanged selected original ASTs, with postponed type annotations to avoid unused
pose/rotation imports. The original transform methods, not a replacement scalar
normalizer, produce the oracle. See [OBJECTS_SSI.md](reference/OBJECTS_SSI.md).
