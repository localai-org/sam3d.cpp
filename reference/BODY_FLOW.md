# Complete Body decoder feedback composition

`src/body_flow.cpp` connects conditioning, every decoder layer, the pose head,
real MHR geometry, Body output mapping, camera/keypoint/vertex projection and
2D/3D feedback. It starts at backbone features and uses its **own intermediates**
throughout. Neither reference meshes nor reference camera/feedback tensors are
accepted by the composed entry point.

This is **composition parity with synthetic SAM parameters/features and the real
released MHR asset**, not trained SAM image-to-body parity. It does not unblock
the denied Body/Objects checkpoint access or provide a usable image demo.

## Control flow that must remain intact

The implementation follows the unchanged pinned `SAM3DBody.forward_decoder` and
`PromptableDecoder.forward`:

1. Construct initial/previous/prompt/keypoint tokens and camera-conditioned image
   features. Invalid prompts do not introduce a decoder attention mask.
2. Run each decoder layer with its evolving two-way image context. Only layer
   zero has `skip_first_pe`; image positional features remain fixed.
3. Normalize the layer output with the same affine `norm_final`, epsilon 1e-6.
   Use its **first token** for the pose and camera heads.
4. Every layer adds the **fixed learned initial pose/camera**, not the previous
   layer's predicted pose/camera. A supplied previous estimate affects the
   conditioning token, not these residuals.
5. Decode the full MHR mesh and skeleton, map Body outputs, and project both 70
   keypoints **and all 18,439 mesh vertices**. The vertex pass reuses the predicted
   camera without evaluating its FFN or adding its residual again.
6. Except after the last layer, update the **raw, unnormalized tokens** and token
   positional features. Sampling uses the **original camera-conditioned NCHW
   image**, not the mutable two-way attention context. Pelvis indices are 9/10;
   the default keypoint index lists are 0..69.
7. Return final normalized tokens and every layer's pose output. With hand-box
   tokens enabled, return the original two-token slice instead. No final-layer
   feedback is performed.

The supported composition is DINO Body with 70 2D and 70 3D keypoint tokens,
intermediate predictions and no external initial-estimate override. It rejects
unsupported configurations instead of silently substituting behavior. Separate
hand context, wrist-centric hand refinement and alternate ViT crops are not
implemented here. Geometry and sampling currently run on CPU between GGML
graphs, with repeated parameter transfer: this is not a performance result.

## Original reference and safeguards

`reference/capture_body_flow.py` runs the original decoder, camera/prompt modules,
heads and unchanged Body/BaseModel method ASTs in the reviewed container. The AST
extraction avoids unrelated backbone imports; it changes no method expressions.
The original MHRHead loads only the hash-verified public TorchScript asset through
its original loader (`MOMENTUM_ENABLED=0`). Neural/PCA/index/mapping state is
explicitly synthetic; the MHR state is never randomized.

Two cases use 8-wide synthetic neural layers, full geometry and two-way attention:

| Case | Batch | Layers | Hand-box tokens | Previous estimate | Repeat PE | Camera center |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 2 | No | Initial fallback | Yes | Image center |
| 1 | 1 | 6 | Yes | Explicit, different from initial | No | Intrinsics center |

Each has an invalid prompt and a labelled keypoint prompt. Camera/crop transforms
are asymmetric. The synthetic signed keypoint regression uses mesh vertices and
joints; it is **not** the learned semantic mapping and must never ship as one.

Non-replacing module hooks, source-frame locals and dispatch observations capture
372 boundary tensors across these trajectories. All returned tensor outputs from
an **uninstrumented** original run are also saved independently. Instrumented vs
uninstrumented runs agree exactly on CPU and CUDA for these fixtures. Native
comparison adds 154 checks against the uninstrumented outputs, rather than only
checking diagnostic taps against other taps.

Use the isolated procedure in [MHR.md](MHR.md), mounting a dedicated output
directory such as `generated/fixtures/body-flow-cpu`, with these entry arguments:

```sh
/work/reference/capture_body_flow.py \
  --model /work/generated/models/mhr-public/assets/mhr_model.pt \
  --upstream /work/reference/upstream/sam-3d-body \
  --roma-wheel /work/generated/wheels/roma-1.6.1-py3-none-any.whl \
  --output /output --device cpu
```

Capture CUDA independently with the documented GPU exposure and `--device cuda`.
Reference TF32 is disabled; SDPA uses its math backend. Source, model, wheel,
script and artifact hashes are retained in each manifest.

```sh
uv run --frozen python scripts/run_body_flow.py \
  --reference generated/fixtures/body-flow-cpu \
  --runner build/debug/bin/sam3d-body-flow-capture \
  --module build/debug/bin/libggml-cpu.so \
  --gguf generated/models/mhr-public/mhr-lod1-f32.gguf \
  --output generated/fixtures/body-flow-native-cpu --threads 12
```

For NVIDIA Vulkan, use that build's runner/module, the CUDA reference and
`--backend Vulkan --description 'NVIDIA GeForce RTX 5070 Ti'` on the tested
machine. Device index, description, thread count and driver selection are caller
configuration. The comparison runner enforces strict F32 Vulkan settings. The
ordinary CPU build retains ASan/UBSan/LSan; the documented driver-init exception
uses the separate Vulkan UBSan build. Native inference loads GGUF plus explicit
test parameter arrays; it has no Python or PyTorch runtime dependency.

## Results

Both backends pass **526 checks**. Limits established for these component units
are max-absolute 1e-4 (latents/geometry), 1e-3 (pixels/scaled box), and relative-L2
2e-5. No threshold was relaxed to obtain a pass.

| Worst error over all layers/cases | CPU vs PyTorch CPU | Vulkan vs PyTorch CUDA |
| --- | ---: | ---: |
| Normalized tokens | 2.72e-6 | 9.51e-6 |
| Updated raw tokens | 1.41e-6 | 4.00e-6 |
| Mesh vertices, meters | 4.77e-7 | 7.16e-7 |
| 3D keypoints, meters | 3.58e-7 | 4.77e-7 |
| Keypoint pixels | 1.07e-4 | 1.84e-4 |
| Vertex pixels | 1.38e-4 | 2.45e-4 |

Normal CTest adds an eight-case original `norm_final` regression and composition
shape/parameter-contract rejection checks. The adjacent JSON records provenance
for `tests/fixtures/decoder-norm.txt`. This small fixture isolates normalization,
**not** the full decoder. The full comparison requires the actual MHR GGUF.
The projection-only camera path is also compared exactly with the existing full
camera-head path. All 24 native ASan/UBSan/LSan and 29 Python tests pass.

| Tensor artifact | SHA-256 |
| --- | --- |
| Original CPU | `e59962693b05be1398120c5f467a801129abac36e991237bdd72ff301a45c0e3` |
| Original CUDA | `6c5a939b8c684299842b4b371f362a9a93dbf12e2cc94aef6c1bbb68d0429509` |
| Native CPU | `9ea5eb78e07c34b646bacacee34111e287b17232d3f73144a5068102701ef872` |
| Native Vulkan | `44c63cac96e5a163547ca05d2db6fcb8c1fbd2662bd7d4a28621273da6f193cf` |
| Normal norm fixture | `a05edb7031c09adb523ad908c0933d11f6454e634d2a56e43fae3c3634fe0452` |

## Full-width run: initial numerical failure and resolution

`capture_body_flow.py --full-width` now captures a separate six-layer case with
512x512 input metadata, patch 16, 32x32x1280 context, 143x1024 tokens, eight
64-dimensional attention heads and 4096-wide decoder FFNs. Pose and camera heads
use two-layer FFNs with hidden width 128. This tests the large architecture
contract, not the still-unavailable trained configuration/weights. Use separate
`body-flow-full-{cpu,cuda}` reference directories and
`body-flow-full-native-{cpu,vulkan}` outputs with the same capture/run commands.

Original instrumented vs uninstrumented outputs remain exactly equal on both
backends. Under the initial raw-relative-Euler criterion, CPU passed 385/389 checks and
Vulkan 383/389. Failures are `global_rot` at layers 0/3 on CPU and 1/4/5 on
Vulkan, duplicated in boundary and uninstrumented-output comparisons. Their
absolute error is 1.1920928955078125e-7 radians; relative-L2 about 2.62–2.65e-5
exceeds the unchanged 2e-5 limit. No tolerance was relaxed and no sample changed.
All other outputs pass, including full meshes (<=5.97e-7 m CPU / 7.16e-7 m Vulkan)
and projected vertices (<=1.68e-4 / 1.93e-4 pixels).

`reference/trace_body_rotation.py` isolates the original global rotation functions
on **both original and native-produced prediction vectors**. It does not run
heads or inject anything into the composed model. `sam3d-global-rotation-capture`
runs exactly the production native rotation helper on those same vectors;
normal CTest verifies diagnostic and production outputs are identical.
`scripts/check_body_rotation_trace.py DIRECTORY` writes dependency-ordered
exact-difference reports. This diagnostic reports observations, not model pass/fail.

The captured first differences are:

- CPU: normalized 6D axes, rotation matrix, quaternion, Euler intermediates
  `(a,b,c,d)` and `hypot` values are bit-identical on all 12 vectors. The first
  difference is `2*atan2(hypot(c,d),hypot(a,b))`, one F32 step at ~1.57 radians.
- CUDA: axes, matrix, quaternion and `(a,b,c,d)` also match exactly. First
  differences appear in `hypot`, one F32 step on four vectors. Middle angles
  differ by one step on six vectors.
- Subtracting F32 pi/2 leaves the middle angle near 0.00234, amplifying relative
  error. The failing CPU example is 0.0023403168 upstream vs 0.0023401976 native.
  This isolates scalar math rounding, not a different rotation convention or
  a runaway geometry/decoder feedback error. It does not by itself justify
  changing acceptance thresholds. A double-precision `atan2` followed by F32
  rounding was also checked and does not reproduce the CPU original in the two
  failing cases; no such speculative replacement was made.

At that point inference math and criteria were left unchanged, and the gate was
recorded as failed. The initial failure reports remain preserved separately from
the revised-policy reports below.

An additional **upstream-vs-upstream** control feeds byte-identical vectors to
original PyTorch CPU and CUDA (`body-rotation-full-cpu` versus
`body-rotation-shared-input-cuda`). Five of the six original-prediction vectors
themselves exceed the same 2e-5 relative-Euler criterion: max-abs 1.1920928955078125e-7
rad and relative-L2 2.62–2.65e-5. Their first difference is `hypot`, after identical
quaternion intermediates. This independently establishes that the strict
near-zero criterion can reject original upstream across backends; it is not
evidence that native feedback is wrong. Preserve this control when defining
angular numerical-floor handling rather than tuning a threshold to native output.
Reproduce the report with:

```sh
uv run --frozen python scripts/check_body_rotation_trace.py \
  generated/fixtures/body-rotation-full-cpu \
  --compare-reference generated/fixtures/body-rotation-shared-input-cuda
```

The shared-input CUDA capture uses `trace_body_rotation.py --device cuda` with
the **CPU** full-flow original/native tensor files, not the CUDA full-flow files.
Its input bytes must match the CPU diagnostic input exactly; the comparison
rejects different inputs. Reports include hashes and per-vector absolute/relative
errors, and deliberately do not mark full-model parity passed.

| Full-width tensor artifact | SHA-256 |
| --- | --- |
| Original CPU | `b8af39b44219d29ba6674412470a4e75a9d39f07e063ce62c2967c3638d40932` |
| Original CUDA | `b02abb30af2016786bb60b338ca66709684bb1b97075a0628f82bc329252f4a8` |
| Native CPU, identical before/after policy change | `91b5d622f3d0ff39fee3ebcc2b9cbe79ff479b3b87c319fac0afeaa4e7167a48` |
| Native Vulkan, identical before/after policy change | `f74fffcd8ddbe64c5530fca2a2e05f6647145d2e61181546c94511750610607f` |

### Frozen angular policy and rerun

The initial numerical gate is now resolved using `euler-radians-unit-circle-v1`.
This is an explicit **metric revision**, not a native math fix or a claim that
the old raw-relative criterion passes. `scripts/rotation_parity.py` retains the
raw maximum-absolute limit of 1e-4 radians, and raw Euler relative error is still
reported. The relative-L2 assertion, still 2e-5, compares each angle's `(sin,cos)`
unit-circle representation, evaluated in F64 by the checker. Its reference norm
is always the square root of the coordinate count, independent of how near the
rotation is to identity. No tuned small-angle denominator or native-dependent
epsilon is used. The raw absolute limit still rejects wrapping and large errors;
the unit-circle bound rejects meaningful small-angle differences as well.

Only global Euler outputs use this policy. Other tensor thresholds and native
computation are unchanged. The policy is frozen against six **original-only**
CPU/CUDA control vectors in `rotation-policy-v1.json`, generated by
`scripts/calibrate_rotation_policy.py`. Native-produced diagnostic vectors are
explicitly excluded from calibration. Six normal tests check that provenance,
original roundoff, zero rotations, and rejection of material errors, swapped
axes, wrong units, wrapping and malformed tensors.

Fresh full-width runs in `body-flow-full-calibrated-{cpu,vulkan}` each pass
**389 checks** (274 boundaries plus 115 uninstrumented output comparisons).
Angular unit-circle relative error is <=6.883e-8 on both backends. Native tensor
artifact hashes are **identical** to the old failed runs listed above: the change
affects only measurement/acceptance, not values. Reports hash the policy code and
retain raw-relative-angle errors for audit. Original failed reports are not
overwritten. CPU ASan/UBSan/LSan and Vulkan UBSan remain enabled.

`src/body_pipeline.cpp` now composes RGB preparation, camera/CLIFF inputs, the
native backbone and complete decoder without injected activations. This separate
internal static target is not a public model-session API. Its square-crop,
single-person geometry preparation is checked against original camera fixtures;
the complete RGB-to-geometry neural chain **has not yet been compared** against
the original image pipeline. Compilation or this preparation test does not prove
that next composition gate.

Remaining: native image/backbone-to-decoder
composition, actual trained checkpoint/config/mapping validation, hand refinement,
public inference API, Objects, real upload/render/export Chrome QA and optimized
CPU/CUDA/Vulkan end-to-end performance. None is satisfied by this fixture.
