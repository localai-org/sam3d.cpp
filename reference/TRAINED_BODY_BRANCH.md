# Trained Body image-to-mesh branch

This is the actual released DINOv3 Body checkpoint, not the earlier synthetic
SAM integration. It is **F32 body-branch inference**, not published BF16
execution, automatic detection, hand-crop refinement or the complete estimator.

## Inputs and reference

The pinned official `notebook/images/dancing.jpg` is decoded to RGB, with the
explicit box `[600,80,1330,1250]` and diagonal-image-length focal assumption used
by the existing image fixture. No detector output or upstream hidden state is
injected. Original crop/preparation, streamed trained DINO, camera/prompt
conditioning, all six decoder layers and their own MHR feedback, and hand-box
heads execute in the reviewed offline container.

`trained_body_state.py` checks the configuration and safe-state hashes and
loads **all 339 state tensors** belonging to the constructed original branch.
Missing tensors, shape/type mismatches and nonfinite weights fail; no random
initialization survives. The real MHR TorchScript remains separately verified.
Training-only modules and the separate hand-refinement branch are outside this
capture. `capture_body_flow.py --trained-extraction ... --trained-config ...`
uses the checkpoint's 1024-wide decoder MLP and heads, disabled reverse
image attention, enabled hand tokens and intrinsic-center condition.

The no-mask path is not a zero embedding: the trained 1280-channel
`prompt_encoder.no_mask_embed.weight` is added to backbone features upstream.
The native branch now does this too. Original mask convolutions still execute
in the reference; their result is unselected because mask scores are absent.
Native no-mask inference computes only the selected branch. Supplied-mask
inference is not implemented by this path.

Both CPU/CUDA original captures repeat exactly and their observed/unobserved
final outputs are byte-identical. CPU reference camera rays still use the
original hardcoded CUDA calculation before transfer; **this is not a CPU-only
performance baseline**. All neural and MHR computation in that case uses CPU.

## Comparisons and current limits

- The first Vulkan capture passes 442 comparisons: 325 boundaries plus 117
  independently returned output fields, including all six meshes, cameras,
  keypoints, rotations and final hand boxes/logits.
- Adding real observations inside all six decoder layers extends this to
  **646 passing Vulkan comparisons**, including 204 decoder operation tensors.
  SDPA logits/probabilities are captured from the actual math implementation,
  not reconstructed attention formulas. Final observation remains byte-exact.
- The initial CPU capture passes all final outputs but fails four internal
  centimetre-valued MHR vertex boundaries: `0.000106812 cm` versus the unchanged
  `0.0001 cm` limit. These are retained failures, not rounded into passes.
- Final vertex max error in the initial runs is `9.53674e-7 m` on Vulkan and
  `1.07288e-6 m` on CPU. Matching final vertices alone is not the whole gate.
- Decoder math-SDPA operand scaling has since been aligned with the actual
  original operation order. Vulkan still passes all 646 checks; its final
  vertex error is `8.34465e-7 m`. CPU passes 640/646, including all 204 decoder
  operation tensors. Remaining failures are four internal centimetre-vertex
  comparisons (worst `0.000122070 cm`) and layer 1's vertex-pixel projection,
  counted as both a tap and an independent field (`0.001159668 px` versus
  `0.001 px`). CPU's selected last-layer final output fields pass.
- All 290 earlier synthetic Vulkan decoder checks still pass, including masks
  and reverse attention. Normal tests pass: 36 native sanitizer and 72 Python.

Corrected Vulkan report SHA-256:
`4834888c8a389692028ecafd7070c4cd76df6405d291881ad15c170d58ac9643`.
Original full-output SHA-256s: CPU
`2d2d772fa401ffb65360085cd9911f03d44d8ed856c68d922076b05eb378386c`,
CUDA `33628ff23fbe56b9a2eaf4f4ea899afa5a572b43792d32a6caf3bf3a06d87d85`.
The separately captured compact and decoder-traced original final tensors are
also byte-identical on each backend.

### CPU geometry diagnostic (not acceptance)

`diagnose_trained_geometry.py` replays each layer's original and native-produced
pose/shape/expression parameters through original MHR. The original-input replay
is byte-identical to the original full branch at every layer. With identical
native inputs, native-versus-original MHR stays below `0.0001 cm` at every layer
(worst `0.000091553 cm`). At layer 1, using native inputs in **original** MHR
already produces `0.000122070 cm` discrepancy against the original trajectory.
Thus the remaining error includes propagated pose-input rounding; changing
skinning alone cannot explain or necessarily fix it. This intervention is for
localization only and does not replace the failing own-intermediate gate.

A further CUDA reference run verifies all 339 loaded state tensors by exact
byte readback and records the four capture-helper hashes. Both its full outputs
and all 529 observed tensors are byte-identical to the earlier capture.

`check_trained_body_branch.py` uses the **already frozen** trained DINO stage
policy only after all 36 original backbone tensors are verified byte-identical
to its original control cohort. The raw pipeline reports using obsolete
synthetic backbone limits remain retained. No new limits are calibrated from
these native runs. Geometry/head limits remain `1e-4` absolute and `2e-5`
relative L2; pixel/box quantities retain `1e-3` absolute. The established Euler
unit-circle policy remains separate. Normal tests check that backbone budgets
cannot leak into mesh or other final-output limits.

## Reproduction

Use the verified weights and reviewed reference image from
[TRAINED_BODY.md](TRAINED_BODY.md). Capture with `capture_body_flow.py` using
`--full-width --image-pipeline --trained-extraction` and `--trained-config`;
add `--decoder-operations` for real per-operation taps. Every reference starts
from the original RGB sample, not a normalized-input or feature fixture.

Run the native comparison (the module/device flags select CPU or Vulkan):

```sh
uv run --project reference/python --frozen python scripts/run_body_flow.py \
  --runner build/debug/bin/sam3d-body-flow-capture \
  --module build/debug/bin/libggml-cpu.so --backend CPU --threads 6 \
  --gguf models/mhr-lod1-f32.gguf \
  --backbone-gguf models/body-dinov3-f32.gguf \
  --reference generated/fixtures/body-trained-decoder-operations-cpu \
  --output generated/fixtures/body-trained-decoder-operations-native-cpu
uv run --project reference/python --frozen python scripts/check_trained_body_branch.py \
  --reference generated/fixtures/body-trained-decoder-operations-cpu \
  --candidate generated/fixtures/body-trained-decoder-operations-native-cpu \
  --backbone-reference generated/fixtures/body-trained-backbone-cpu \
  --report generated/fixtures/body-trained-decoder-operations-native-cpu/parity-trained-v1.json
```

The first command deliberately retains its old generic diagnostic limits and
can exit nonzero on the trained backbone. The second validates cohort identity
and recomputes the full comparison with the frozen backbone policy. It does not
turn non-backbone failures into passes. Reports are not overwritten.

## GGUF-only runtime

The internal `body_model` now loads three checked GGUF components: DINOv3,
the trained pose-branch companion and MHR. The new `sam3d-body-model-capture`
runner receives only raw RGB pixels, dimensions, box and intrinsics in its
input file. It receives no inline model parameters or reference hidden states.
The companion retains 313 F32 tensors and three checked I32 index tensors,
including the original hand-index partitions and mesh faces. Its 316 shapes
agree exactly between the Python converter and native contract.

The first GGUF-only Vulkan capture passes **646/646** unchanged checks. Its
entire 529-tensor output is byte-identical to the inline-weight Vulkan capture:
`a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`.
The GGUF-only comparison report SHA-256 is
`90c262960df95adef2cbc7d75a81c9e219806bf80eb5c4ca336b5b7d7d6bccf4`.
This does not close the outstanding CPU failures or prove hand refinement.

See [GGUF conversion](GGUF.md) for the companion command. Run
`scripts/run_body_model.py --help` for the capture arguments, then use the same
`check_trained_body_branch.py` comparison shown above. The input-extraction
helper is a reference harness; the C++ inference path has no Python dependency.
The session validates compatible checkpoint/config/geometry metadata and exact
mesh topology across its companions. Backend binaries and selected model files
must be trusted and immutable. The strict F32 Vulkan settings must precede
backend initialization; the session rejects known non-strict or externally
initialized Vulkan state whose precision cannot be verified. These are
process-wide initialization settings, not yet per-session precision controls.

Normal regression tests now pass 38 native ASan/UBSan/LSan cases and 79 Python
cases, including deterministic companion rejection and RGB-only extraction.
Vulkan continues to use UBSan due to the documented NVIDIA ICD/ASan conflict.

## Still required

A further fixed-token CPU diagnostic separates decoder input drift from pose
head arithmetic. It replays the original trained pose head with each of the six
original normalized tokens, then each of the six native tokens. Native pose-head
replays use these identical saved inputs. All **492 operation checks** pass;
original-on-original and native-on-native replays reproduce their full-run pose
inputs exactly. Differences already exist at the first head linear operation
(same-input maximum up to `1.9073486e-6`); both incoming-token drift and head
arithmetic contribute to the final pose parameters. This rules out a gross
head mapping mismatch on this case, but does **not** close any full-pipeline
geometry failure. The diagnostic deliberately injects saved tokens and is not
an acceptance run. Reproduce with `reference/capture_body_pose.py`'s three
`--trained-*` arguments in the isolated reference environment, then
`scripts/run_pose_diagnostic.py`; its manifest records the safe-state and
both full-capture identities.

The GGUF-only body pose branch is now exposed through the public opaque model
API. Its real pure-C Vulkan call returns all 19 fields byte-identical to the
accepted diagnostic capture; mesh topology separately matches the verified safe
state. Input and result lifetime checks run during that actual inference call.
See [API documentation](../docs/API.md) and
`generated/fixtures/body-c-api-native-vulkan/parity.json`. Current normal tests
are 40 native sanitizer and 81 Python cases; request/options and valid-result
getter fuzzers each pass 100,000 ASan/UBSan/LSan cases (seeds 932 and 933).

Close remaining CPU accumulated-geometry/projection gates. Add hand refinement and segmentation-mask prompt
cases, raw-image export, real browser upload/render QA, then optimized matched
performance comparisons. Objects trained inference remains a separate unfinished
track. None of these are implied by the Body branch result.
