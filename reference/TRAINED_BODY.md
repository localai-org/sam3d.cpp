# Trained Body checkpoint: extraction and current parity evidence

Official access was confirmed on 2026-09-09. The selected checkpoint and config
match the size/SHA-256 identities in `sources.json`. This is now **real learned
state**, not the synthetic state used by the earlier component fixtures.
Full trained Body reconstruction, hand refinement and the web demo remain
unfinished. No performance acceptance is claimed.

The trained backbone now passes all 704 operation checks on CPU/Vulkan; see
[operation evidence](TRAINED_OPERATIONS.md). The subsequent real RGB-to-body
branch passes 646 Vulkan checks, with six strict CPU geometry/projection checks
remaining; see [trained branch evidence](TRAINED_BODY_BRANCH.md).

## Safe extraction

`inventory_checkpoint.py` uses PyTorch `weights_only=True`, `mmap=True` and
`FakeTensorMode` inside the isolated, offline reference container. It imports no
upstream model. Body is a flat dictionary of 1,127 dense state tensors:

| Storage dtype | State elements |
| --- | ---: |
| BF16 | 840,755,216 |
| F32 | 112,333,138 |
| I64 | 221,642 |

These are **state elements**, not a count of learned parameters: the state also
includes buffers and index mappings. All 551 checkpoint backbone tensors are
BF16. `extract_body_checkpoint.py` exactly upcasts floats to F32, preserves I64
mapping values, and splits backbone state from the remaining 576 tensors. The
streaming writer opens its result with the independent safetensors reader and
checks every tensor's bytes against a fresh source view/upcast before publishing.
Source hashes are checked before and after extraction. Existing output files,
including dangling symlinks, are never overwritten.

Two important differences from the synthetic/factory schema were found:

- The checkpoint omits `backbone.encoder.mask_token`. Original pinned
  `DinoVisionTransformer.init_weights` initializes it to zero; extraction adds
  this one explicitly documented `[1,1280]` F32 tensor. The source file's hash
  and reason are recorded in `extraction.json`. No other missing state is filled.
- All 32 stored QKV bias masks are **entirely zero**, not the factory's
  Q=1/K=0/V=1 pattern. The first converter run correctly stopped at its overly
  restrictive schema check. Converter, archive and graph validators now accept
  either complete pattern, and preserve the actual values. The graph already
  multiplies the bias by its mask. A native regression verifies that an all-zero
  mask suppresses the stored biases identically to zero biases; malformed masks
  remain rejected. No tensors were rewritten to match factory defaults.

Verified artifacts (generated and Git-ignored):

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Backbone safetensors, 552 tensors | 3,363,078,840 | `98afde8ac5c13c68f3b7c7baf7cc8fa525df901fe1cec44b57d018fba11d53cc` |
| Other Body state, 576 tensors | 451,167,984 | `4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3` |
| F32 backbone GGUF | 3,363,058,592 | `9228c12b5b34cdb3627fce731a3e3d5890c54dc78ca7eb357d66056893f5bf1f` |

The native GGUF verifier reads and validates all 552 real tensors under
ASan/UBSan/LSan. All 35 normal native tests and 66 Python tests pass. GGUF remains
excluded from fuzzing; the streaming-writer tests cover exact mixed F32/I64
values, scalar/signed-zero state, noncontiguous inputs, changed readback, invalid
data, cleanup and output ownership.

## Reproduction

Use the reviewed reference image/environment described in `README.md`. Download
and run host byte/source preflight first. No host PyTorch checkpoint loading.
Mount the repository read-only at `/work` and a fresh output directory at
`/output`; run the container without network, privileges or credentials, with a
4 GiB memory limit for extraction. Container entry command:

```sh
python /work/reference/extract_body_checkpoint.py \
  --model-directory /work/generated/models/sam-3d-body-dinov3 --output /output
```

Then on the host, using the safetensors-only converter:

```sh
python scripts/convert_gguf.py \
  --input generated/extraction/body/body-dinov3-f32.safetensors \
  --manifest generated/extraction/body/body-dinov3-manifest.json \
  --output generated/models/sam-3d-body-dinov3/body-dinov3-f32.gguf
build/debug/bin/sam3d-gguf-verify \
  generated/models/sam-3d-body-dinov3/body-dinov3-f32.gguf
```

## Initial trained backbone comparison (failure history retained)

`capture_trained_backbone.py` uses the original pinned H+ factory and Body
wrapper with extracted real F32 weights. Its input is the hash-verified original
normalized 512×512 dancing crop from the earlier image-preprocessing reference.
This is a **backbone-component** test, not raw-image-to-body trained parity.
There are 36 boundaries: patch projection, token preparation, 32 block outputs,
final normalization and features. Each CPU/CUDA reference passes three identical
uninstrumented forwards and an exactly equal instrumented forward. Math SDPA,
disabled TF32 and F32 execution are explicit; this is not the published BF16
execution configuration or a performance benchmark.

The native runner reads the real GGUF, the normalized image and nothing else;
all 32 blocks propagate native-produced intermediate activations. Use
`run_patch_capture.py --gguf ...` with `sam3d-backbone-capture`.

Under the existing `max_abs=1e-4`, `relative_l2=2e-5` gates:

- Vulkan/CUDA first fails absolute error at block 1 (`3.35694e-4`); 31 of 36
  boundaries fail. Final features **pass**, max abs `1.68085e-5`, relative L2
  `1.54228e-6`. The largest intermediate absolute difference is `0.111328125`
  at block 22, with relative L2 `2.36994e-6`.
- Original PyTorch CPU/CUDA also fails the absolute gate: first at block 0,
  32 boundaries in total. Its largest absolute difference is `0.06640625`;
  final features pass at max abs `3.53456e-5`, relative L2 `1.56671e-6`.
- Native CPU sanitizer execution subsequently completed: under these original
  synthetic limits, its first failure is block 6, with 26 failing boundaries.
  Final features pass at max abs `2.21133e-5`, relative L2 `1.34846e-6`.

Reports are `generated/fixtures/body-trained-backbone-native-vulkan/parity.json`
and `generated/fixtures/body-trained-backbone-cpu/cpu-cuda-control.json`. Tolerances
in those reports remain unchanged. A passing final feature tensor does not waive
intermediate gates. The separately versioned trained policy below must not be
confused with these original failures.

## Rounding diagnosis and trained-only stage policy

`capture_trained_blocks.py` observes all 22 existing DINO operation/diagnostic
boundaries in real blocks 0, 1 and 22. The **same stored CUDA backbone input** is
supplied to each CPU/CUDA/native block: this is isolation, not a replacement for
the own-intermediate 32-block test. Original CPU/CUDA captures each pass three
uninstrumented repeats and observation-neutrality checks.

The first isolated native Vulkan threshold failure is block 0's QKV projection
(`1.220703125e-4` absolute, `5.466e-7` relative L2); its preceding LayerNorm passes.
`diagnose_trained_rounding.py` computes an explicitly diagnostic F64 linear
reference on each backend's **own normalized input**, separating norm propagation
from dot-product accumulation. Native Vulkan differs from that F64 result by
`1.22039e-4`; original CUDA by `4.54925e-5`, original CPU by `2.35744e-5`.
In block 22, CPU/CUDA/Vulkan LayerScale is byte-exact F32 multiplication. Its
learned gain reaches 13.625, magnifying prior projection differences; the output
magnitude reaches 31,459.7. At that magnitude, one F32 step already exceeds the
old `1e-4` absolute gate. This supports a trained-activation policy, not a change
to pixel, rotation or metre tolerances.

`scripts/calibrate_trained_dino.py` accepts **only original CPU/CUDA references**
with identical input/weight bytes, pinned source identities, F32/math-SDPA/no-TF32,
three repeats and exact observation neutrality. It has no native-candidate
argument. Each stage's budget is the larger of the old floor and four times the
original CPU/CUDA discrepancy, separately for max-absolute and relative L2.
Relative L2 is capped at `1e-3`; the absolute budget is capped at `1e-4` of the
reference peak (with a `1e-4` floor). Fourfold headroom is an explicit engineering
allowance for independent F32 reduction orders, **not a statistical guarantee**.
No thresholds adapt when a native candidate fails.

The frozen 36-boundary policy, controls and source hashes are checked in as
[`trained-dino-backbone-policy-v1.json`](trained-dino-backbone-policy-v1.json),
SHA-256 `cdc4bd14042edb87e43c9f0655e469e572874856c29a5776cec3d6c9f4364299`.
It was frozen **before** the native accumulation change below. Normal tests
verify its exact hash/formula and show that every stage rejects a 0.1% gain error
and a local over-budget spike. This policy is specific to the trained backbone
capture; it is not a blanket acceptance threshold for the remaining models.

Native CPU passes all 36 stage boundaries under this policy. The unchanged
Vulkan graph still failed blocks 11 and 12, so calibration alone did not pass it.
The strict-F32 Vulkan shader's long dot-product reductions were then split into
four GGML matmuls and a balanced sum for reduction widths at least 1024 and
divisible by four. The CPU graph is unchanged; operands and accumulation remain
F32, with no CPU fallback and no GGML submodule edits. The initial graph makes
contiguous slice copies: performance tuning remains a later acceptance task.

With this change, Vulkan passes all **36 stage boundaries** under the **unchanged
frozen policy**. Isolated block-0 QKV error falls from `1.22070e-4` to
`4.95911e-5`. Final feature errors are `1.54972e-5` absolute and `1.04123e-6`
relative L2, still below the original tighter final-feature limits.
The original 88 synthetic CUDA/Vulkan block-operation regression checks also
pass after this change without changing their rules.

Reports:

- CPU: `generated/fixtures/body-trained-backbone-native-cpu/parity-trained-v1.json`.
- Vulkan: `generated/fixtures/body-trained-backbone-native-vulkan-split4/parity-trained-v1.json`.
- Rounding diagnosis: `generated/fixtures/body-trained-rounding/diagnostics.json`.

**Remaining gate:** this establishes trained backbone **stage** checks on one
official crop, not all trained per-operation checks. The isolated block policy
attempt explicitly rejected the original softmax-probability control as too
large for its generic peak-relative cap. Those logits/probabilities are auxiliary
post-matmul diagnostics, not captured internal operations of PyTorch math SDPA.
Investigate its actual scaling/softmax sequence and extend trained operation
coverage to all blocks before declaring the full operation gate passed. The
decoder, full raw-image trained Body path, hands, browser and performance gates
are still open.

Actual math-SDPA order has since been observed and matched in the native graph,
and full 32-block operation traces added. See
[TRAINED_OPERATIONS.md](TRAINED_OPERATIONS.md) for current acceptance status;
the earlier failures above are retained as history.

## Objects inventory

The original `ss_generator.ckpt` restricted-loader inventory succeeds: 1,741
dense F32 tensors, 1,672,313,398 state elements. Its `state_dict` also contains an
empty `__mixhavior__` metadata dictionary, explicitly recorded as metadata and
not counted as weights. Both configured DINOv2 ViT-L/14-register encoders are
embedded in this checkpoint, along with the 512-channel PointPatch encoder,
fusion, generator and decoder state. Separate DINO weight fetching is therefore
not needed for this component. MoGe and the remaining dependency closure still
need resolution and verification.
