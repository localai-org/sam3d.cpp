# Actual trained backbone operation traces

This continues [the trained checkpoint work](TRAINED_BODY.md), not full Body,
Objects, browser or performance acceptance.

## Real math-SDPA observation

`inspect_math_sdpa.py` observes the **unmodified** PyTorch math-SDPA call on
original trained Q/K/V from blocks 0, 1 and 22. On CPU and CUDA, observation
leaves its output byte-identical to the original block's attention output.
The actual `aten._safe_softmax` input/output are captured, not reconstructed.

PyTorch scales **both operands** by `sqrt(1/sqrt(head_dimension))` before their
matrix product. This formula matches observed logits/probabilities byte-for-byte
on both backends. The old auxiliary `(q @ k.T)/sqrt(head_dimension)` formula does
not: block-0 logits differ by up to `0.00244140625`, probabilities by `0.000295654`
on CPU and `0.000243753` on CUDA. Native DINO now uses the observed operand-scaling
order. Old auxiliary reports are not relabelled as real SDPA observations.

Report hashes:

- CPU: `56f609f87ade86322f61218c6d89f42d66fdb4e5e3cacccbd80c480d2c9295e5`.
- CUDA: `eccd4c9d23088536fc6f0f9926b22ec2c2917d0219a22ba18ec074619af2382a`.

## Full own-intermediate tracing

`capture_trained_backbone.py --operation-traces` attaches
`observe_trained_dino.py` to all 32 original blocks. Each streams 22 real module,
layout or actual SDPA boundaries to a safetensors file: 704 tensors overall.
Both original CPU/CUDA captures are complete, pass three unobserved repeats,
and match the unobserved final result exactly with observation active. Only
the image and model state enter the backbone; hidden states are not injected.
Streaming bounds memory to one block's observations.

Native has a separate optional per-block observer, enabled by
`run_patch_capture.py --block-traces --gguf ...`. It also propagates its own
intermediates from image and GGUF, writing 22 F32 buffers per block. A normal
CPU sanitizer test verifies observer coverage/order and byte-identical final
output with and without tracing.

`check_trained_operations.py` checks identities, complete 32×22 coverage, exact
file lengths/layouts, finite values, max-absolute and relative-L2 errors. It maps
one native block at a time and rejects missing/truncated data. Run comparisons
with `OPENBLAS_NUM_THREADS=1` alongside inference to avoid oversubscribing CPU
reduction workers; this does not change native inference threads.

## Current acceptance

- CPU and NVIDIA Vulkan each pass **704/704 own-intermediate operation checks**
  and all 36 stage checks under the frozen original-control policies.
- The original four-part Vulkan accumulation passed 703 operations, failing
  block 5's second LayerScale: `0.000640869` versus its `0.000610352` absolute
  budget. Eight partial dot products with balanced reduction fix this without
  changing any budget. This is not yet a performance-optimized implementation.
- Traced and untraced Vulkan stage outputs are byte-identical (SHA-256
  `ceee2ff141230e627cb497291dcb53f2808e3a14edc45b9630e4f5dd6bdcf1fb`).
- All 88 original synthetic Vulkan operation checks still pass. Normal tests
  pass: 35 native ASan/UBSan/LSan tests and 69 Python tests. NVIDIA uses UBSan;
  the documented driver/ASan initialization incompatibility still applies.

Frozen operation policy SHA-256:
`bded7ec65145f7d94fbdb4bf1ea3731ab4e8dfc4296bedbcf9f346e65609abb8`.
CPU operation report: `8e2b57bdd4d68fe1982c20a0d2db3b9cf8f254b73724713abf3c2bca235aa4a7`;
Vulkan: `86f4730f518a1194e9d21719e67e796f270f3af3c98b620eb52b6d33e4bb8383`.
This closes the trained backbone gate, **not** the trained decoder, geometry,
hand-refinement, complete image inference, demo or performance gates.

Operation-policy generation consumes only original CPU/CUDA traces, using four
times their measured discrepancy with the old floors and relative-L2 ceiling
`1e-3`. Generic activation absolute budgets are capped at `1e-4` of the reference
peak, with a `1e-4` floor. Softmax probabilities have a separate absolute ceiling
`0.01`: the generic ceiling rejected original cross-backend softmax rounding.
The actual per-boundary budget is four times the original discrepancy (with the
old floor), clipped to the absolute ceiling, and still requires both metrics.
If the original discrepancy itself exceeds a ceiling, calibration fails.
Thresholds do not adapt to native failures.
Tests cover this distinction, invalid controls, gain errors and local spikes.
This is a bounded activation policy, not a statistical guarantee or a change to
pixel, pose, metre or final-output requirements.

Freeze generated budgets before applying them to native traces, retain old
failures, and record policy/reference/native artifact identities. Stage checks
and isolated blocks never substitute for all-operation or end-to-end gates.
