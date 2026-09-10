# Experimental GGML patches

Upstream stays pinned and pristine at `ggml-org/ggml`
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` (0.23.0).
`SAM3D_EXPERIMENTAL_GGML_PATCHES=ON` applies these reviewable patches to a
content-addressed copy below the build directory. No submodule edits or private
commits are required. Default builds use unmodified GGML. Configure requires Git
and the initialized submodule; stale/inapplicable patches fail closed.

`0001` adds ordinary scalar F32 matrix-multiplication pipelines alongside NVIDIA
cooperative-matrix-2 BF16 pipelines. It avoids GGML's implicit F32-to-F16 operand
conversion for F32×F32 matrix multiplication, including noncontiguous copies.
The scalar shader and tile configuration come from the same pinned upstream
Vulkan implementation (MIT; see GGML's license). It preserves the original
non-CM2 path. `MUL_MAT_ID` and CM1 are outside this narrow capability contract;
Body does not use `MUL_MAT_ID`. Actual BF16 CM2 support is checked after device
initialization; a CM1 fallback is rejected. The KHR cooperative-matrix extension
must stay enabled because CM2 depends on its feature declarations too.

Testing selection: `SAM3D_BF16_COOPMAT2=1`, `GGML_VK_DISABLE_F16=1`, with
`GGML_VK_DISABLE_COOPMAT` and `GGML_VK_DISABLE_COOPMAT2` unset. The runtime
requires an explicit backend capability handshake; these variables are not
sufficient to make an unpatched backend safe. This is not yet an accepted
default. The separate experimental `SAM3D_BF16_FLASH_ATTENTION=1` switches only
the BF16 encoder to GGML fused attention; math-logit operation capture is rejected
in that mode because a fused kernel cannot expose those internal values.

The build-copy approach is adapted from skin-tokens.cpp's `PrepareGGML.cmake`,
with content-addressed destinations in place of recursive directory replacement.

`0002` adds an opt-in `GGML_VK_FUSE_BF16_ROUND=1` fusion for a contiguous
F32 → BF16 → F32 cast pair. One shader retains the exact original integer
round-to-nearest-even conversion, including signed zero, subnormals, ties and
finite overflow. It does not remove the BF16 rounding boundary. The intermediate
must be a fresh, unobserved cast with just one consumer; shared/explicit-output
intermediates and strided inputs use the original two kernels. Existing backend
overlap checks still apply. The shader and dispatch machinery are adapted from
the pinned GGML `contig_copy.comp` and Vulkan implementation, under its MIT
license. The flag is off by default and does not affect F32 graphs.

The model-free `sam3d-vulkan-precision-test` checks 12 strict F32 cases and
24 bit-exact BF16 cast cases on the actual device. With `GGML_VK_PERF_LOGGER=1`,
eligible cases must show `BF16_ROUND`, while protected intermediates retain
separate `CPY` operations. Full-model output and encoder-boundary regressions
are also required; synthetic kernel agreement alone is not model parity.

`0003` preserves accumulator precision for the BF16 CM2 flash-attention
softmax denominator. The original shader sums already-rounded BF16
probabilities with a matrix product against ones. The patch instead reduces
the F32 probabilities before their BF16 conversion for the numerator's P×V
product. Padding remains cleared before either operation; non-BF16 shaders
are unchanged. This follows the precision split in
[FlashAttention's original softmax implementation](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/softmax.h).
The implementation adapts the pinned GGML shader's existing cooperative-matrix
reduction, under GGML's MIT license; no third-party engine is executed.
Three model-free denominator tests distinguish this behavior from the previous
rounded sum, with signed values, multiple heads and padded key tails. See
[numerical results](../../reference/BODY_PRECISION.md#bf16-fused-attention-denominator-correction)
for trained-operation/full-trajectory checks and the remaining rider projection
failures; this patch is not an accepted demo default.

An unshifted-softmax experiment is retained only in
`reference/experiments/rejected-bf16-unshifted-softmax.patch` and is **not applied**.
It passed focused tests but regressed full encoder/final-output checks.

`0004` instead improves BF16 attention probability precision: after the usual
BF16 P×V product, it adds a second tensor-core product for the residual between
the F32 probability and its BF16 high part. The denominator remains F32.
This approximates the original math-SDPA numerator while retaining BF16 Q/K/V
and never materializing the full attention matrix. It does not claim bit-exact
F32 probabilities or change the model's BF16 rounding boundaries. Non-BF16
shaders are unchanged. The new code adapts the same GGML shader under MIT.
Eighteen model-free cases compare against mathematical softmax using constant
and varied signed V, padded key lengths, multiple heads and global logit shifts.
All 32 trained isolated attention checks pass; mean relative-L2 error improves
10× over denominator-only attention. This alone does not pass every complete
encoder-stage gate. The native graph's opt-in `SAM3D_BF16_PRECISE_PREFIX=1`
uses F32 math attention for the five class/register queries, retaining all
keys/values, and this patched fused attention for the 1024 image queries.
Together they pass all 36 stages on both dancer and rider and both final-body
policies (one dancer hand-logit check is explicitly non-blocking). Four focused
whole-block tests check exact prefix results, token/batch layout and dependence
on image keys/values. F32 behavior is unchanged. This is not an 80–85 ms claim;
see the current measurements and artifacts in `reference/BODY_PRECISION.md`.

`0005` adds the separately opt-in `GGML_VK_FUSE_BF16_BINARY=1` optimization
(also requires `GGML_VK_FUSE_BF16_ROUND=1`). An F32 ADD or MUL immediately
followed by the exact BF16 narrowing/widening pair can execute in one shader.
It retains the F32 operation and original RNE conversion, including the
rounding boundary; it does not combine arithmetic into an FMA. The producer
must be fresh, single-use, unobserved and contiguous, with contiguous F32
operands. Aliases, shared/observed producers or narrow results, strided inputs,
and intervening views retain the unfused operation. Existing backend overlap
checks remain in effect. Code adapts the pinned GGML `add.comp`, `mul.comp`,
binary dispatch and cast-fusion machinery under MIT.

The device precision test adds 192 cases, each run with fusion off and on,
against independent F32-then-BF16 scalar bits. `GGML_VK_PERF_LOGGER=1` plus
`scripts/check_bf16_binary_log.py` checks all 384 executions: 24 eligible
fusions and 360 disabled/refused paths. This prevents treating an ignored flag
as evidence. Complete-model comparisons remain required. UBSan and repeated
release alternating-image runs have exact baseline outputs; both complete
encoder captures and all 529 strict F32 tensors remain unchanged. All 646
F32 assertions pass. The demo selects this fusion only for its accepted
Vulkan BF16 mode; CPU and F32 selections strip inherited experimental flags.
See `reference/BODY_PRECISION.md` for measurements and real-browser QA.

`0006` adds opt-in `GGML_VK_BF16_BINARY_LINEAR=1` indexing for the already
guarded binary-round fusion. Contiguous, same-shape tensors use direct linear
addresses; a repeated single row or scalar uses its corresponding specialized
index. General multidimensional broadcasts keep the original shader. Inputs
and output must all be dense F32, and the output shape must match the first
operand. The exact F32 operation, RNE conversion, descriptor offsets, dispatch
tails and original fusion/overlap guards remain unchanged. The shader adapts
GGML's MIT-licensed add/multiply and generic binary helpers; it does not change
F32-only graphs. The device test adds 64 four-dimensional/offset/large-dispatch
cases, each compared with both the old shader and independent scalar bits.
Both complete encoder captures and all final outputs remain byte-identical;
the 646 strict F32 assertions also pass. Repeated release benchmarks measure
109.5–109.6 ms versus 111.3 ms with this flag disabled in the same binary.
The demo selects it only for Vulkan BF16. See `reference/BODY_PRECISION.md`.

`0007` is a default-off matrix-tile selector.
`GGML_VK_BF16_MATMUL_TILE=small|medium|large` selects an existing
NVIDIA CM2 BF16 matrix kernel instead of GGML's tile heuristic. Unset/unknown
values preserve the heuristic. The guard checks the actual BF16 pipeline:
the model uses F32 activation carriers which GGML converts to BF16 before
the multiply. F32/F16/quantized kernels and other architectures are untouched.
Existing alignment, tail handling and split-K logic remain in use. Optional
`GGML_VK_BF16_MATMUL_TRACE=1` proves selection but adds timing overhead.

`sam3d-vulkan-bf16-matmul-test` uses exact analytical dyadic products for
trained matrix shapes, odd dimensions, batching and both BF16/F32 carriers.
All candidates still require complete encoder and final-output comparisons
against the frozen policies; isolated timing or correctness is insufficient.
Small tiles retain byte-exact full outputs, both complete encoder captures
and all 529 strict F32 tensors. Repeated full inference measures 99.1–99.5 ms
versus 104.87 ms with the original heuristic in the same binary. The demo
overrides inherited tile choices with `small` only for NVIDIA Vulkan BF16
and strips tracing; CPU/F32 selections strip both variables. Medium/large
remain diagnostic choices, not accepted demo settings. This selector adapts the pinned
GGML Vulkan pipeline dispatch under its MIT license; no new shader is copied.

The rejected F32 matvec row-group experiment is archived in
`reference/experiments/rejected-f32-matvec-rows.patch`, not applied. It gave no
meaningful full-model improvement. In particular, the multi-column/multi-head
prefix product falls back to GEMM despite the generic profiler's matvec label.

`0008` instead explores `GGML_VK_F32_NARROW_MATMUL=1`, default off. It uses the
existing scalar F32 GEMM shader with a 16×8 output tile for N≤8, retaining its
hard-coded K=32 tile and ordered dot-product implementation. It disables split-K for this
tile to avoid a new reduction order. The new pipeline is available only on
NVIDIA CM2 with 32-lane subgroups and only for F32×F32 GEMM, not matvec,
BF16/F16/quantized or expert paths. `GGML_VK_F32_NARROW_TRACE=1` reports actual
selection. This adapts the pinned GGML MIT pipeline/shader machinery. Both
complete encoders, all 529 F32 operation tensors and full outputs remain exact;
all frozen comparisons pass. Repeated inference measures 90.8–91.1 ms versus
94.5 ms disabled in the same binary. The demo selects it only for Vulkan BF16
and strips tracing. See `reference/experiments/f32-narrow-matmul.md`.

`sam3d-vulkan-matvec-test` checks 56 signed dyadic/non-dyadic cases, odd/tail
dimensions, multiple head/batch layouts and independent double-precision
products. It saves outputs for exact process-to-process comparisons. Full
encoder and body/F32 policy checks and matched latency remain required.

`0009` adds narrow-tile exploration using the same ordered F32 arithmetic.
`GGML_VK_F32_NARROW_TILE=tiny32` selects an 8×8 output tile with the original
K=32 depth. The default/unrecognized setting retains 0008's 16×8 tile.
The `tiny64` and `deep64` controls use an added F32 specialization constant 14
(default 32) to request K=64; neither improved timing over `tiny32` and neither
is selected by the demo. F16/quantized shader reduction depths are unchanged.
All four variants pass the 56-case oracle and match all 25 full outputs exactly.
See `reference/experiments/f32-narrow-depth.md` for matched measurements and
the important correction concerning the first experiment's ignored K setting.

`0010` adds an experimental exact SiLU → BF16 roundtrip → multiply → BF16
roundtrip fusion, selected by `GGML_VK_FUSE_BF16_SILU_GATE=1` alongside the
round/binary flags. It preserves both RNE rounding boundaries and avoids FMA.
Only checked dense, single-consumer patterns qualify; observed intermediates,
broadcasts and unsafe source/destination overlaps fall back. The explicit
copy-destination variant allows the graph allocator to reuse the multiply
buffer safely. A related diagnostic correction clears rejected fusion labels
instead of reporting proposed work as executed work. GGML remains pristine;
this is applied solely to the content-addressed build copy.

408 device off/on cases, both complete trained encoders, full outputs, strict
F32 captures and CPU sanitizers pass. The full model demonstrably executes 32
fused gates. Repeated release timing is 88.25–88.29 ms versus 90.61 ms disabled
in the same binary, with all full results exact. The demo enables it only for
Vulkan BF16 after actual headless Chrome QA; see `reference/experiments/bf16-silu-gate.md`.

`0011` adds opt-in `GGML_VK_FUSE_BF16_AFFINE=1` scale/add fusion, preserving
either one final BF16 rounding or both product/final rounding boundaries.
Separate precise F32 multiplication and addition prevent FMA contraction.
Only checked dense leading-block broadcasts qualify; observed/shared values,
host buffers, padded layouts and unsafe overlaps retain the existing path.
The shader uses the pinned GGML MIT indexing, descriptor and BF16 conversion
helpers. All 1,248 off/on device cases, both complete encoder trajectories,
full body outputs and strict F32 comparisons pass. Actual traces show 162
fusions per image. Repeated release runs measure 86.68–86.98 ms, versus
88.03 ms disabled in the same binary, with all results exact. Demo integration
is still pending; the demo strips this experimental flag. See
`reference/experiments/bf16-affine.md`.

`0012` explores `GGML_VK_FUSE_BF16_NORM_AFFINE=1` alongside the affine flags.
It combines the original GGML NORM reduction with scale/add and final BF16
rounding. The precise affine operations are isolated in a function: applying
`precise` directly in the normalization body changed contraction throughout
the reduction and failed an exact device test. The corrected shader passes
720 off/on cases and all complete trained/F32 gates, with 64 actual fusions
per image and about 0.6 ms less GPU work. Repeated release runs measure
86.39–86.55 ms versus 87.12 ms disabled, with complete outputs unchanged.
Demo integration remains pending; the demo strips this flag until acceptance.
See `reference/experiments/bf16-norm-affine.md` for the failed case and fix.
