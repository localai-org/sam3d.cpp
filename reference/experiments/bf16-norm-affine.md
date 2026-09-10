# BF16 normalization/affine fusion candidate

The eleven-patch affine candidate reaches 86.68–86.98 ms release latency,
with complete outputs unchanged. Its GPU profile still has 65 encoder NORM
dispatches (about 0.939 ms) followed by separate affine work. This experiment
attempts to remove that intermediate memory traffic and dispatch overhead.

`0012-vulkan-bf16-norm-affine.patch` adds separately gated
`GGML_VK_FUSE_BF16_NORM_AFFINE=1`, requiring the existing affine/round/binary
flags. It targets NORM → MUL → ADD → BF16 narrow → F32 widen only. The shader
reuses the pinned GGML `norm.comp` reduction algorithm (MIT), then performs
separate F32 scale/add and RNE BF16 conversion. Observed/shared intermediates,
padded input layouts, host buffers and unsafe overlaps fall back. Reduction
input/output overlaps are rejected even when exactly equal. Dispatch row tails
are bounded before any shader memory access or workgroup barrier.

The new device suite has 720 off/on cases: six row widths, three row counts
(including a 513-row dispatch tail), ten coefficient/layout/observation modes,
two repetitions and independent/graph-reused allocations. It requires exact
device bytes and separately compares the mathematical operation to a double
oracle. Existing affine tests retain their rounding and FMA negative controls.

## First divergence and correction

Build-v1 completed with UBSan, peak 3,132,006,400 bytes and zero OOM/max events.
Device precision-v1 correctly rejected the first shader: width 7, 513 rows,
full-shaped bias, independent buffers, element 3253 changed from approximately
0.000622 to 0.000618. Profiling confirms an actual fused NORM dispatch.

SPIR-V inspection explains the divergence: declaring the final multiply/add
`precise` inside `main` propagates `NoContraction` backward through the entire
normalization computation (13 decorated operations), whereas the upstream
normalization has no such decorations. This changes reduction and variance
arithmetic despite retaining the same formula.

The correction places the two affine operations in their own function. A
standalone shader experiment confirms exactly two `NoContraction` decorations,
on the multiply and add only, both without and with shader optimization and
inlining. The normalization reduction remains undecorated like upstream.
`norm-affine-function.comp` preserves this compiler probe.

The corrected build-copy fingerprint is
`7b82a69595046481bd4b2142e9a83ae26d12e7dbd7a41bb700ed9114acc94b50`.
Build-v2 and device precision-v2 pass with UBSan: all 720 norm-affine cases
are byte-exact, including the previous failure, and 248 actual norm-affine
dispatches are observed in the unit section. Existing precision suites pass.

All 25 complete outputs in each of the enabled, disabled and GPU-profiled
runs are byte-identical to the accepted baseline. Both 36-stage trained
encoder captures retain their accepted hashes and pass frozen policies;
both final-body gates pass, with the existing dancer hand-logit miss still
reported. All 529 strict F32 operation tensors retain their accepted hash
and all 646 original F32 assertions pass. No tolerance changed.

Same-binary `-O1` UBSan warm medians (five warmups, twenty timed alternating
inputs) are **112.697 ms disabled / 112.271 ms enabled**. GPU profiling
confirms **64 norm-affine fusions/image**, totaling 0.925 ms. Total GPU time
falls from 68.038 to **67.441 ms**. This is a modest candidate gain, not a
release timing or achievement of the 80–85 ms goal.

Both 48-test normal suites (CPU ASan/UBSan/LSan and the rebuilt Vulkan UBSan
tree), 112 Python tests and Go race tests pass.

The first release-build attempt was rejected by the host-headroom guard
before starting. A separate `-O3 -g0` timing build is now attempting to fit
under 3 GiB; only debug information is omitted there, not sanitizers from the
correctness builds. Omitting symbols did not prevent soft-watermark stalls.
At 2026-09-10 03:32:24 UTC, only this build scope's `MemoryHigh` was raised
from 2,560 to 2,944 MiB. Its **3,072 MiB hard cap, zero swap and 10 GiB reserve
are unchanged**. The budget report records startup limits; the separate
`bf16-norm-affine-release-memory-adjustment-v1.json` records this verified
runtime adjustment. No unrelated process or global setting changed.
The same build then completed, with peak 2,871,508,992 bytes, no hard-limit
or OOM events, and 390.1 seconds total elapsed including the soft stalls.

## Release results

The `-O3 -g0` binary passes the complete device suite, including 720 exact
norm-affine cases. Five warmups and twenty timed alternating inputs give:

| Selection | Median | p95 | Range |
| --- | --- | --- | --- |
| Norm-affine off (affine remains on) | 87.121 ms | 88.248 ms | 86.714–89.886 ms |
| Norm-affine on | 86.546 ms | 87.483 ms | 86.195–87.521 ms |
| Norm-affine on, repeat | 86.393 ms | 87.023 ms | 85.940–87.326 ms |

All 75 complete release results are byte-identical to the accepted baseline.
The 25-request public-F32 UBSan run also matches every baseline result exactly.
This is a useful ~0.6 ms improvement, but the **80–85 ms goal remains open**.
Demo integration/Chrome QA of these two new fusions is still pending; the
working demo remains the accepted SiLU build.

## Further profile leads, not implemented

The resident stack creates 32 identical sin/cos pairs from one immutable
angle tensor. Whole-model SIN/COS time is only about 0.217 ms/image, so sharing
the pair would be a small dispatch optimization rather than a major target.
The host image sampler still performs identical neighbor bounds/address checks
for each RGB channel. A four-neighbor gather shared across the channels is
distinct from the rejected normalization-only SIMD/LUT experiments and merits
a controlled measurement, preserving fixed-point sampling and F32 normalization.
The one-way decoder image snapshot is already resident and should not be
reimplemented. The current trace does still contain two 5,242,880-byte async
downloads and one same-sized async upload per image. A direct resident
encoder-output → conditioning input path is worth auditing: it must preserve
capture boundaries, the original conditioned image used by feedback, and all
public outputs. Do not replace this with a partial-output benchmark.

The first configure failed closed on insufficient patch context. Corrected
configure-v2 succeeds; build-v1 uses the spare Vulkan UBSan tree at `-O1 -g1`,
one compiler, a 3.5 GiB cap and the unchanged 10 GiB reserve. The live demo
remains unchanged. No Nix derivations are rebuilt. Ignored
evidence uses `generated/diagnostics/bf16-norm-affine-*`.
