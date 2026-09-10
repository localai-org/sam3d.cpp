# Exact BF16 SiLU gate fusion (accepted)

The candidate keeps the original separate feed-forward projections, unlike
the [rejected packed experiment](packed-ffn.md). It fuses the six adjacent
nodes SiLU → F32/BF16 cast → BF16/F32 cast → multiply → F32/BF16 cast →
BF16/F32 cast into one shader, preserving both original round-to-nearest-even
boundaries and upstream's `x / (1 + exp(-x))` expression.

The native BF16 graph swaps the equal-shaped F32 multiply operands so that the
second projection is traversed first. This makes the activation chain adjacent
without changing the arithmetic or F32 graph. The patched backend requires
`GGML_VK_FUSE_BF16_SILU_GATE=1`, in addition to existing round/binary fusion
flags. Captured intermediate outputs, extra consumers, broadcasts, non-dense
layouts and real unsafe aliases retain the original implementation. The demo
strips inherited experimental flags and selects the validated fusion only for
Vulkan BF16.

## First investigation

Patch fingerprint `0c123879da3b7cff0c11324fbba3f4aa728ef696c6affbaf152887ef7558d52d`
passed 180 exact off/on device cases with independent arithmetic checks.
The test initially expected 48 eligible executions but observed only 36:
DFS had inserted a gate VIEW inside one intended positive sequence. Explicitly
expanding that view first corrected the setup; v2 records all 48 positives.

Full UBSan inference measured 111.303 ms off / 110.986 ms on, and every one of
the 25 outputs per run was byte-identical to the accepted baseline. However,
the unchanged kernel counts exposed that these were **not proof of executing
the full fusion**. GGML assigned `fusion_string` before its overlap check and
did not clear that label when cancelling fusion. Thus a `BF16_SILU_GATE` timing
label alone was misleading: the full model's allocator had caused fallback.
These timings are not claimed as an accepted fusion speedup.

## Allocator-aware follow-up

For this fully checked six-node pattern only, a fresh elided CPY's
`src[1] == self` is a destination marker, not a live input. An earlier cast's
allocation can overlap the final output after lifetime reuse; the generic
overlap loop was rejecting that harmless marker. The follow-up skips only
these elided self-markers and retains all real source/output overlap checks.
It also clears the profiler fusion label whenever overlap cancels fusion.

The device test now repeats with a graph allocator as well as independent
tensor buffers, reuploading inputs between executions. Both paths check exact
off/on outputs, observed/shared intermediates, padding, broadcast fallback,
misaligned views, both multiply orders, signed values and tails. V3 passed
360 cases off/on exactly, with 54 actual fused executions. Full-model v2
outputs (25 off and 25 on) remained byte-identical to the accepted baseline;
UBSan medians were 111.035 / 110.815 ms. But kernel profile v2 correctly showed
**zero** full gate fusions: the profiler-label fix is effective, and the
remaining rejection is not an elided destination marker.

V3 test / full v2 patch fingerprint:
`6f559d560ee95ce01b5a57db172729b980538b32cdfd10812777cebb2185079f`.

## Explicit destination candidate

`scripts/trace_silu_allocations.c` inspects the actual public graph tensors
without changing backend behavior. On the first trained block, both live F32
inputs are 21,073,920 bytes. The gate starts at offset `0xa4dd00`, but the
final output starts at `0x547900`: their ranges partially overlap. The backend
is correct to refuse fusion. The SiLU input at `0x1e66d00` does not overlap
that destination. These are Vulkan buffer offsets, not host addresses.
The profiling harness reports missing timing trace for this allocation-only
shim; its exit status is not used as a numerical/performance acceptance test.

The next graph uses ordinary `ggml_cpy(BF16(multiply), multiply)` to widen into
the now-dead F32 multiply result. This keeps six operations and both rounding
boundaries, provides explicit allocator lifetime information, and is executable
on unmodified CPU GGML. The fusion matcher checks that exact destination,
use counts, shapes and all original real-source overlap restrictions. F32
graphs are unchanged. New device modes exercise this destination with both
allocation schemes and an observed intermediate fallback (408 total cases).
Explicit-destination patch fingerprint:
`450e2d0b20edbac32d31b73e090c48a608abcd19b9a54499abfb473bd746fa4a`.

V6 device tests pass all **408** off/on comparisons. Per-case profiler markers
confirm 68 fused executions, including 14 explicit-destination positives;
remaining allocator overlaps and observed/shared cases safely fall back.
Earlier v4/v5 test versions marked the final view OUTPUT, recursively marking
its underlying multiply OUTPUT too. Those checked fallback, not the intended
positive. A downstream CONT now observes the positive's result, matching the
trained graph; the observed-view negative remains. This correction does not
change the shader, model or numerical gate.

Full trained kernel profile v3 confirms **32 actual gate fusions per request**,
with standalone SiLU dispatches gone and corresponding copy/multiply work
removed. GPU timestamps total **69.138 ms**, versus the accepted 72.232 ms.
The clean UBSan full-model v3 median is **108.509 ms** (five warmups, twenty
timed alternating images), versus the preceding 110.8 ms. Timestamp-instrumented
wall time is not used as clean latency. Every one of the 25 clean and 25
instrumented outputs matches the accepted full-result binary exactly.

Both complete BF16 encoder captures are unchanged and pass all 36 frozen
stage gates each. Both frozen final-body policies pass, with the existing
non-blocking dancer hand-logit discrepancy retained. All 529 captured F32
tensors are byte-identical, and all 646 strict original F32 assertions pass.
CPU ASan/UBSan/LSan passes 48 normal tests; Python passes 112; Go race tests pass.
The first F32 capture was refused before launch for RAM headroom. V2 used a
3 GiB cap, above the prior measured 2.33 GiB peak, without reducing the reserve.

The public F32 regression also matches all 25 complete outputs exactly.
Demo deployment/browser QA now pass; see below.

## Release timing and next profiling target

The reduced-debug O3 build measures **88.285 / 88.251 ms** median across two
enabled runs, versus **90.614 ms** with fusion disabled in the same binary.
Each has five warmups and twenty timed alternating inputs. Enabled nearest-rank
p95 values are 88.565 / 88.502 ms. All 75 complete outputs match the accepted
baseline byte-for-byte. The arithmetic workload, outputs and frozen tolerances
are unchanged. The **80–85 ms goal remains open**.

Warm GPU samples average 75.2 / 76.9%, peaking at 79%; this is not a 90%
utilization claim. A fresh CPU profile measures 88.314 ms with sampling enabled
and retains all 25 exact outputs. Its 977 warm user-CPU samples (zero lost) put
41.76% in fence polling, 5.83% in memory copies, 5.53% in image preparation and
3.28% in skinning. These are CPU sample fractions, not additive wall-clock
stage times. GPU work still dominates; image preprocessing and copies are
concrete remaining host targets. Do not remove validation or change pixel
rounding to obtain a speedup. Evidence is `bf16-silu-gate-host-profile-v1/`,
including the exact warm interval and symbol report.

## Build safeguards

Build v1 failed patch-context validation before compilation; v2 succeeded.
The follow-up v3/v4 were refused for RAM headroom before execution. Stopping
the idle demo did not release enough memory (there was no orphaned worker),
so the validated demo was immediately restarted. No unrelated service was
stopped. V5 uses a single compiler and a 4.5 GiB cap, above previous measured
build peaks, with the unchanged 10 GiB reserve and no swap. This is a bounded
build, not a reduction in the required host reserve. V5 completed in 495.1 s,
peaking at 4,197,048,320 bytes, with zero OOM/max events; `memory.high` throttled
it 23,760 times. V6 completed in 516.2 s with a 4,208,721,920-byte peak and zero
OOM/max events, under the same limits. No Nix rebuilding.

Release-build attempts v1/v2 were refused before launch as available RAM fell.
The isolated performance build now uses `-O3 -g1` instead of `-O3 -g` for C/C++:
only debug-symbol detail changes, not optimization or assertions. The sanitizer
build and live demo are untouched. V3 tries this reduced-debug build with one
compiler and a 3.5 GiB cap; the 10 GiB reserve and zero-swap constraint remain.
V3 completed in 223.2 s with a 3,132,006,400-byte peak (2.92 GiB), 316
`memory.high` events and zero max/OOM events. The same-binary off/on controls
above use that build. Reducing debug detail proved sufficient without touching
the reserve, the sanitizer configuration or any unrelated service.

## Demo acceptance (2026-09-10)

All 48 normal tests also pass in the updated Vulkan UBSan build; the updated
Go environment selection passes race tests. The existing service was replaced
in place, preserving history. The first Chrome run was refused by the native
RAM guard before inference, not a crash. The deployed BF16-only worker now has
a 2 GiB cap, above its measured 1.03 GiB peak, while retaining the full 10 GiB
reserve. Browser and native workers have separate cgroup caps.

`bf16-silu-gate-demo-qa-v2/` passes actual Chrome 151 upload, invalid/oversized
input, box selection, cancellation/recovery, cold/warm generation, reference
overlay, orbit, history/mobile scrolling and GLB export/reload. The front
overlay was visually inspected. Every native output byte and every exported
vertex/index is unchanged; GLB SHA-256 remains
`534749117b98935a3304081deef667ef9a470ff5f974acd6ed0841e2a4b519cc`.
Individual cold/warm UI jobs took 3.979 / 0.372 s including export; these are
not benchmark medians. No page/browser errors. The live native build is now
`build/vulkan-bf16-cm2`; do not rebuild it while that instance is running.

Evidence: ignored `generated/diagnostics/bf16-silu-gate-*` artifacts. The live
demo now uses the accepted ten-patch UBSan build with the same output geometry.
