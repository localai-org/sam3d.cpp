# Narrow F32 GEMM for the precise-prefix attention path

2026-09-10. Numerical/performance acceptance completed; the demo is updated
with its UBSan build and passes actual headless Chrome QA.

The preceding [matvec row grouping](f32-matvec-rows.md) showed no meaningful
gain. Source inspection established that the five-query/twenty-head prefix
uses the general F32 GEMM path, despite the profiler's generic matvec label.
The original small tile is 32×32, leaving most columns unused for N=5.

Build-copy patch `0008-vulkan-f32-narrow-matmul.patch` reuses the same scalar
F32 shader with a 16×8 tile, 32 threads, WM=16/WN=8, WMITER=1 and TM=TN=2.
The F32 shader fixes K at 32 (specialization slot 3 is ignored for F32),
and each output retains the same ordered F32 dot-product helper.
Split-K is explicitly disabled for the new tile: otherwise N=8 could newly
meet the split heuristic and change reduction order. The selection requires
N≤8, F32×F32, NVIDIA CM2 and 32-lane subgroups; all other paths remain unchanged.
`GGML_VK_F32_NARROW_MATMUL=1` opts in, while
`GGML_VK_F32_NARROW_TRACE=1` reports actual selection. The demo enables the
optimization only for Vulkan BF16 and strips inherited choices/tracing. The
library flag remains opt-in.

The model-free test now has 56 cases including N=8/K=3000 with head/batch
broadcasting to check the split-K guard. It covers row/reduction tails and
independent dyadic/non-dyadic products, with saved outputs for comparison.
All 56 cases pass bit-exactly against the previous tile, including the
non-dyadic cases; actual tracing verifies selection on the targeted shapes.

## Full model and regression evidence

Five warmups and twenty timed alternating official images give **90.806 /
91.071 ms** unsanitized medians, versus **94.491 ms** in the same binary with
the option off. First-run p95 is 91.137 ms, range 90.546–91.659 ms; repeat p95
is 91.753 ms. UBSan-O2 is **111.243 ms** versus 114.416 ms off.
Every one of the 25 complete outputs per accepted run/profile matches the
previous baseline exactly. The normal F32 API also matches all 25 prior results.
No layer, feedback step, geometry or host output is omitted.

Both BF16 complete encoders retain their hashes and pass all 36 frozen checks
each; all 529 F32 diagnostic tensors retain their hash and pass 646 strict
assertions. Both final-body policies pass with the same non-blocking dancer
hand-logit discrepancy. No tolerance changed. CPU ASan/UBSan/LSan and Vulkan
UBSan pass 48/48 tests; the existing actual-device precision/fusion/attention
tests pass, along with 112 Python and Go race tests.

GPU kernel timestamps fall to **72.284 ms**. The targeted prefix product drops
from approximately 5.115 ms to **2.553 ms** across 32 layers. Transfers and graph
count are unchanged: 33,193,136 bytes uploaded, 32,403,596 downloaded, 83 graphs.
Warm GPU telemetry averages 77.3%, peak 79% (18 samples). This is a latency
improvement, not a claim of 90% utilization or achieving the **80–85 ms goal**.

Candidate patch fingerprint:
`eba02773aeaa5ed1d92077066ea98b0cd9dba27e400d8a7cebf8fb41ee0d96fd`.
Local evidence starts at `generated/diagnostics/bf16-narrow-matmul-*`.

## Demo QA

The existing demo was replaced in place with the UBSan build, preserving its
address and saved history. Actual Chrome 151 passes upload, box selection,
invalid-input handling, cancellation/recovery, warm reuse, original overlay,
orbit controls, history/mobile scrolling and exact GLB export/reload. The front
overlay was visually inspected; no geometry change is visible. Native and GLB
hashes remain unchanged. Individual cold/warm requests took **4.211 / 0.253 s
including export**, not benchmark medians. Evidence:
`generated/diagnostics/bf16-narrow-matmul-demo-qa-v1/`.
