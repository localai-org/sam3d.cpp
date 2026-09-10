# Bounded pinned-transfer batching

Numerical, performance and actual headless Chrome demo acceptance completed
on 2026-09-10. No GGML patch is added.

## Motivation

The fresh warm CPU profile of the small-tile baseline measured 99.81 ms
instrumented latency. Samples attribute 37.33% to Vulkan fence polling, 7.21%
to memory copies, 6.34% to CPU skinning, and 5.07% to image preparation.
These are CPU sample fractions, not disjoint wall-clock stage times.
Full inference still has 202 uploads and 233 downloads for 83 compute calls.
Evidence: `generated/diagnostics/bf16-tiles-host-profile-v2/`, including saved
matching warm symbols. A preceding profile launch safely refused insufficient
RAM for the default 6 GiB allowance; the measured native workload then passed
with a 4 GiB cap and the unchanged 10 GiB host reserve.

## Candidate contract

`SAM3D_BATCHED_TRANSFERS=1` enables an internal synchronous-call wrapper around
GGML's existing async upload, graph compute and download operations. It copies
inputs into a reusable pinned host allocation, queues disjoint input/output
regions, synchronizes, and copies complete outputs to their caller-owned
buffers. All arithmetic, output fields and finite checks remain in place.

- Validate all complete contiguous tensor extents and nonnull pointers before
  submitting any work; the graph/scratch buffers remain alive until completion.
- Limit the reusable pinned pool to 32 MiB. Oversized valid requests and
  backends without suitable same-device pinned storage use synchronous IO.
- A pinned-allocation fallback to ordinary CPU memory is not counted as a
  successful batched transfer.
- Complete outstanding work on error before callers can release graph inputs,
  scratch tensors or outputs. Public model calls already serialize a session.
- Do not assume GGML's Vulkan host buffer belongs to the selected GPU: the
  pinned upstream currently associates it with device zero. Other-device
  buffers fall back safely.

Integration covers 82 of the 83 full-inference graphs: decoder, pose-head,
camera-head, feedback, conditioning, prompts and MHR projections. The image
encoder retains its original synchronous path. Device capabilities are cached
once per session: measuring them per graph cost 1.83 ms per image in the initial
35-graph prototype. A model-free test alternates input values and off/on settings across
odd sizes, tensor views, staging growth/reuse, output guards, malformed
requests and recovery. A 36 MiB transfer verifies the bounded fallback.
CPU ASan/UBSan/LSan and the Vulkan-UBSan build pass all **48** normal tests.
The actual NVIDIA test passes 84 alternating cases and 84 pinned batches,
including with GGML's timestamp logger enabled.

The profiling shim separately records async enqueue calls and synchronization.
Async host enqueue duration is not GPU compute time; nested synchronous/
asynchronous wrapper totals must not be added as independent wall time.
The demo selects this flag only for Vulkan BF16; inherited conflicting values
are stripped. Other precision/backend modes retain their previous defaults.

## Measurements

The expanded candidate measures **95.417 / 95.401 ms** median without sanitizers
(five warmups, twenty timed alternating images; nearest-rank p95 95.892 ms,
range 95.212–98.193 ms in the first run). Its UBSan-O2 counterpart measures **116.201 ms**.
All 25 complete results in each run are byte-identical to the accepted baseline.
The smaller initial 35-graph version measured 96.200 ms after caching device
properties, versus 98.922 ms before caching. The same-binary off control is
100.406 ms. After adding the diagnostic-only guard below, the clean median is
**95.021 ms**, p95 95.902 ms, range 94.745–95.935 ms; UBSan is **115.942 ms**.
The GPU utilization sample mean is 76.6%, peak 78% (19 warm samples), not 90%.
The 80–85 ms goal remains open.

The expanded candidate makes 199 async uploads and 232 async downloads per
image, plus three synchronous uploads and one synchronous download. Total
bytes remain 33,193,136 uploaded and 32,403,596 downloaded, with all 83 graphs
and final host outputs retained. This reduces submission/copy overhead, not
the model's work or data volume. Evidence: `generated/diagnostics/bf16-transfers-*`.

All **529 F32 captured tensors** retain SHA-256
`a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`;
all **646** frozen original F32 comparisons pass. Both original final-BF16
policies pass with the same non-blocking dancer hand-logit discrepancy.
No tolerance changed. Python regression passes 112 tests.

## Timestamp logger compatibility

The first kernel-profile attempt aborted with SIGABRT at GGML's
`GGML_ASSERT(ctx->compute_ctx.expired())`: async uploads had opened the compute
context before the timestamp logger reset its query pool. This was not an OOM
(zero memory events) and did not occur in ordinary inference. Preserve that
failed run as `bf16-transfers-kernels-v1`; it is not accepted evidence.

When `GGML_VK_PERF_LOGGER` is present, the helper now drains uploads before
entering graph compute. Production retains a single final synchronization.
The device test and full 25-request profile now pass, with every result exact.
Kernel timestamps total **75.943 ms**; instrumented full inference is 107.572 ms.
The extra profiling-only boundary means this is a diagnostic kernel measurement,
not the production latency. CPU sampling without that logger measures 95.543 ms.

## Demo and public-output checks

All 25 normal F32 public results match with batching on/off; the on run records
199 async uploads and 232 async downloads, so this exercises batching rather
than only the oversized diagnostic fallback. Go race tests pass. The existing
UBSan demo is updated in place with history retained. Actual Chrome 151 QA
passes upload, cancellation/recovery, warm inference, overlay, history/mobile
scrolling and exact GLB export/reload; the front overlay was visually inspected.
The warm request took 0.174 s including export. Evidence is
`generated/diagnostics/bf16-transfers-demo-qa-v1/`; native and GLB hashes remain
unchanged. The 80–85 ms goal is not yet complete.
