# BF16 saved-output visual comparison

## Current precise-prefix candidate and live demo

The latest smaller-BF16-matrix-tile optimization is byte-exact and passes
actual Chrome 151 QA: `generated/diagnostics/bf16-tiles-demo-qa-v2/`.
The front original/native overlay was inspected; body and GLB hashes remain
unchanged, with every exported vertex and index exact. Upload, invalid/oversized
input rejection, box dragging, cancellation/recovery, warm reuse, camera orbit,
history reload and mobile scrolling pass with no page/GL errors. Cold/warm
browser jobs took **4.188/0.283 s** including export (single UI requests, not
benchmark medians). Separate native medians are **99.1–99.5 ms unsanitized /
120.1 ms UBSan**. The updated demo continues to use UBSan.

The first QA attempt (`bf16-tiles-demo-qa-v1/`) failed safely before inference:
Chrome plus other host processes left too little room for a 6 GiB worker
allowance and 10 GiB reserve. It is not counted as a pass. Measured native
peaks were 2.22–2.41 GB, so the local demo was restarted with an explicit
4 GiB worker cap, no swap, and the **unchanged 10 GiB reserve**. The complete
rerun above then passed. No unrelated processes were stopped and no memory
guard was disabled. The QA harness now reports terminal job errors directly
instead of waiting for a cancellation state that cannot occur after failure.

The preceding host-validation optimization is byte-exact and passes actual
Chrome 151 QA: `generated/diagnostics/bf16-host-demo-qa-v2/`.
The front overlay was inspected and the body/GLB hashes remain unchanged.
Cold/warm browser jobs took **4.312/0.177 s** including export; there were no
page/GL errors. Separate native medians are **105.0 ms unsanitized / 125.8 ms
UBSan**. These are complete static pose-branch outputs, not video inference.

The preceding linear-indexing optimization is byte-exact and passes actual
Chrome 151 QA: `generated/diagnostics/bf16-linear-binary-demo-qa-v1/`.
The inspected front overlay, uploaded body and exported/reloaded GLB keep the
same accepted geometry. Cold/warm browser jobs took **4.184/0.185 s** including
export, with no page/GL errors. These single UI jobs are not native benchmark
medians; repeated native medians are **109.5–109.6 ms release / 133.9 ms UBSan**.

The preceding resident-stem optimization is also byte-exact and passes actual
Chrome 151 QA: `generated/diagnostics/bf16-resident-stem-demo-qa-v1/`.
The inspected front overlay, uploaded result and exported/reloaded GLB retain
the same accepted geometry. The cold/warm browser jobs took **4.148/0.369 s**
including export, with no page/GL errors; these individual UI timings are not
native performance medians. Separate repeated native medians are
**111.4–111.9 ms release / 136.2 ms UBSan**.

The earlier exact binary-round fusion also passes complete live-demo QA,
with unchanged accepted body and GLB geometry. Its preceding evidence is
`generated/diagnostics/bf16-pointwise-demo-qa-v1/`: eight screenshots, fresh
worker 4.435 s and warm inference+export 0.193 s. Repeated native warm medians
are **114.6–115.1 ms unsanitized / 147.4 ms UBSan**.

The preceding layout-only optimization's evidence is
`generated/diagnostics/bf16-condition-layout-demo-qa-v1/` (eight screenshots,
report; fresh worker 4.131 s, warm inference+export 0.303 s). Native warm timing
was separately **116.7 ms unsanitized / 149.3 ms UBSan**. The earlier precise-prefix
demo run below measured 0.205 s warm; these individual browser jobs are not a
median performance benchmark. Both runs have zero browser/GL errors and exact
accepted-output identity.

The residual-corrected/precise-prefix candidate was rechecked in real Chrome
151.0.7922.173. All six dancer/rider front, side and oblique screenshots pass
and were visually inspected: matching silhouettes, torso and limb orientation,
and shared-frame overlays without native-only arm/wrist/body distortion.
Mean/max vertex distances are **0.300/0.969 mm** (dancer) and
**0.509/1.468 mm** (rider). One dancer hand-logit check remains reported and
non-blocking. Both complete encoder trajectories now pass all 36 stages.
Current saved-output reports/screenshots are
`generated/diagnostics/bf16-precise-prefix-{dancer,rider}-visual-v1/`.

The demo now supports explicit `--precision bf16`, with F32 remaining the
portable default. It sets the validated arithmetic environment for persistent
and one-shot workers, reports precision in the UI and saved job metadata, and
does not relabel old F32 history. Go race tests cover conflicting inherited
precision switches, both backends, both launch modes and invalid options.

**Actual live-demo QA also passes**, separately from the saved-output renderer:
real image upload, invalid/oversized rejection, cancellation/recovery, warm model
reuse, mesh and reference overlay, orbit/front/side controls, exact GLB vertex
and index export/reload, history/reload and desktop/mobile scrolling. No browser
or GL errors occurred. The completed result is byte-identical to the accepted
full native BF16 output, SHA-256
`d9cba0af27668be06acfb2491fae838cef7a3128f131f2f94ec466b2492d2fb6`.
The warm browser job is **0.205 s including export** on the UBSan worker.
Desktop overlay and mobile screenshots were inspected. The demo's optional
overlay is the existing labeled original **F32** example; the separate six
saved-output comparisons above use the frozen original **BF16** outputs.

Evidence: `generated/diagnostics/bf16-precise-prefix-demo-qa-v1/report.json`
and its eight screenshots. The harness's new `--expect-precision bf16` and
`--expected-native PATH` options enforce actual job precision and accepted-output
identity. Numerical tolerances are not used to excuse export changes.
The 80–85 ms performance target remains open; at that checkpoint unsanitized
native latency was 121.5 ms. Historical evidence below predates these corrections.

## Verified scope

On 2026-09-09, real headless Chrome 151.0.7922.173 rendered the complete
original PyTorch and native GGML BF16 pose-branch outputs for the official
dancer and rider photographs. Both already pass their respective frozen
18-field final-output policies. This check does not run inference, inject
reference intermediates, change the demo server or substitute for upload/E2E QA.

`scripts/qa_body_captures.py` verifies the successful native profile, native
result hash, exact original/native RGB input identity, original model topology,
camera/box metadata and frozen reference-result identity. It reads the actual
predicted vertices and projected vertices from the saved final outputs.
The displayed PNG is encoded losslessly from captured RGB, not a separately
decoded JPEG. The original state supplies topology, which must equal the native
result's faces exactly.

The view presents the input/projected vertices, original mesh, native mesh and
an overlay. All 3D panels share one camera, metric scale and origin, using the
demo's fixed `(X,-Y,-Z)` coordinate conversion. Only reference bounds frame the
camera; no candidate-specific fitting, centering, rescaling or alignment occurs.
The bundled, licensed Three.js modules render through Chrome's software WebGL.
This rendering choice is not an inference or GPU-performance benchmark.

## Results and visual inspection

Front, side and oblique screenshots were captured and inspected for both cases.
All six screenshots show matching overall body silhouette, torso orientation,
limb placement and arm/wrist pose. No gross native-only distortion or explosion
is visible. The dancer's raised knee and overhead arm, and the rider's arm near
the head and asymmetric leg positions, agree with the original reconstruction.
Both original models retain the pose-branch's limitations; this is not a claim
that it recovers the true body perfectly or matches hand-refined Meta examples.

| Case | Mean vertex distance | Max vertex distance | Browser checks |
| --- | ---: | ---: | --- |
| Dancer | 0.389 mm | 1.278 mm | All 3 views pass |
| Rider | 0.475 mm | 1.723 mm | All 3 views pass |

The checks assert nonempty rendered mesh pixels, draw calls and no GL, JavaScript
or HTTP errors. Reports preserve browser version, all source hashes, policy
checks, shared camera coordinates and screenshot hashes. The rider render was
repeated after final provenance checks were added; screenshot hashes are exact.
Python regression coverage is now 106 tests, including rejection of changed
photos, camera settings, weights, neural sources and framework identities.

## Dancer capture provenance

The early dancer benchmark had no saved decoded-RGB artifact. A fresh original
capture now supplies that artifact, with identical neural source hashes,
framework versions, weights, photo and camera/box settings. Only capture-script
identity and an already-corrected residency description differ. The native input
hash remains `3badb322c580a89275c8f4257a70decad730aa0ef30f2816466d4b0e3e88366b`.

The fresh original final result is not bit-identical: maximum vertex-coordinate
difference from the earlier original is 0.775 micrometres; maximum projected
vertex-coordinate difference is 0.000611 pixels. The visual harness correctly
rejected substituting that result for the frozen reference. Instead the explicit
`--image-reference` option supplies only RGB provenance; the old frozen original
geometry and numerical policy remain unchanged. No tolerance was widened.

## Reproduce

Run from the repository root with existing verified captures and Chrome. For
example, inside the memory-bounded runner:

```sh
python scripts/run_bounded.py --report generated/new-visual-budget.json -- \
  python scripts/qa_body_captures.py \
  --native-profile generated/diagnostics/bf16-rider-resident-performance-v1 \
  --request 1 --reference generated/benchmarks/body-rider-bf16-eager \
  --policy reference/bf16-body-rider-policy-v1.json \
  --safe-state generated/extraction/body/body-other-state.safetensors \
  --name 'Official rider: original and native BF16' \
  --output generated/new-rider-visual
```

Output/report paths must be new. Temporary HTTP and DevTools servers bind only
loopback and are stopped on completion. No model, photo or generated screenshot
is added to Git.

Local ignored evidence:

- `generated/diagnostics/bf16-dancer-visual-v1/{front,side,oblique}.png` and `report.json`
- `generated/diagnostics/bf16-rider-visual-v2/{front,side,oblique}.png` and `report.json`
- `generated/benchmarks/body-dancer-bf16-visual-eager/`

The earlier fused-attention encoder misses and missing live BF16 QA above are
historical: the accepted precise-prefix path and subsequent exact-output
optimizations now preserve both complete-image stage policies. The **80–85 ms
performance target remains open**.

## Binary-round fusion demo recheck

`generated/diagnostics/bf16-pointwise-demo-qa-v1/` records another successful
real Chrome 151 upload/cancellation/recovery, warm generation, original overlay,
GLB export/reload and desktop/mobile history test. The saved front overlay was
visually inspected with no new gross pose distortion. The native result remains
`d9cba0af27668be06acfb2491fae838cef7a3128f131f2f94ec466b2492d2fb6`;
all exported vertices and indices match exactly. The browser overlay uses the
existing original F32 reference; the earlier two-image capture comparisons use
their frozen original BF16 results. These are separate checks, not interchangeable
reference identities. The current UBSan demo's cold/warm jobs took 4.435/0.193 s
including export. See [numerical and performance evidence](BODY_PRECISION.md#exact-binary-operation-bf16-round-fusion).

## Pinned-transfer demo recheck (2026-09-10)

`generated/diagnostics/bf16-transfers-demo-qa-v1/` records successful actual
Chrome 151 QA of the updated UBSan demo: upload, box selection, invalid-input
handling, cancellation/recovery, warm resident reuse, original overlay,
orbit controls, history/mobile scrolling and GLB export/reload. The front
overlay was visually inspected; no new gross body/arm distortion is visible.
All exported vertices and indices remain exact, with the same native and GLB
hashes above. The warm job took **0.174 s including export**, not a benchmark
median. The frozen original BF16 numerical checks and this existing original
F32 browser overlay remain distinct evidence. The performance goal stays open.

## Exact SIMD skinning demo recheck (2026-09-10)

`generated/diagnostics/bf16-skin-simd-demo-qa-v1/` passes the same real Chrome
151 end-to-end workflow after the CPU skinning optimization. The front overlay
was visually inspected; native geometry, GLB vertices/indices and both hashes
remain unchanged. Cold/warm jobs took 4.349 / 0.169 s including export (individual
QA jobs, not performance medians). No browser errors were recorded.

## Narrow F32 GEMM demo recheck (2026-09-10)

`generated/diagnostics/bf16-narrow-matmul-demo-qa-v1/` passes actual Chrome
151 upload/cancellation/recovery, warm reuse, overlay, orbit, history/mobile
scrolling and exact GLB export/reload. The front overlay was visually inspected;
native output and all exported vertices/indices and hashes are unchanged.
Individual cold/warm jobs took 4.211 / 0.253 s including export, not native
benchmark medians. No browser errors were recorded.

## Smaller narrow tile demo recheck (2026-09-10)

`generated/diagnostics/bf16-narrow-depth-demo-qa-v1/` passes the same actual
Chrome 151 end-to-end workflow with the 8×8×32 F32 tile selected. The front
overlay was visually inspected and is unchanged; native and GLB hashes and
every exported vertex/index remain exact. The warm job took 0.161 s including
export (one QA request, not a benchmark median). There were no page/browser
errors. The deployed native worker retains UBSan.

## Exact SiLU gate fusion demo recheck (2026-09-10)

`generated/diagnostics/bf16-silu-gate-demo-qa-v2/` passes actual Chrome 151
upload, cancellation/recovery, repeat generation, overlays, orbit, history,
mobile scrolling and exact GLB export/reload. The front overlay was visually
inspected; complete native output and all exported vertices/indices remain
unchanged. Cold/warm UI jobs took 3.979 / 0.372 s including export (individual
requests, not native performance medians). No page/browser errors. The initial
v1 run was refused before inference by the RAM guard; v2 uses a 2 GiB native
BF16 cap above measured usage, with the 10 GiB reserve unchanged. UBSan remains
enabled in the deployed native worker.
