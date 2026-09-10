# BF16 matrix tile experiment

Small tiles are now accepted after full numerical, timing and browser checks.
The earlier six-patch baseline was about 105 ms unsanitized / 125.8 ms UBSan;
the accepted selector measures 99.1–99.5 ms / 120.1 ms respectively.
The following sections retain the failed/intermediate experiments as evidence.

## Host compiler control

An unsanitized `-O3 -g` build (assertions retained; no fast-math) measured
104.744 ms median over five warmups and twenty alternating-image runs,
range 104.384–106.731 ms, p95 106.358 ms. All 25 outputs were byte-identical
to the `-O2 -g` baseline. All 47 normal tests and existing Vulkan precision
cases passed. This marginal difference is not sufficient evidence for a
meaningful speedup, and the O3 build has not replaced the demo.
Evidence: `generated/diagnostics/bf16-o3-*`.

## Initial tile probe and correction

The first selector guarded on BF16 source tensor types. Model-free BF16×BF16
products showed faster small kernels, but complete inference stayed at
126.162 ms UBSan and had exact baseline output. Source review found that
the model uses F32 activation carriers: GGML converts these to BF16 while
retaining the original RHS type in the tile-selector arguments. Thus that
first selector did not affect the trained model. It is not a full-model
performance result for small tiles.

The revised guard checks the actual `pipeline_matmul_bf16`, leaving F32
decoder kernels unchanged. The oracle now also tests the F32 carrier path.
Existing kernel alignment and split-K behavior are preserved; any changed
accumulation must still pass the frozen full-trajectory numerical policies.

An initial tiny single-batch N=7 oracle case hit pinned GGML's unsupported
BF16-RHS matrix-vector assertion (its vector path handles up to eight
columns). This was a test-dispatch limitation, not a model inference failure
or OOM. The test now uses N=65 for that tiny M/K case; a batched N=7 case
continues to cover matrix tails. The failed log is retained, not counted
as a passing run. The revised first-selector test passed all 32 cases with
and without trace logging before adding F32 carriers.

Evidence: `generated/diagnostics/bf16-tiles-*`. The selector is default-off
in the backend; the demo chooses small tiles only for Vulkan BF16 and strips
inherited choices/tracing. No policies were relaxed.

## Corrected selector: UBSan results

All 64 analytical matrix cases pass for both BF16 and F32 carriers, with
and without tracing. The actual trained path now changes: full warm inference
is **120.119 ms median UBSan**, range 119.602–122.291 ms, p95 121.569 ms
(five warmups, twenty timed requests, two alternating images). This compares
with 126.162 ms for the ineffective selector and 125.803 ms for the previously
accepted build; it is not an unsanitized performance claim.

All 25 complete outputs are byte-identical to the accepted baseline. Both
complete 36-stage encoder captures are also identical:

- Rider: `63dd08704a3fe33f8376c7dfaecaf0e213870d1e5c593a02182b72683eb7a5bf`.
- Dancer: `5a8301b8158b7511adbfba071c8c83336cbc81970f98e8f546288a8e4fe38591`.

The full strict-F32 capture retains all 529 tensors and hash
`a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`.
Existing device precision cases pass, including the 384-execution fusion
guard audit (24 eligible, 360 refused/disabled). Candidate build-copy patch
fingerprint: `872577fb87b243456d948600a5bb9cf02c700f7ffbe13a462c5b2eec7d7fff1c`.

Evidence: `bf16-tiles-carrier-{oracle,timing}-v1.log`,
`bf16-tiles-small-ubsan-v2/`, `bf16-tiles-{rider,dancer}-stages-v1.bin`,
`bf16-tiles-precision-v1.log`, `bf16-tiles-fusion-guards-v1.json` under
`generated/diagnostics/`, and `generated/fixtures/bf16-tiles-full-f32-v1/`.
The demo was unchanged during these numerical tests, then updated after
acceptance. Repeated unsanitized runs measured 99.460 / 99.128 ms versus
104.870 ms with the old heuristic in the same O3 binary. Fresh upstream
comparisons pass both 36-stage encoder policies, both final-body policies
and all 646 strict F32 assertions. All 47 normal tests pass in CPU
ASan/UBSan/LSan, Vulkan UBSan and performance builds; 112 Python tests and
Go race tests also pass. Actual Chrome QA passes in `bf16-tiles-demo-qa-v2/`,
including exact exported geometry. See `reference/BODY_PRECISION.md` and
`reference/BODY_BF16_VISUAL.md` for profiling, visual and memory-budget details.
