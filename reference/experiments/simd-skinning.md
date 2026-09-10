# Exact four-influence CPU skinning

2026-09-10. The preceding pinned-transfer version measured 95.0–95.4 ms
unsanitized / 115.9 ms UBSan. Warm CPU samples attributed 8.15% to skinning.

`SAM3D_SIMD_SKINNING=1` processes four independent influences with baseline
SSE2 on x86. It preserves both quaternion normalizations, each multiply/add/
cross-product's order, and serial scatter-add order into mesh vertices. Duplicate
vertex IDs are not reduced in parallel. There is no matrix approximation,
fast-math, FMA contraction, influence pruning, validation bypass or model change.
The scalar path handles tails, diagnostic operation capture and other CPUs.
Without the flag the previous path is retained. No C ABI or GGML patch changed.

## Verification

- All 48 CPU ASan/UBSan/LSan and Vulkan-UBSan normal tests pass.
- The existing original skinning fixture still checks 12 boundaries. It now
  also exercises the SIMD final output. Ninety-six added deterministic cases
  compare scalar, SIMD and captured outputs **bit for bit**, including one/two
  batches, varied rotations/scales/translations, shuffled and duplicate vertex
  indices, zero weights, changing influence counts and every short tail.
- All 25 BF16 outputs in each accepted timing/profile run are byte-identical
  to the previous baseline. The normal public F32 path also matches all 25
  previous outputs; this is important because operation capture deliberately
  remains scalar and cannot alone validate the fast path.
- All 529 F32 captured tensors retain their previous hash. Frozen original
  checks pass all 646 assertions, not replaced with a fast-versus-slow comparison.
- Both final BF16 policies pass with the same reported non-blocking dancer
  hand-logit difference. All 112 Python tests and Go race tests pass.

## Full inference performance

All runs have five warmups and twenty timed alternating official images,
resident weights and complete host outputs. BF16 encoder/F32 decoder and all
six pose/MHR feedback layers remain enabled.

| Run | SIMD on | Same-binary off |
| --- | ---: | ---: |
| First | 94.165 ms | 95.573 ms |
| Repeat | 93.803 ms | 95.294 ms |

The repeat's p95 is 94.597 ms, range 93.539–95.853 ms. UBSan-O2 measures
114.150 ms (p95 114.869 ms). CPU sampling measures 94.159 ms; skinning's sample
share falls to 3.52%, with fence polling 44.51%, copying 5.44% and image
preparation 4.93%. These sample fractions are not additive wall-clock stages.
GPU arithmetic and transferred bytes are unchanged. The **80–85 ms goal stays
open**; GPU work now offers more potential than further skinning work.

Local ignored evidence: `generated/diagnostics/bf16-skin-simd-*`,
`generated/fixtures/bf16-skin-simd-full-f32-v1/`. Heavy jobs are serialized,
memory bounded and retain the 10 GiB reserve. The GGML submodule is pristine;
no Nix derivation rebuilding is involved.

## Demo acceptance

The existing demo was updated in place with the UBSan build and saved history
retained. The server selects SIMD skinning only for its Vulkan BF16 mode;
inherited conflicting precision switches are stripped. Real Chrome 151 QA
passes upload, cancellation/recovery, warm reuse, original overlay, orbit
controls, history/mobile scrolling and exact GLB export/reload. The front
overlay was visually inspected; body and arm geometry are unchanged. Native
and GLB hashes remain identical. Cold/warm browser jobs took 4.349 / 0.169 s
including export, not benchmark medians. Evidence:
`generated/diagnostics/bf16-skin-simd-demo-qa-v1/`.
