# Exact image normalization SIMD experiments (not selected)

2026-09-10. The accepted SiLU-fusion build measures 88.285 / 88.251 ms warm
full inference. A fresh warm CPU profile attributes 5.53% of user-CPU samples
to image preparation. Disassembly confirms scalar `divss/subss/divss` sequences
in the existing path. The older normalization-LUT experiment was already
rejected, so these prototypes retained arithmetic rather than repeating it.

Both variants preserve affine coordinates, fixed-point interpolation, output
RGB bytes, planar layout and the F32 divide/subtract/divide order. There is no
reciprocal approximation, FMA, native-only ISA or validation removal. Baseline
SSE2 has a scalar fallback on other architectures.

| Variant | Same-binary off | On, repeated | Outcome |
| --- | ---: | ---: | --- |
| Four pixels per SIMD vector, separate normalization pass | 88.525 ms | 88.226 / 88.284 ms | No gain over accepted production |
| Three RGB channels per vector, inside the sampling pass | 88.501 ms | 88.333 / 88.192 ms | No repeatable gain over accepted production |

Each run has five warmups and twenty timed alternating official images,
complete own-intermediate inference and host outputs. Every one of all 150
outputs matches the accepted full-result binary exactly. Both variants pass
all 50 CPU ASan/UBSan/LSan tests: the usual 48 plus the image C API with SIMD
enabled and 56 exact-byte-value/offset/tail/guard-buffer normalization cases.
These numerical successes do not establish a useful performance improvement.

The experimental flag added a conditional to the scalar path; comparing only
against that same-binary off path would overstate benefit versus production.
The observed enabled timings remain within the existing production range.
Both variants were removed, and the second is recoverable as
`rejected-image-simd.patch` (`git apply --check --recount` verified after removal).
The live demo never used either variant. Its precision environment continues
to strip the rejected flag, and profiling records it for archived-run provenance.

Ignored evidence: `generated/diagnostics/bf16-image-simd-*`. Restored-source
tests and a fresh full-model run have separate `...restored-...` reports:
48 CPU sanitizer tests pass, and all 25 complete outputs match exactly at
88.404 ms median. This is consistent with the earlier accepted 88.25–88.29 ms
measurements and does not establish a useful prototype improvement.
The next GPU hypothesis is fusion of scale/add/round operations, preserving
each rounding boundary and all real source/output alias checks; no gain is
claimed for that unimplemented hypothesis.
