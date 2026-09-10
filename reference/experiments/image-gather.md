# Final image-preparation optimization (2026-09-10)

`SAM3D_IMAGE_GATHER=1` shares bilinear-neighbor bounds/address calculations
across RGB channels. Integer interpolation and the original F32 normalization
order are unchanged. This remains opt-in, not enabled in the deployed demo.

Five warmups plus twenty alternating dancer/rider full image-to-mesh calls,
resident weights, BF16 encoder/F32 decoder, release (no sanitizers), twelve
validated GPU patches enabled:

| Case | Median | p95 |
| --- | ---: | ---: |
| Original sampling, off v1 | 86.644 ms | 87.098 ms |
| Shared scalar gather, on v1 | 85.819 ms | 86.788 ms |
| Shared scalar gather, on v2 | 85.871 ms | 86.749 ms |

All 75 complete outputs are byte-identical to the accepted baseline. The added
80-case image C API test covers singleton dimensions, rotation, crop borders,
strides, repeated calls and exact final-row allocation. RGB and normalized F32
buffers match byte-for-byte. The upstream rotated-crop/all-256-byte fixture
and all 49 CPU ASan/UBSan/LSan regressions pass.

A final SSE2 integer-interpolation/normalization variant also passed exact tests
but was rejected as unnecessary complexity: same-binary scalar control was
85.659 ms, SIMD 85.610 ms. Both sets of 25 full outputs are exact. That 0.049 ms
delta does not establish a useful improvement. The helper is archived in
`rejected-image-gather-simd.hpp`, not included by production code.

Evidence: `generated/diagnostics/bf16-image-gather-{off,on}-performance-v1`,
`bf16-image-gather-on-performance-v2`, `bf16-image-gather-simd-performance-v1`,
`bf16-image-gather-scalar-performance-v3`, and `bf16-image-gather-cpu-test-v2`.
The CPU v2 run includes the subsequently rejected SIMD path.
The restored scalar-only source was rebuilt and all 49 sanitizer tests passed
again in `bf16-image-gather-cpu-{build,test}-final`.

This closes the optimization pass at the user's request. It does **not** claim
the 80–85 ms target was met. Affine/norm-affine/image-gather deployment remains
separate from the video feature; the live video demo retains the previously
Chrome-validated UBSan SiLU build. No further optimization was started.
