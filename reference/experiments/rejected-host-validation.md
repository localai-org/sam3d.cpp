# Host optimization experiments not selected

2026-09-09. The accepted direction uses explicit baseline SSE2 finite checking
on x86, with the original scalar `std::all_of` fallback elsewhere. No checks are
removed. Two alternatives were measured and not retained:

1. A scalar integer reduction intended for auto-vectorization:

   ```cpp
   uint32_t invalid = 0;
   for (float value : values)
       invalid |= uint32_t((std::bit_cast<uint32_t>(value) & 0x7f800000u) == 0x7f800000u);
   return invalid == 0;
   ```

   GCC 15 at the actual `-O2 -g` unsanitized performance settings emitted a
   scalar loop (`mov`, `test`, `sete`, `or`), not SIMD. It passed negative tests
   and all 25 complete output identities but regressed median inference from
   109.5 ms to 115.24 ms. The instrumented CPU profile repeated 115.91 ms.
   Evidence: `bf16-finite-performance-v1/`, `bf16-finite-host-profile-v1/`.
   The later explicit SSE2 variant measured 105.25/105.14 ms with exact outputs.

2. U8 image normalization lookup, built once per request with the original
   `(float(value)/255.0f - mean[channel])/stddev[channel]` operation order for
   all 256 values and three channels. It kept affine sampling unchanged and
   passed exact all-byte C API tests and all 25 complete outputs, but measured
   105.68 ms versus 105.14–105.25 ms for SSE2 validation alone. This did not
   establish an improvement and was removed. The all-byte regression test is
   retained. Evidence: `bf16-host-performance-v1/`. Final host-candidate runs
   numbered v2 and later must use the original per-pixel normalization.

These are five-warmup/twenty-timed alternating-image native measurements, with
complete host outputs. Neither failed performance experiment changes the
frozen numerical gates or is counted as an accepted speedup.
