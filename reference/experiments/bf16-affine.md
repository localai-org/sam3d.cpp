# Exact BF16 affine fusion (in progress)

The accepted SiLU gate build remains deployed at about 88.3–88.4 ms release
latency. This candidate targets additional GPU memory traffic, not image
preprocessing: the previous LUT and both exact SSE2 normalizers did not
establish useful full-model gains and remain rejected.

`0011-vulkan-bf16-affine.patch` adds a separately selected
`GGML_VK_FUSE_BF16_AFFINE=1` shader for two graph patterns:

- MUL → ADD → narrow BF16 → widen F32 (one final rounding boundary).
- MUL → narrow BF16 → widen F32 → ADD → narrow BF16 → widen F32 (two boundaries).

Multiplication and addition are explicitly separate `precise` F32 operations,
not FMA. Dense leading-block broadcasts, row/scalar broadcasts and full-shaped
operands are supported. Padded layouts, interior-dimension broadcasts, observed
or shared intermediates and unsafe real-source overlaps retain the existing
path. The matcher checks every real input, including leaf tensors. A hardened
follow-up also rejects host buffers before using GGML's Vulkan-specific overlap
helper. Elided cast self-destination markers are ignored only for the fully
checked pattern, as in the accepted SiLU fusion. No native model graph or
rounding policy is changed. The demo strips the inherited experimental flag.

`tests/vulkan_bf16_affine.hpp` exercises 1,248 off/on cases across independent
buffers and graph-allocator reuse, six sizes, both rounding contracts, two
input repetitions, coefficient/bias broadcasts, misaligned views, padded and
observed/shared fallback paths. All observed outputs are compared byte-for-byte
and checked against an independent scalar oracle. The corpus must detect both
omitted/interposed BF16 rounding and accidental FMA. In particular,
`(1 + 2^-23) * (1 - 2^-23) - 1` is zero with separate F32 operations but
`-2^-46` under FMA, which survives BF16 conversion.

The first build-copy fingerprint is
`e92f6ba4ec6b692e823572a1b555bd81e26753d2676a4c8a434eb4afcfc9a0b7`.
The host-buffer hardening and push-constant layout assertion were added while
that build was running; they require a subsequent rebuild and verification
before acceptance. That obsolete build was deliberately stopped (exit 143):
its compiler reached about 3.3 GiB RSS and spent almost all its time in cgroup
memory throttling (26,085 `high` events, zero `max`, `oom` or `oom_kill` events
at inspection). This is not an inference crash or an OOM result.

The hardened build-copy fingerprint is
`09fb83e1ac717e91251a15a20bb4dcc21ed8daf2a643737475d24e1dbcd267e0`.
The live `build/vulkan-bf16-cm2` tree is untouched. The spare
`build/vulkan-bf16-pointwise` now uses UBSan and `-O1 -g1`, one compiler, a
3.5 GiB cap and the unchanged 10 GiB reserve. Configure-v2 succeeded; the
first replacement build declined to start for insufficient headroom before
the old compiler's memory had been released. Build-v3 is the actual corrected
build attempt. Release timing must still use the separate `-O3` tree; an
`-O1` sanitizer result is only correctness evidence. No Nix derivations are
rebuilt.

## Hardened candidate results

- UBSan build-v3 completed in 235.1 seconds, peak 3,132,006,400 bytes,
  166 soft-throttling events and zero OOM/max events. Lower compiler
  optimization resolved the earlier prolonged compile throttling.
- Device precision-v2 passed all existing precision tests and all 1,248
  affine off/on cases. Its independent oracle recorded 1,954,176
  rounding-boundary and 1,300 FMA negative-control differences. GPU profiling
  confirms 494 actual affine dispatches inside this test section; this is not
  an all-fallback result. The bounded test exited zero with UBSan enabled.
- Both 25-input alternating full-model runs (`off-body-v1`, `on-body-v1`)
  match all complete baseline result files byte-for-byte. Same-binary `-O1`
  UBSan warm medians: **114.201 ms off / 112.878 ms on**, five warmups and
  twenty measurements each. These are not release performance results.
- Kernel profile-v1 executes **162 affine fusions/image**. Mean warm GPU
  time is **68.038 ms**, versus 69.138 ms in the accepted SiLU profile.
  The affine dispatches themselves total 1.711 ms/image. All 25 profiled
  requests retain full model work and outputs.
- Both 36-stage encoder captures are byte-identical to the accepted captures
  and pass their frozen upstream policies. Both final-output policies pass;
  the existing dancer `hand_logits` discrepancy remains reported and
  non-blocking. No tolerance was changed.
- The strict F32 529-tensor capture retains SHA-256
  `a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`;
  the complete strict upstream comparison passes with no failures.

## Release timing

The hardened release build passed the same device precision suite, including
the 1,248 exact affine cases and both negative controls. On the same binary,
five warmups plus twenty timed alternating images give:

| Selection | Median | p95 | Range |
| --- | --- | --- | --- |
| Affine off | 88.026 ms | 88.518 ms | 87.656–88.674 ms |
| Affine on, first run | 86.976 ms | 87.803 ms | 86.555–87.929 ms |
| Affine on, repeat | 86.683 ms | 87.293 ms | 86.501–88.073 ms |

All 75 complete result files are byte-identical to the accepted baseline.
This establishes a useful approximately 1 ms gain, **not achievement of the
80–85 ms goal**. No model work or outputs are omitted. The separate 25-request
public-F32 sanitizer run also matches every baseline result exactly.

All 48 CPU ASan/UBSan/LSan tests, all 48 normal tests in the rebuilt Vulkan
UBSan tree, 112 Python tests and Go race tests pass. The first Go attempt
failed before testing because `gcc` was not on PATH; the explicit cached
compiler configuration passed without rebuilding Nix dependencies.

Pending: demo integration/Chrome QA. The first
3.5 GiB release-build attempt declined to start for insufficient headroom.
The 3 GiB build-v2 was stopped after soft-throttling stalled its compiler
(10,332 high events, zero max/OOM events at inspection). Once host headroom
permitted, build-v3 resumed at 3.5 GiB, still reserving 10 GiB, and completed in
145.6 seconds with 2,973,249,536 bytes peak memory and zero high/max/OOM events.
The deployed demo remains
the accepted ten-patch build. Evidence lives in ignored
`generated/diagnostics/bf16-affine-*` artifacts.
