# Narrow F32 tile follow-up

This follows the eight-patch 16×8 tile described in
[f32-narrow-matmul.md](f32-narrow-matmul.md). The selected follow-up is
`tiny32`, an 8×8 output tile with unchanged K=32. Deeper reduction controls
are not selected. The 80–85 ms full-inference goal remains open.

## Output-tile control

The first experimental patch fingerprint was
`aedb2162ce5e59751765f6f20fdd79780f4ff9da9dae883624a6ec94ca6a18d1`.
It tested an 8×8 output tile against 16×8. Although the experimental environment
value was named `tiny64`, **actual BK was 32**, not 64: upstream's F32 shader
hard-codes BK and ignores specialization constant 3. A `deep64` branch was
therefore a no-op and was not benchmarked. The prior accepted experiment's
description has been corrected to say BK=32 as well.

Both tested tiles pass all 56 independent double-oracle cases and produce
identical binary outputs (SHA256
`afc1ce14aaca695e68ffb9f5e062e8474bd619928d17cf507377ffe88a46154b`).
Five warmups and twenty alternating-image complete requests give UBSan medians
111.108 ms (16×8) and 110.735 ms (8×8). All 25 smaller-tile full outputs are
bit-identical to the established full-inference baseline. GPU prefix P×V cost
is 1.820 ms across 32 layers, versus the preceding accepted 2.553 ms. This
small initial gain required the release and full-parity checks below before
deployment; it is not a claim of reaching 80–85 ms.

Evidence: ignored `generated/diagnostics/bf16-narrow-deep-*-v1*` artifacts.
The first control timing predates adding `GGML_VK_F32_NARROW_TILE` to the
profiler's environment report; its command explicitly set `default`.

## Reduction-depth experiment

The second patch introduces a new F32-only specialization constant 14,
defaulting to the original BK=32. Only the experimental narrow pipelines may
request BK=64. F16 and quantized shaders remain unchanged. It now distinguishes
`tiny32` (8×8×32), `tiny64` (8×8×64), and `deep64` (16×8×64), with the accepted
16×8×32 as default. These retain the ordered four-element dot-product helper
and disable split-K as before. BK must not exceed 64 for this 32-thread,
unaligned two-element loader, otherwise its row load stride becomes zero.

The second fingerprint is
`468ca30f8c90bfe6b4c833295b15b1af8fa3e45b267fa9f05eb4257ad97b9fe9`.
All four variants pass all 56 device cases, retaining the above exact hash.
All 25 full outputs from each variant remain bit-identical to the prior baseline.
UBSan medians for default/tiny32/tiny64/deep64 are
111.829 / 110.812 / 111.014 / 111.868 ms. Increasing BK did not improve on
`tiny32`; the default remains unchanged for callers that do not opt in.

## Release timing and acceptance

The same release binary measures **91.531 ms** with default versus
**90.798 / 90.574 ms** with `tiny32`, using five warmups and twenty timed
alternating-image complete requests per run. Candidate p95 is 91.856 / 91.880 ms;
ranges are 90.395–92.053 / 90.203–93.502 ms. Every full output is exact.
The goal remains open, and the gain is small compared with remaining work.

Both 36-stage BF16 encoder captures retain their hashes and pass frozen policies.
All 529 F32 capture tensors retain their hash and pass all 646 strict assertions.
All 25 public F32 results are also exact. Both 48-test native sanitizer suites,
the real-device precision/fusion/attention checks, 112 Python tests and Go race
tests pass. Existing CPU ASan/UBSan/LSan and the diagnosed Vulkan UBSan exception
are retained; no C ABI change or tolerance relaxation is involved.

Release GPU timestamps total 72.232 ms; prefix P×V is 1.826 ms and Q×K is
0.477 ms across 32 blocks. Graph/transfer structure is unchanged. The repeat
timing run has 19 warm GPU samples, mean 72.5%, peak 79%; this is not 90% GPU
utilization. Timing and utilization are reported rather than inferred from the
single-kernel saving. Local evidence uses `bf16-narrow-depth-*` diagnostics and
`generated/fixtures/bf16-narrow-depth-full-f32-v1/`.

The demo explicitly selects `tiny32` only for Vulkan BF16 and overwrites
conflicting inherited values. Other precision/backend modes strip the flag.
All builds used cached local tools, not Nix derivation rebuilding, with the
existing serialized memory caps and 10 GiB reserve. The two UBSan patch builds
peaked near 4.3 GiB and recorded no OOM events.

## Demo verification

Actual Chrome 151 QA in `generated/diagnostics/bf16-narrow-depth-demo-qa-v1/`
passes upload/box selection, invalid input, cancellation/recovery, resident
reuse, original overlay, orbit, history/mobile scrolling and GLB export/reload.
The front overlay was visually inspected; it retains the same pose. Native
and GLB hashes are unchanged, and every exported vertex and index is exact.
The individual warm UI job took 0.161 s including export, not a timing median.
The live service uses the UBSan build; unsanitized numbers above are benchmarks.

## Next profiling target

Update: the packed-projection hypothesis below was tested and rejected: exact
outputs, but ~4.1 ms slower with UBSan. See [the recorded experiment](packed-ffn.md)
for measurements, the archived implementation and the next activation-fusion
hypothesis. It is not enabled in source or the demo.

The two separate encoder feed-forward projections (`mlp.w1` and `mlp.w2`)
account for about 12.3 ms in the existing profile. A future experiment could
pack immutable weights at model load and issue one wider product, retaining
both original BF16 bias/rounding boundaries. This is not implemented: first
check split-K/accumulation identity and whether the resulting strided views
lose binary-round fusion or add copies that erase the benefit. Do not add
per-inference weight concatenation. Preserve default and full-boundary capture
paths for same-binary controls. The current GPU/host profile does not justify
claiming that this experiment will reach the target.
