# Packed encoder feed-forward projection (rejected)

Hypothesis: replace the two resident BF16 encoder projections `mlp.w1` and
`mlp.w2` with one wider matrix product. Existing profiling attributes about
12.3 ms across 64 products to these projections. This does not imply that
packing saves 12.3 ms: the arithmetic and weight traffic remain necessary.

The archived `SAM3D_BF16_PACKED_FFN=1` experiment used the resident BF16 graph.
The original two parameter names are uploaded into views of a single immutable
weight allocation and bias allocation at load time. There is no extra weight
copy or per-inference concatenation. The combined product retains F32
accumulation, bias addition and the original BF16 rounding. Two output views
then feed the original separately rounded SiLU/multiply operations. F32 and
the streamed reference/control implementation are unchanged.

The extra CTest runs the existing whole-backbone regression with packing
enabled. It checks streamed versus resident boundaries, repeated/alternating
inputs, image/token graph switching, both supported batches, different prefix
lengths, immutable snapshots, invalid inputs and recovery. All 49 CPU
ASan/UBSan/LSan tests passed while the prototype was present. The trained
Vulkan results below also remained exact, but performance regressed. The
prototype and extra test have been removed from the active source and preserved
in [rejected-packed-ffn.patch](rejected-packed-ffn.patch); `git apply --check
--recount` passes against the restored source. The demo strips the rejected
flag. Normal CTest returns to 48 tests.

Known performance risk: output slices are strided. The existing Vulkan
binary-round fusion requires contiguous inputs, so the gated multiply can
lose fusion. Measure full inference and actual per-operation timestamps,
rather than attributing the entire two-matmul cost to dispatch.

The initial 6 GiB spare-build job was refused before execution for insufficient
host headroom. CPU verification used the usual smaller cap. The spare Vulkan
build uses one compiler and a 5 GiB job cap, above preceding measured build
peaks of about 4.3 GiB; the 10 GiB host reserve and no-swap policy remain intact.
No unrelated process or demo instance was stopped. Evidence is under ignored
`generated/diagnostics/bf16-packed-ffn-*`.

The single-compiler build completed in 440.6 s, peaking at 4,474,183,680 bytes
(4.17 GiB), with 74 `memory.high` pressure events and zero `max`/OOM events.
The lower job cap did not change the 10 GiB reserve. Subsequent incremental
restoration builds used a 4 GiB cap. No Nix derivations were rebuilt.

## Full trained outcome

Five warmups and twenty timed alternating-image complete requests, UBSan:

| Variant | Median | p95 | Range |
| --- | ---: | ---: | ---: |
| Original projections, same binary | 110.649 ms | 112.973 ms | 110.423–113.502 ms |
| Packed projections | 114.746 ms | 116.601 ms | 114.290–117.787 ms |

All 25 outputs from both runs and from the kernel-profile run are bit-identical
to the established accepted full-inference baseline. This demonstrates a
performance failure, not a numerical failure. No release build, new tolerance,
or demo deployment was justified by the slower candidate.

After removal and rebuild, the restored UBSan path measures 110.902 ms and
all 25 full outputs again match the accepted baseline exactly. The 48 normal
CPU sanitizer tests, 112 Python tests and Go race tests pass. The original
demo service invocation remained active throughout; its native build was not
modified or restarted.

GPU timestamps total 75.954 ms for packing. The combined matrix product takes
12.043 ms versus the preceding 12.335 ms for the two original products: only
about 0.3 ms saved. Downstream costs increase: fused binary-round ADD is 6.446
versus 4.598 ms, generic MUL 3.891 versus roughly 2.5 ms, and BF16-round CPY
2.437 versus 1.444 ms. Fused binary-round MUL drops from 2.231 to 0.641 ms
because the strided gated input no longer satisfies the fusion guard. These
measurements come from separate profile runs and are diagnostic, not additive
wall-clock accounting; the same-binary complete timings establish regression.

## Next experiment

Keep the original projections and contiguous tensors. Investigate a guarded
fusion of SiLU → BF16 rounding → multiplication → BF16 rounding, retaining
both exact rounding boundaries and the original SiLU arithmetic. In the
accepted profile SiLU is 1.039 ms, standalone BF16-round CPY 1.444 ms, and fused
binary-round MUL 2.231 ms in aggregate; first establish which portion belongs
to this exact pattern. Requested intermediate outputs, extra consumers,
unsupported types/layouts and diagnostic captures must prevent unsafe fusion.
Use independent off/on signed/tail/rounding-boundary GPU cases and the complete
encoder/body/F32 gates. This is a hypothesis, not an implemented speedup.
