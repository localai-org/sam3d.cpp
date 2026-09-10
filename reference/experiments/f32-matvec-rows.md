# F32 Vulkan matrix-vector row grouping

2026-09-10. Rejected for lack of meaningful speedup; not deployed. The live validated model
remains at 93.8–94.2 ms unsanitized / 114.2 ms UBSan.

The existing F32 matrix-vector kernel computes one output row per workgroup,
even though its source supports a row-count specialization. The accepted GPU
profile shows the precise-prefix `m=64,n=5,k=1029,batch=20` product costing
about 4.7 ms across the encoder, plus repeated MHR projections. Grouping
independent rows might improve reuse/dispatch without changing the dot-product
lane count or reduction order. This is a hypothesis, not an assumed speedup.

The archived `rejected-f32-matvec-rows.patch` exposes NVIDIA-only row counts 2/4/8, default one, using
the same shader and dispatch denominator. No lower-precision operand conversion
or mathematical simplification is added. Unknown values preserve the default.
The source submodule stays pristine. The separate UBSan candidate is built
without rebuilding the live demo directory or any Nix derivation.

The model-free test covers 48 cases, exact dyadic and independent double
non-dyadic products, row/reduction/column tails and broadcast heads/batches.
Compare its saved outputs across process-start choices, then run complete
trained encoder/body trajectories and the original frozen policies, including
strict F32. A synthetic pass alone is not sufficient. Use five warmups and
twenty timed alternating images for full-inference comparisons.

Local ignored evidence starts at `generated/diagnostics/bf16-matvec-rows-*`.
The first configure attempt rejected a missing-context patch hunk before
compilation; corrected context is in the subsequent build. No live model was
affected. Failed evidence remains retained, not treated as successful work.

## Results and revised target

All 48 device cases pass for each of 1/2/4/8 rows. The grouped variants match
one another but differ from one-row output at 457 of 449,804 floats, maximum
5.3644e-7 (relative L2 2.7643e-8). This is not bit-exact evidence. Full UBSan
medians are 114.263 / 114.125 / 114.039 / 115.282 ms respectively, each with
five warmups and twenty timed alternating requests. This does not establish a
useful improvement; eight rows regresses. No unsanitized rebuild or deployment
is justified for this candidate, and it is removed from applied build patches.
GPU kernel timestamps confirm this: 75.670 ms at one row versus 75.645 ms at
four. The complete outputs change slightly; frozen full-model numerical
acceptance was not claimed for this rejected performance experiment.

The important dispatch finding is that GGML's generic timing label is
misleading here: `ggml_vk_mul_mat` only selects the matvec kernel for multiple
columns when the head/batch count is one. The precise-prefix five-column,
twenty-head operation actually uses scalar F32 **matrix multiplication**.
Changing matvec row grouping cannot improve it. The next target is a narrower
F32 GEMM tile for small-N batched products, preserving K accumulation order.
The new independent device test is retained for that path too.
