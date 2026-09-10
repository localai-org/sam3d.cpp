# BF16 Body: practical parity and 80–85 ms Vulkan inference

Requested by the user on 2026-09-09. This supersedes the broader completion
requirements for the next work phase. Objects, video and detailed hand-crop
refinement are not prerequisites. The previous goal was cleared by the user;
this replacement goal was registered and activated on 2026-09-09.

**2026-09-10 checkpoint:** the user requested stopping after the image-gather
optimization and then adding offline/live video. The optimization pass closes
at 85.82–85.87 ms release medians; 80–85 ms is not claimed achieved. Numerical
evidence is retained in [the final experiment](experiments/image-gather.md).
Video development is recorded separately in [BODY_VIDEO.md](BODY_VIDEO.md),
and currently uses the previously validated UBSan SiLU inference build.

## Outcome

Establish defensible BF16 numerical/visual agreement with official SAM 3D Body,
then target **80–85 ms median warm image-to-mesh inference on NVIDIA Vulkan**,
guided by profiling. At goal creation native BF16 was about 274 ms; the measured original
CUDA baselines are 84.5 ms eager and 81.6 ms with the backbone compiled.

Match the benchmark scope: one image/person, explicit box and intrinsics,
512×512 DINO encoding, all six pose/mesh-feedback layers, MHR, final host output
buffers and resident weights. Exclude model loading, automatic detection,
segmentation, hand-crop refinement, HTTP upload and exports from this latency
target. Report these exclusions and cold-start costs separately. Do not achieve
the target by dropping geometry feedback, reducing dimensions or returning
partial outputs. CPU remains a correctness/sanitizer target, not an 80 ms target.

## Tasks and gates

1. **Establish BF16 acceptance from upstream controls.**
   Capture representative official images with identical pixels, boxes,
   intrinsics and model identities across original BF16 math/automatic SDPA and
   genuinely compiled execution. Observe layer/operation boundaries where
   supported and verify instrumentation does not change the reference. Derive
   explicit per-boundary tolerances from upstream-only variation, BF16 rounding
   resolution and numerical conditioning; freeze and hash the policy before
   using it to accept optimized candidates. Include negative controls for
   incorrect layout, scaling, missing operations, offsets and nonfinite values.
   Do not enlarge tolerances in response to candidate failures.
2. **Validate native CPU/Vulkan quality.**
   Compare both isolated operations on matched inputs and complete own-intermediate
   trajectories. Report maximum absolute and relative-L2 errors, final vertex/
   body-joint distances, projections and overall pose. Inspect original/native
   mesh overlays on representative images. Preserve strict F32 regression tests.
   Numerical variation consistent with the frozen BF16 policy is acceptable;
   mathematically different operations and implausible geometry are not.
3. **Keep hands from becoming a detour.**
   Preserve all existing model outputs and report hand/finger errors, but isolated
   hand discrepancies and hand-refinement parity are non-blocking for this goal.
   Do not zero or remove hand outputs to hide errors. Gross arm/wrist/body
   distortion or nonfinite/exploding values still fail overall quality checks.
4. **Profile and optimize the dominant costs.**
   Separate encoder, decoder/feedback, geometry, transfers, graph dispatch,
   synchronization and host preprocessing/layout work. First investigate BF16
   tensor-core execution without implicit F16 conversion of the F32 decoder;
   then use measured costs to prioritize attention, graph reuse/residency and
   host work. Retain upstream GGML as a fetchable submodule; any necessary
   changes must be reviewable build-copy patches, not private submodule commits.
   Recheck numerical acceptance after each material optimization and record its
   latency/VRAM/transfer impact. Utilization is diagnostic, not a substitute for
   latency or a reason to add unnecessary GPU work.
5. **Verify the final performance and demo.**
   Use synchronized warm measurements with at least five warmups and twenty
   timed runs, plus alternating-input/repeat checks and representative-image
   measurements. Report median and tail/range, precision, output scope and
   instrumentation overhead. Aim for 80–85 ms rather than one lucky sample.
   Preserve opaque C API controls and F32 fallback/default until BF16 acceptance.
   Integrate accepted BF16 into the demo, perform actual headless Chrome upload,
   rendering/overlay/export QA, and document measured results and limitations.

## Execution safeguards

Follow [MEMORY_SAFETY.md](MEMORY_SAFETY.md): serialize heavy jobs, bound native
and Docker RAM separately, retain host headroom, and avoid unrelated processes.
Keep ASan/UBSan enabled where supported; the diagnosed NVIDIA ICD exception
uses the separate UBSan Vulkan build, with CPU sanitizer coverage retained.
No Nix derivation rebuilding is needed merely to run benchmarks. Existing
failed comparisons remain evidence, not successes reclassified after the fact.

Baseline details and artifacts: [BODY_PRECISION.md](BODY_PRECISION.md).
