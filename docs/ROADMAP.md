# Roadmap

This is the current task list. The [design](DESIGN.md) defines architecture and
acceptance criteria; [development history](../reference/HISTORY.md) preserves
earlier experiments, including superseded results. Neither is a second backlog.

## Available now

The supported scope is the **SAM 3D Body pose branch**: explicit image/person
box/camera input, native CPU/Vulkan inference, MHR mesh decoding and opaque C
APIs. The optional demo supports photos, offline video and live webcams, with
history, static posed-mesh export, live take recording and skeleton animation
GLB export. Video is frame-wise estimation. See [skeleton export](SKELETON-EXPORT.md)
for formats, coordinate conventions and importer limitations.

Official sources/checkpoints and safe conversion schemas are pinned. The
reference captures, layer/operation comparisons, official-image visual checks,
headless Chrome QA, sanitizer/fuzz tests and optimized upstream profiling have
been implemented. This does not imply that every numerical or full-model gate
in the design has passed.

Licensing, the model card, explicit-file uploader and clean-checkout build
verification are prepared. GGML uses a public submodule commit, with optional
patches applied to build copies.

## Remaining release steps

- [ ] Publish the source repository and identify its release commit/tag.
- [ ] When authorized, publish the verified Body bundle to
  `LocalAI-io/sam-3d-body-dinov3-GGUF` using the
  [publication workflow](../distribution/README.md).
- [ ] After publication, make downloading the bundle the recommended setup
  and verify an end-to-end installation from public source/model downloads.

## Known limits and numerical work

- Six strict F32 CPU accumulated-geometry/projection comparisons remain open
  (640/646 pass). Keep them visible; do not substitute Vulkan success for CPU
  acceptance. See [trained branch evidence](../reference/TRAINED_BODY_BRANCH.md).
- BF16 has separate upstream-derived tolerances and a recorded hand-logit
  discrepancy. Keep acceptance scoped to the tested Body branch/images, not
  the complete hand-refined estimator. See [precision](../reference/BODY_PRECISION.md).
- The measured warm BF16 native time is about 85.8 ms versus 81.6–84.5 ms for
  the compared upstream CUDA workloads. Optimization was paused after that
  result; further work is optional, not an active demand to hit an exact number.
  Matched complete-pipeline CPU performance parity is not established.
- Backend precision currently includes documented process-wide settings.
  A future per-session contract should remove that configuration dependency.
- Live webcam estimates can jitter or lag; the client interpolation does not
  add temporal understanding or guarantee physical motion accuracy.

## Future features — separate scope

- Complete and validate both-hand image refinement and optional user prompts.
  Existing hand experiments do not establish final refined-image parity.
- Expose a supported Body feature-only API for GEM-X integration; GEM temporal
  inference, its separate ViTPose features and SOMA decoding belong to that track.
- Library-level cancellation and configurable graph memory budgets. The demo's
  process cancellation is not a public-library cancellation API.
- Automatic person detection/calibration or multi-person tracking, with separate
  model contracts and acceptance if added.
- SAM 3D Objects: finish its learned conditioning/generation pipeline and demo
  under a new scope. Existing component tests are not end-to-end Objects inference.

For any implementation change, retain operation/layer comparisons and final
output checks; for demo changes, repeat the real browser workflow. Performance
claims require matching workload, precision and output scope.
