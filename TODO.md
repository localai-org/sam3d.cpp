# Goal: native SAM 3D Body with an end-to-end web demo

## Current requested goal: BF16 parity and 80–85 ms

The latest user request prioritizes practical BF16 body reconstruction agreement,
then approximately 80–85 ms warm Vulkan inference guided by profiling. Detailed
hand refinement and isolated hand/finger discrepancies must not become a blocking
detour. See [the bounded goal and acceptance gates](reference/BODY_BF16_GOAL.md).
The replacement goal is active in the goal tracker as of 2026-09-09.

- [ ] Freeze upstream-derived BF16 numerical tolerances and negative controls.
- [ ] Validate native CPU/Vulkan layers and complete outputs on representative images; inspect overall body/mesh agreement and keep hand metrics non-blocking.
- [ ] Profile encoder, attention, decoder/geometry, host work, transfers and synchronization.
- [ ] Optimize toward 80–85 ms warm NVIDIA Vulkan inference at unchanged workload/output scope, rechecking BF16 quality and strict F32 regressions at each step.
- [ ] Verify repeated/alternating inputs, sanitizer/fuzz coverage, final warm benchmarks and real headless Chrome demo QA.

## Earlier project scope and backlog

Scope reduced by user request on 2026-09-09. Objects is deferred to a later
goal. Its code, fixtures and completed work remain intact, but unfinished
Objects tasks (including shared checklist items mentioning both models) are
not prerequisites for Body completion. This supersedes the earlier two-model
goal wording; Body parity, demo/Chrome QA and performance requirements remain.

The active goal and acceptance criteria are specified in [DESIGN.md](DESIGN.md).
Layer/operation parity against the real pinned upstream implementation is the
default method throughout. Passing an intermediate test never implies full
pipeline, demo or performance acceptance.

## Reference and foundations

- [x] Benchmark resident original F32/BF16 CUDA execution, including genuinely compiled backbone entry points and output-difference reports.
- [x] Implement explicit native BF16 encoder mode through opaque model options and CLI, preserving F32 default/decoder/MHR; CPU and Vulkan inference run.
- [ ] Complete trained BF16 layer/operation acceptance. Small original BF16 fixtures pass, but trained own-intermediate traces do not meet strict F32 limits; final-mesh proximity is not a substitute.
- [ ] Accelerate BF16 matrix operations without enabling implicit F16 conversion in the F32 decoder, then re-profile toward upstream CUDA performance. See `reference/BODY_PRECISION.md`.

- [x] Audit official references, prior C++ ports and reusable local code without executing third-party engines.
- [x] Record source revisions and published weight identities; distinguish these from locally verified weight bytes.
- [x] Specify required demo, real headless Chrome workflow and final visual/numerical comparison.
- [x] Fetch and verify clean pinned official Body/Objects source checkouts (source-only preflight, not R0).
- [x] Implement hash/path/source preflight and ordered safetensors layer comparison; pass 12 weight-free tool tests.
- [x] Port Body bbox/affine crop geometry with opaque C API; match 135 actual-upstream cases and pass sanitizer-enabled native tests (partial B1 only).
- [x] Port Body RGB resampling/ToTensor/normalization; match 12 actual-upstream image cases including the dancing photograph (B1 neural backbone still pending).
- [x] Pin public GGML 0.23.0 submodule and official DINOv3 source; build explicit dynamic CPU/Vulkan backends without private GGML commits.
- [x] Implement F32 patch graph and actual DINOv3 PatchEmbed synthetic-weight fixtures; CPU and strict-F32 NVIDIA Vulkan pass all 12 boundary checks (not trained-model parity).
- [ ] Carry strict-F32 Vulkan configuration into native model session setup; per-session/backend precision must not depend on silent operand downcasts or undocumented environment settings.
- [x] Implement DINOv3 eval block/RoPE/SwiGLU contract; CPU and strict-F32 NVIDIA pass 88 original-upstream synthetic-weight checks, including full Body block shape. Add a compact actual-upstream fixture to normal CTest.
- [x] Compose patch/prefix/block/final-norm/feature layout path with parameter streaming and a GGUF adapter; full 32-block 512x512 synthetic contract passes ASan/UBSan CPU and NVIDIA versus PyTorch CUDA. Normal CTest includes original three-block regression; trained backbone parity remains pending.
- [x] Port CameraEncoder ray antialiasing, Fourier features, projection and LayerNorm2d; 32 original-upstream synthetic component checks pass CPU/NVIDIA and a compact original fixture runs in normal CTest.
- [x] Construct native camera rays/CLIFF condition from crop/intrinsics, including upstream batch F32 metadata rounding; all 60 original geometry boundaries match byte-for-byte. Opaque C API, normal regression and sanitizer fuzz coverage added.
- [x] Port the Body F32/GELU/eps1e-6 decoder layer, including separate Q/K/V, positional repeat/skip, masks and reverse image attention. All 290 synthetic original-layer boundaries pass CPU/NVIDIA, including CUDA comparison; six small cases run in normal CTest.
- [x] Port Body dense/keypoint/pixel positional encoding and learned label-table selection; all 108 original-operation synthetic checks pass CPU/NVIDIA. Preserve CPU direct versus CUDA reciprocal scalar division explicitly; both arithmetic modes have normal CPU-only regression tests.
- [x] Compose native camera/prompt conditioning, initial/previous-estimate projections and keypoint token assembly through the first actual decoder layer. All 52 synthetic original-method boundaries pass CPU/NVIDIA; no upstream intermediate injection. Normal CTest includes three composed cases.
- [x] Port PerspectiveHead FFN/residual and complete full-perspective reprojection in F32 GGML; 68 original synthetic operation boundaries pass CPU/NVIDIA, including mesh-sized point sets. Normal sanitizer tests include zero-depth rejection.
- [x] Port full-image-to-crop projection and 2D/3D keypoint feedback; 77 original synthetic boundaries pass CPU/NVIDIA, with border/depth masking, pre-projection masking bias and last-layer identity regression tests. This consumes supplied geometry, not native MHR output yet.
- [ ] Keep feedback sampling and token updates resident on the selected backend when composing the full model; the current checked native bilinear sampler runs on CPU between GGML graphs.
- [x] Port body-mode MHRHead through actual MHR inputs: 519-value FFN/residual, distinct global/body/hand rotation paths, PCA scales/hands, masks and 204-value parameter assembly. All 320 synthetic original boundaries pass CPU/NVIDIA; no generated geometry is substituted.
- [ ] Validate Body geometry/keypoint outputs with the trained head and checkpoint mapping buffers; the native implementation is connected, but SAM head state in current integration tests is synthetic.
- [x] Obtain the independently public official MHR geometry release; verify archive/member/license hashes and exact SAM MHR byte identity. Execute actual TorchScript geometry on CPU/CUDA and extract safe state in isolation.
- [x] Convert required MHR geometry state to F32/I32 GGUF and validate all 18 real tensors using the native sanitizer-enabled reader. Normal tests cover typed format/metadata/index rejection; this is not geometry inference parity.
- [x] Implement own GGUF parameter projection, local skeleton transforms and exact F64 prefix schedule. CPU/Vulkan pass 46 original operation taps plus the uninstrumented CPU/CUDA skeleton; normal sanitizer tests cover isolated local/FK and invalid schedules. Not final mesh parity.
- [x] Implement MHR identity/expression blendshapes, both pose-corrective projections and weighted skinning; CPU/Vulkan pass 48 original operation/final-output checks with native-produced intermediates on official demo parameters. Normal tests include original selected-vertex skinning. Native first COO projection is materialized densely for GGML, compared against original sparse computation.
- [x] Implement Body's meter/axis/keypoint/rotation mapping and compose pose prefix -> real MHR -> Body outputs. 70 original comparisons pass CPU/Vulkan using real geometry plus explicitly synthetic head/PCA/index/mapping state. Normal mapping regression and quaternion/axis rejection checks added; not trained Body parity or full decoder feedback.
- [x] Compose the full promptable decoder with its own pose-head/geometry/keypoint feedback between layers. Two/six-layer synthetic SAM-state trajectories with real MHR pass 526 original boundary/full-output checks on CPU and NVIDIA Vulkan/CUDA; no injected intermediate poses or geometry. See `reference/BODY_FLOW.md` for the deliberately limited scope.
- [x] Validate the composed decoder at full neural widths with synthetic SAM state and real MHR. CPU/Vulkan each pass 389 checks after independent upstream CPU/CUDA controls established the explicit unit-circle relative-Euler policy. Raw absolute/raw relative errors remain recorded and output bytes are unchanged; see `reference/BODY_FLOW.md`.
- [ ] Validate the full decoder with actual trained state/config/mapping; synthetic composition does not establish learned model parity.
  - [x] Connect verified trained state and actual no-mask/hand-box branches to raw-RGB inference; original CPU/CUDA captures are repeatable and observer-neutral. Initial Vulkan passes 646 boundary/operation/final-field checks.
  - [x] Verify actual SDPA/module operations in all six trained decoder layers on CPU/Vulkan. All 204 operation tensors pass after matching operand scaling; 646 complete branch checks pass on Vulkan.
  - [ ] Close the remaining six CPU accumulated-geometry/projection comparisons (640/646 pass). Original MHR on native pose inputs also exposes the layer-1 discrepancy; investigate incoming pose rounding, not only skinning. Do not weaken geometry limits.
  - [x] Package trained decoder/head state as a checked GGUF companion. Verify all 316 tensors and a GGUF-only RGB inference session: Vulkan passes 646 checks and matches the inline-weight output byte-for-byte.
- [x] Connect native-produced crop/image/backbone/camera results through conditioning into decoder. The official dancing RGB image passes 435 original boundary/full-output comparisons per CPU/Vulkan backend with synthetic SAM state and real MHR; see `reference/BODY_IMAGE_FLOW.md`. No reference intermediates are supplied. This is not trained reconstruction or demo acceptance.
- [x] Confirm official Body/Objects model access (2026-09-09); verify Body's selected bytes and pin Objects pipeline/component configuration hashes.
- [x] Download and locally size/SHA-256 verify all six selected Objects inference checkpoints and seven pipeline/component configurations.
- [ ] Finish license inventory and pin the full auxiliary dependency/config/asset closure.
- [ ] Establish the isolated official reference environment; verify downloaded weights and capture safe tensors/assets.
- [ ] Capture repeatable official Body and Objects image examples, unmodified final outputs and validated per-operation taps (R0).
- [ ] Establish upstream GGML submodule, reviewed build-copy patches, Linux build/install, optional Nix, sanitizer/fuzzer presets and tests.
- [ ] Implement checked conversion/loading and opaque-handle C API foundations.
  - [x] F32 safetensors-only Body-backbone GGUF converter and bounded native component reader; 552 shapes match original meta factory; sanitizer-enabled format/rejection checks pass.
  - [x] Isolated restricted-loader Body extraction with exact safe-reader byte verification; real F32 backbone GGUF conversion and native sanitizer verification. Preserve all remaining Body state for decoder/geometry integration.
  - [x] Diagnose trained QKV accumulation and LayerScale amplification; freeze a reference-only trained stage policy with negative tests, improve Vulkan F32 accumulation and pass all 36 backbone stage checks on CPU/Vulkan.
  - [x] Observe actual original math-SDPA scaling/softmax on CPU/CUDA, match native operand scaling, and implement full 32-block streamed original/native operation traces with observer-neutrality checks.
  - [x] Pass all 704 trained own-intermediate operation checks on CPU and NVIDIA Vulkan against the frozen original-control policy, plus 36 stages and observer-neutrality checks. See `reference/TRAINED_OPERATIONS.md`.
  - [ ] Complete model identities/companions and public model/session C API.
    - [x] Body pose-branch model/options/request/result API, named tensor descriptors and coordinate metadata, exact real Vulkan output check, C-only installed consumer, deterministic error/ownership tests and sanitizer fuzzing of non-GGUF surfaces.
    - [ ] Body feature-only API, cancellation, memory budgets, full refinement capabilities and installed end-to-end inference QA. Objects model APIs are deferred.

## Body and its demo

- [x] Audit the actual full hand-refinement path and match hand-box/crop/mirroring preparation: 69 exact original comparisons, normal pixel regression and 100k sanitizer fuzz cases. See `reference/HAND_REFINEMENT.md`; this is not hand-network or refined-geometry acceptance.
- [x] Capture/port the hand head's wrist-centric conversion, non-hand parameter mask and output keypoint mask using real checkpoint buffers: 21 comparisons pass against each upstream CPU/CUDA capture, plus synthetic-only regression cases.
- [x] Connect the separate learned hand head through wrist transfer, parameter masking, actual MHR and output mapping. Vulkan passes 158/158 operation/final-field/structural checks on two synthetic-token batches with real head weights; original repeated/observed outputs are byte-identical. This is not image/decoder refinement parity.
- [ ] Close two strict CPU hand-head internal geometry comparisons (156/158 pass); upstream MHR replay localizes the discrepancy to propagated global rotation/translation rounding. Final returned fields pass, but that does not waive internal limits.
- [x] Connect actual hand-specific decoder weights, independent dense positional encoding and the learned hand head through six layers of own-intermediate geometry/feedback. Capture unchanged original hand methods with observer-neutrality checks; feature inputs are synthetic, not an image reconstruction.
- [ ] Resolve strict hand-decoder numerical failures: CPU 328/593 and Vulkan 383/593 pass. Original CPU/CUDA controls pass only 330/593 on identical inputs; retain failures and establish operation-level causes before changing arithmetic or acceptance policy.
- [x] Connect the actual learned hand-image composition from explicit ROI/RGB, including padding 0.9, original shared no-mask embedding and shared box heads. Right-hand Vulkan passes 590/646 generic checks and all independent final fields; original CPU/CUDA controls pass 538/646. This is wiring and diagnostic evidence, not accepted hand-image parity.
- [ ] Complete both hand-image CPU/Vulkan comparisons and operation-level acceptance, then native body-derived crop selection, unmirroring and the full refinement merge using its own intermediates. See `reference/HAND_IMAGES.md`.

- [ ] B1: preprocessing and DINOv3 H+, CPU and Vulkan layer-by-layer parity.
- [ ] B2: conditioning, decoder, heads and GEM feature contract parity.
- [ ] B3: MHR geometry, keypoints, cameras and final deformed-vertex parity.
- [ ] B4: supported prompts and hand refinement with explicit reference cases.
- [x] D1 body-branch demo: LocalAI image upload/person selection, real public-C-API inference, 3D mesh/skeleton viewing, static GLB/OBJ downloads and persistent history. Detailed hand refinement is explicitly unavailable.
- [x] Frame-wise offline video sequences and live webcam mode: selected person, bounded crop following, one shared resident worker, configurable maximum Hz, interpolation, saved sequence playback and headless browser QA.
- [ ] Broader real-motion temporal-quality evaluation, robust person detection/re-identification, and learned temporal/GEM integration (not implied by frame-wise video support).
- [x] Headless Chrome exercises real upload, cancellation/recovery, inference, camera, GLB reload, history and mobile layout. Screenshots inspected; every public field matches the accepted native run and final geometry matches the original official-example Body branch. See `reference/BODY_DEMO.md`.

## Objects and its demo — deferred, not part of the active goal

- [x] Port original-default RGBA image/mask crop, padding, bicubic-AA/nearest resize and full/object views. All 114 original checks pass, including official kid_box image/mask. Opaque C API, installed-consumer test and 100k sanitizer fuzz run pass. This does not include pointmaps, learned conditioners or checkpoint-config validation; see `reference/OBJECTS_IMAGE.md`.
- [ ] O1: native image/mask/pointmap preprocessing and conditioners, including required MoGe stages.
  - [x] Port PointPatchEmbed eval with all remappers, invalid/forced-dropout tokens and timm window transformer. 196 original synthetic operation/full-output checks per CPU/Vulkan backend pass, including default 256/8/768 size. Normal regression and rejection tests included; see `reference/OBJECTS_POINTPATCH.md`.
  - [ ] Validate trained PointPatch state/configuration and connect native SSI/MoGe pointmaps; the component currently consumes supplied XYZ.
  - [x] Port all eight SSI normalization variants and own denormalization with original PyTorch3D homogeneous invalid-point behavior. Native CPU passes 475 comparisons against each CPU/CUDA reference; 249 small boundaries run in normal CTest and 100k internal sanitizer fuzz cases pass. See `reference/OBJECTS_SSI.md`.
  - [ ] Connect the original pointmap-aware image/mask/crop transform sequence and SSI results into native conditioning; preserve its soft-mask behavior, distinct from default RGBA preprocessing.
    - [x] Port the explicit original triple resize/crop/rembg chain with soft alpha and NaN-aware XYZ alignment. All 129 original checks pass, including the full official photograph paired with synthetic XYZ; all earlier RGBA outputs remain byte-identical. See `reference/OBJECTS_JOINT.md`.
    - [x] Compose SSI, joint transforms and individual/full-image transforms through the actual upstream PointMap preprocessing method. All 536 original checks pass on CPU, including independent final fields and the full official photograph with supplied synthetic XYZ. Normal regression and 100k sanitizer fuzz cases pass; see `reference/OBJECTS_PREPROCESS.md`.
    - [x] Connect raw-input preprocessing to shared PointPatch and the fuser in an explicit point-only configuration. Small CPU/Vulkan comparisons pass 108 checks plus an original nonfinite-domain negative control per backend. Full-size CPU (sanitizers) and Vulkan each pass another 54 checks. Normal raw-input/final-field tests added; learned configuration remains separate. See `reference/OBJECTS_POINT_CONDITION.md`.
    - [ ] Add the actual image encoders, verify the complete checkpoint-selected conditioning configuration and learned state, and validate full raw-input conditioning.
  - [ ] Keep point-window weights resident and select strict F32 precision at session level; current correctness runner explicitly disables Vulkan operand downcasts.
  - [x] Port the original post-encoder EmbedderFuser eval graph: shared LayerNorm/SwiGLU projections, positions, forced drops and token/channel fusion with optional compression. All 140 original synthetic checks pass per CPU/Vulkan backend; 111 boundaries run in normal sanitizer CTest. See `reference/OBJECTS_FUSER.md`.
  - [ ] Wire actual native encoder outputs into the fuser in checkpoint-selected order, then validate trained state and complete conditioning from raw inputs.
- [ ] O2: sparse-structure flow/sampling/decoding and pose/scale parity.
- [ ] O3: structured-latent flow and final Gaussian parity using native-produced sparse support.
- [ ] O4: native mesh decoding/export and final topology/vertex/attribute parity.
- [ ] O5: separately scoped official-equivalent repair/UV/texture processing when textured-PBR output is enabled; do not label raw export as this mode.
- [ ] D2: Objects image/mask workflow in the same demo, real inference, supported geometry viewing/download and persistent history.
- [ ] Headless Chrome exercises the real Objects workflow; inspect screenshots and compare with an official example in matched views.

## Final correctness and performance

- [x] Profile current Body demo Vulkan latency with CPU stacks, transfer timings and device timestamps; see [measured bottlenecks](reference/BODY_PERFORMANCE.md).
- [x] Benchmark/deploy optimized Vulkan with UBSan/assertions; remove repeated backbone scans, retain immutable MHR weights on CPU/GPU, reuse bounded scratch buffers; pass upstream parity and real Chrome QA (~3.3 s sample job vs ~40 s).
- [ ] Extend residency to backbone/decoder weights and activations, separate parity taps from final-result inference, and use a persistent bounded worker; re-run parity before deployment.
- [ ] Complete Body CPU/Vulkan final-output acceptance from original upstream sample images, with no injected reference intermediates.
- [ ] Complete sanitizer/fuzzer, C API, malformed-input, clean-checkout/install and demo failure/recovery QA.
- [ ] After functionality is complete, validate and profile optimized upstream PyTorch, testing compilation and appropriate backend/thread settings.
- [ ] P1: achieve Body CPU performance parity at matched complete-pipeline workloads and accepted precision/quality.
- [ ] P2: achieve Body GGML Vulkan performance parity with PyTorch CUDA on this system's NVIDIA GPU.
- [ ] Re-run correctness and real headless Chrome QA after performance changes; retain reproducible measurements, screenshots and artifact identities.

The separate gem-x.cpp temporal model/integration track is described in the
design, but implementing all of GEM-X is not a prerequisite for this Body-only
goal. Trained component/branch results do not close complete-model, demo or
performance gates.
