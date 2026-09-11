# Body hand refinement: original path and current evidence

This track implements original `run_inference(inference_type="full")`, not a
post-hoc replacement of hand coefficients. The public API still advertises only
the body pose branch. No full/refined-hand parity claim is made here.

Sources are the pinned Body revision in `sources.json`. The audited
`sam3d_body.py` SHA-256 is
`851b7475f18b56891aa02606e7c0ee9e03120fa208cc85df5127b792e1abfeee`;
the estimator defining hand transforms is
`8bfa34316d82eabbe185c1cd64a77c47670785cdb0fa489b1481fb468266a145`.
The already verified safe Body extraction contains the hand decoder/head
parameters; additional checkpoint downloads are not required for these tensors.

## Original stages that must be preserved

1. Run the body pose branch and predict two hand boxes. `_get_hand_box` scales
   normalized boxes by the body crop size, treats the first two values as
   **centres**, squares the extents, and maps them into the full image using the
   stored F32 affine. The original detector comment says `x1,y1,w,h`, but its
   consumer explicitly uses centres; follow executable behavior.
2. Mirror the image and left box with `width - x - 1`. Prepare left and right
   crops with **padding 0.9**, not body padding 1.25. Original intrinsics are
   copied unchanged into the mirrored crop. After left inference, its box centre
   is unmirrored for downstream validity checks; its inference input remains
   mirrored.
3. Run the backbone and separate `decoder_hand` with its own initial pose,
   camera, ray encoder, image positional Gaussian, token embeddings, feedback
   FFNs and heads. Prompt embeddings and hand detection tokens are shared where
   upstream shares them. The hand head is still 519-dimensional, but it is not
   merely a body head with renamed weights.
4. The hand MHR head converts wrist-centric rotation/translation into the body
   frame using `local_to_world_wrist`, `right_wrist_coords` and `root_coords`.
   It zeros the **145 checkpoint-defined `nonhand_param_idxs` entries** before MHR and
   masks all keypoints outside indices 21..41 afterward. Its local/global
   rotation conventions must be captured explicitly, not inferred from names.
5. Unmirror the left prediction: scale coefficient conversion, right-wrist
   global matrix to the left slot with sign changes, and right-hand coefficients
   into the left-hand half. Check the four original acceptance masks: local
   wrist angle (<1.4 radians), both crop dimensions (>64 px), crop keypoints
   inside ±0.5, and wrist reprojection distance (<0.25 crop extent).
6. Use valid hand wrists and body elbows as prompts to **rerun the body
   decoder**. Single-person prompt filtering is strict at the ±0.5 edges and
   can produce no prompts, in which case this update is skipped.
7. Combine hand pose, hand-specific/shared scale and shape. Run MHR/FK to derive
   zero-wrist frames, solve the original XZY wrist Euler update, and apply the
   final validity mask. Shared coefficients are averaged over accepted hands.
8. Recompute final deformed mesh/joints/keypoints from the merged parameters,
   apply Body coordinate conversion and project final keypoints. Compare these
   actual final outputs, not only initial decoder coefficients.

There are upstream quirks to preserve in the reference rather than silently
"correct": the wrist-distance denominators reference the opposite hand batch;
the final projection uses image-centred scalar focal length; `pred_pose_raw` is
zeroed after merging; the recomputed local `joint_global_rots` value is not
assigned back into the returned dictionary at this point. Any improved output
contract must be separately named and validated, not presented as raw upstream
parity. Cases accepting neither, one, and both hands must be covered, including
the second body pass and final geometry.

## Completed: hand box and image preparation

`capture_hand_crop.py` executes the unchanged original `_get_hand_box` AST,
the actual left flip/crop statements selected from `run_inference`, original
`prepare_batch`, crop transforms and normalization. It uses the saved trained
hand boxes from the official dancing image plus two synthetic boundary cases.
Every captured output is identical over two independent executions. This is an
explicit **component-input capture**, not a full end-to-end refinement run.

Native `body_hand_boxes` / `body_prepare_hands` pass **69/69 exact tensor
comparisons**, including all normalized crop pixels. The image buffer supports
row padding; the left image is mirrored before cropping. The tiny off-diagonal
residual of OpenCV's affine solve (`−3.6815687e−18` on this image) is accepted
using the existing camera path's relative axis-alignment tolerance. Numerical
comparison remains exact; no acceptance limit was relaxed.

Evidence:

- Original manifest: `88eebe0c0db8c755245d591f9dd41a1af44f83056275c3b26b26ddc63fc4da50`.
- Native comparison report: `a4de774eec623fb3f9f754bc1d0bfc8543c336aecea7fca6a7e64a274b98a865`.
- Native safetensors: `84cc225d458d7d9851d0389208fc2265e064ba693f1119191624e328be5739f4`.
- Normal regression file: `1fefdac23701a2f9b550c8a8559324cebc59ff8a0d96c08b23c3ae95b1c97041` (37,870 bytes, no learned weights).

The normal regression checks all 46 small-case tensors by byte fingerprints,
including pixels, plus invalid box/affine/stride cases under sanitizers. The full
external comparison checks every element directly; fingerprints are only the
compact routine regression. Body's existing default padding is unchanged.
The hand-preparation fuzzer passes 100,000 ASan/UBSan/LeakSanitizer cases
(seed 934, peak reported RSS 362 MiB). It exercises mirrored buffer bounds,
nonfinite geometry, crop sizes and valid pixel preparation, without loading
GGUFs or invoking any learned network. All 41 normal native sanitizer tests and
81 Python tests pass after this change.

Reproduce the original capture in the reviewed offline container:

```sh
python /work/reference/capture_hand_crop.py \
  --upstream /work/reference/upstream/sam-3d-body \
  --trained-reference /work/generated/fixtures/body-trained-decoder-operations-cpu \
  --output /output
```

Then run the native sanitizer-enabled comparison on the host:

```sh
uv run --project reference/python --frozen python scripts/run_hand_crop.py \
  --runner build/debug/bin/sam3d-hand-crop-capture \
  --reference generated/fixtures/body-hand-crop-upstream \
  --output generated/fixtures/body-hand-crop-native-cpu
```

Outputs must be new directories. `make_hand_crop_regression.py` extracts the
two small original cases for the normal test; it excludes the photograph and
all trained weights. Preprocessing executes on CPU for either inference backend.

## Completed: wrist-frame conversion and hand masks

`capture_hand_frame.py` executes the unchanged hand-specific blocks from
`MHRHead.mhr_forward`, with the verified original RoMa implementation. It
observes the actual Euler/matrix calls without replacing their calculations;
observed, unobserved and repeated outputs are byte-identical. Three synthetic
input batches use the real checkpoint's wrist frame/root coordinates and
145-entry non-hand mask. These are component tests, not hand-network inference.

Native `body_hand_frame.cpp` passes all **21/21** comparisons against each
original CPU and CUDA capture. The five geometry boundaries retain maximum
absolute error 1e-4 and relative L2 2e-5; both masks match exactly. These small
geometry operations run on CPU with either neural backend. This does not claim
that a Vulkan hand decoder has been implemented or validated.

The rotation path preserves RoMa's axis-quaternion composition, normalization,
matrix conversion and extrinsic xyz convention. It composes the checkpoint
wrist frame before recovering rotation and translating about the wrist/root.
Masking zeros exactly the checkpoint-defined parameter entries, and all
keypoints outside 21..41. Invalid sizes, nonfinite/overflowed values, duplicate
indices and out-of-range indices are rejected.

Evidence (SHA-256):

- Original CPU manifest: `cd5c0d4e390be058767646bd50fb627dac28941b928b2650f491565c73a8b692`.
- Original CUDA manifest: `82172fd735af6f63e589b575426089ad5434aa3fd7eb3fb0befac10cfaabaa34`.
- CPU-reference comparison: `48905d36db48dd006c016d8dc97c7f42a9318781176f1509e8a4d8006ed414f8`.
- CUDA-reference comparison: `a7fd2ece17089aea0a25c4a606cb9853566ab98a3b8dcf90b3c94769e7358c95`.
- Separate synthetic-buffer comparison: `1fb0c43a352654dfa9604f29eac4eb4a347304b2a1e7e4373968fc697bbaf0b5` (another 21/21).

The normal regression `tests/fixtures/hand-frame.txt` is 86,446 bytes, SHA-256
`d4462abdb2493e2da04fbba588215d753066f498e0683e801baef8389c148dbe`.
It contains **synthetic frame and mask buffers**, not checkpoint buffers;
its generator verifies this from the actual input bytes. Two small original
cases cover ordinary, tiny and near-right-angle rotations. Geometry is compared
elementwise, with byte fingerprints for the full masks. Its adjacent JSON
records provenance. The common matrix-to-Euler helper is shared with the body
head; all 42 native sanitizer tests (including existing body rotation tests)
and 81 Python tests pass after this refactor.

The wrist-frame/mask fuzzer also passes 100,000 ASan/UBSan/LeakSanitizer runs
(seed 936, 136 seconds, peak reported RSS 389 MiB). It exercises finite and
nonfinite rotations, shape bounds and valid/invalid mask indices, and checks
mask semantics independently. It does not load GGUFs or learned models.

## Connected: trained hand head through complete MHR geometry

The internal pose/geometry path now accepts explicit hand-head frame/mask
buffers and the **separate learned hand-head weights**. It transfers globals
before assembling MHR inputs, multiplies translation by the original factor
of 10, applies the non-hand mask after assembly, runs real MHR, masks mapped
keypoints before the 70-keypoint selection, then applies output axis signs.
The returned `global_rot` intentionally remains the original prediction,
matching upstream; it is not the transferred rotation used for deformation.
The ordinary body branch defaults are unchanged.

`capture_hand_head.py` instantiates the actual upstream hand `MHRHead`, loads
every non-MHR tensor from verified safe state with exact assignment readback,
and uses the verified original TorchScript MHR. It checks topology against the
asset. Two batches (one and two samples) use synthetic standard-normal tokens
and the real `init_pose_hand`. Original repeated and observed final outputs
are byte-identical. CPython's repeated line events for multiline calls are
deduplicated in the observer; model statements are unchanged.

These tests exercise **64 operation/geometry taps, all 13 independently captured
final fields including topology, and two exact structural masks per case**:

- NVIDIA Vulkan versus original CUDA: **158/158 pass**.
- ASan/UBSan CPU versus original CPU: **156/158 pass**. All final returned fields
  pass, but the single-sample internal skeleton and centimetre vertices exceed
  the unchanged 1e-4 absolute limit: 0.0001678466796875 cm and
  0.000179290771484375 cm respectively. CPU full-head acceptance remains open.

`diagnose_hand_head.py` is a diagnostic intervention, not acceptance. Original
MHR replay on its own captured inputs is byte-exact. On native-produced inputs
it reproduces the same maximum drift against the original; native versus
original MHR with those identical inputs differs by only 2.6702880859375e-5 cm
in vertices and 4.76837158203125e-7 in skeleton state. Rotation and translation
input rounding dominate. The first nonzero head difference is its first linear
projection (9.5367431640625e-7), still within the layer limit; error propagates
through global Gram–Schmidt/Euler conversion and the wrist transfer. This is
why passing individual layer limits does not establish composed geometry parity.
No tolerance was relaxed and no pose/geometry was substituted in acceptance.

Evidence (SHA-256):

- Original CPU manifest: `b842b32e9361818e401c73b79d6e6f345aec83a913b851d163f3213057f74e72`.
- Original CUDA manifest: `1f3ed932bd835b3fb17fb0ad194c6489253febd2bb01a0d2353ab80d4f07ae0b`.
- CPU report: `9911e844fc2dffd801448fcce28c077f33c1e0d0a19a8cbd7b73fc17b0ac5f19`.
- Vulkan report: `846a73c0e93524d5f22de46eb70cfe15ca797fed82a4c53d42a2b2a2f26849f5`.
- Vulkan tensors: `a35ab61cca0a6d5b308c03ac9ff4d4047aadd54285f367638a9ae3899041570c`.
- Original input-intervention diagnostic: `8927d97f461a0abd122ebd2adb25137795678ccaa5c11747973e071775ac2e1d`.

The normal pose/mapping tests additionally verify integration order, unchanged
returned globals, exact masks, axis signs and malformed hand-buffer rejection.
Comparison-tool negative tests reject missing/extra fields and detect bad
vertices, topology, shapes and nonfinite output. All **42 native sanitizer
tests and 85 Python tests pass**. Learned fixtures remain ignored generated
data, not normal-test dependencies.

This standalone test is learned **head** inference, not a trained hand-image
branch. Separate composition must produce its own backbone/decoder tokens,
and full refinement must perform validity checks, body reprompt and IK/merge.
The public model capability therefore still advertises the body pose branch
only. These internal head captures consume inline diagnostic head weights,
not a new hand GGUF companion or a new public hand model API.

## Learned hand decoder: connected, numerical acceptance open

`body_flow.cpp` now selects the original hand-specific modules, including
independent dense image positional encoding, while retaining shared sparse
prompt weights. Its six decoder layers feed their own learned hand head,
wrist transfer, masked MHR geometry, hand camera (scale factor 10) and keypoint
feedback into subsequent layers. No reference intermediate is injected.
The reference captures unchanged `forward_decoder_hand`, camera projection and
both hand keypoint callbacks, with exact trained-state assignment and repeated/
observed output identity. Inputs are synthetic features and camera conditions;
this is not an image-based hand reconstruction or full refinement acceptance.

There are 478 operation/boundary tensors and 115 independently captured final
fields. CPU passes **328/593**, NVIDIA Vulkan **383/593**. All 34 first-layer
decoder operation checks pass. The first failed boundary in execution order is
the first-layer vertex projection: CPU 0.005859375 px and Vulkan 0.00634765625 px,
above the unchanged 0.001 px limit. Feedback then amplifies small differences.

`compare_hand_flow_controls.py` compares original CPU and CUDA with identical
input bytes, state and observation checks. It passes only **330/593** under the
same limits; its first-layer vertex projection differs by 0.0068359375 px.
This is evidence of reference sensitivity, not a new tolerance policy or a
passing native result. The failed synthetic stress cases remain retained.

An eight-way balanced Vulkan decoder dot-product experiment improved hand
checks to 416/593 but regressed the already accepted real-image Body branch
to 643/646. It was rejected and reverted; the accepted DINO backbone arithmetic
is unchanged. Diagnostic artifacts remain under ignored `generated/fixtures`.
After the revert, `body-gguf-model-native-vulkan-hand-wiring-regression` passes
all **646/646** real-image Body checks. Its native tensors are byte-identical
to the previously accepted GGUF-only result, SHA-256
`a77f6554e0e22a16150a7acaf3e0d6bf0fd684cca4cc46e040e8b14155f6c424`.

Evidence (SHA-256):

- Original CPU manifest: `51e17bc1d1d3894d98704e5e9e55f0f25e676faa295b51dc8d3bd54162db017a`.
- Original CUDA manifest: `e02448589dfced195906414a1d4e579b32b1e1e3e1a8e0ed2c4df6fc30258713`.
- Native CPU report: `faa18352276ce2384eb129bdcb5e7e4a2410b907a1310bdbc82785d5147cd847`.
- Native Vulkan report: `4d6f871603b59553d71bcc34840e48cd04f65efeff954bcc5642faaf005ed531`.
- Original cross-backend control: `15b86ec583eee7c017f816f4ff607a72caf40b8a6c6e1cdbf870dc340877f8e0`.
- Rejected balanced-decoder hand report: `8a1fec0baa738c7ee988e76612a82ecd48cbb1c43ddaec8be219832355f0e636`.

Normal tests cover hand module identity, separate dense positional encoding,
invalid buffer rejection and rejection of an incomplete raw-image contract.
All **42 native sanitizer tests and 88 Python tests pass**. No hand GGUF schema
or full-hand public capability is advertised yet.

## Next gates

Close the two strict CPU hand-head geometry comparisons without weakening
limits. Resolve the actual hand decoder/feedback numerical failures and
the [new learned hand-image composition](HAND_IMAGES.md) checks.
Hand images need their own original CPU/CUDA precision controls; do not reuse
a frozen full-image backbone policy without satisfying its cohort requirements.
Extend GGUF identities and the public capability only when the complete learned
hand path is present. Finally validate all validity decisions, body reprompt,
IK/merge, final vertices and projections from each pipeline's own intermediates.
