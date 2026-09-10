# Single-image demo: real end-to-end evidence

Verified 2026-09-09. This milestone is the **trained F32 Body pose branch**,
not the complete hand-refined estimator, automatic detection, video tracking
or performance parity. [Build and use the demo](../demo/README.md).

## Real browser path

Headless Chrome 151.0.7922.173 uploaded a lossless PNG of the official
`notebook/images/dancing.jpg` capture through the actual file input. It selected
the recorded box `[600,80,1330,1250]` and camera
`[2704.16357421875,2704.16357421875,1125,750]`. The browser's generated request
hash is exactly the original RGB/box/intrinsics request:

`3badb322c580a89275c8f4257a70decad730aa0ef30f2816466d4b0e3e88366b`

The uploaded image does not carry a reference embedding, pose or mesh. The
native public-C-API executable loads three real GGUFs and runs all preprocessing,
backbone, decoder and geometry using its own intermediates. Vulkan UBSan is
enabled on the NVIDIA GPU; strict F32 flags are unchanged from accepted parity.
The browser uses software WebGL/SwiftShader for presentation, independently of
the NVIDIA native inference backend.

The first real job took 46.4 seconds; the final post-fix job took 43.2 seconds,
including model loading and exports. Both produce the exact same binary:

`d0930b74259442c3213e6827000bb75c41d884a53445b99fae6c74c59688aa7c`

These are observed demo latencies, **not optimized benchmark or performance
parity results**. Each worker was capped at 6 GiB RAM with no swap; systemd
reported about 1.4 GiB peak for the completed native scopes. The server has
its own 1 GiB cap. A real cancellation terminated its scope and the next
generation completed normally. No other reference/model job ran concurrently.

## Original comparison and exported geometry

All 19 returned fields are byte-identical to the previously accepted native
646-check Body run. Independently, the demo result is compared directly with
the original captured PyTorch CUDA final outputs, without any alignment,
rescaling, smoothing or tolerance changes:

| Final output | Maximum absolute error | Existing limit |
| --- | ---: | ---: |
| All 18,439 mesh vertices | 8.34465e-7 m | 1e-4 m |
| All 127 joint positions | 4.58211e-7 m | 1e-4 m |
| All 70 keypoints | 4.76837e-7 m | 1e-4 m |
| Camera translation | 1.43051e-6 m | 1e-4 m |
| All projected mesh vertices | 0.000854493 px | 0.001 px |
| All projected keypoints | 0.000610352 px | 0.001 px |

Relative-L2 checks also pass the existing `2e-5` limit. All 36,874 triangle
indices agree exactly. The complete JSON transport round-trips every F32/I32
field exactly. GLB reload verifies every exported vertex and index against
the native result after the documented fixed axis conversion, then constructs
and renders the reloaded geometry in Three.js. The GLB SHA-256 is:

`f1b339176a162a268fd0a7d88a4f4bb9e08c110afed7a0634c1065ecda7cc78e`

Original comparison provenance: pinned Body source
`b5c765a0d89d789985e186d396315e7590887b94`, original capture manifest
`fd97d6a192aeb66d8fe71ef1af149b9a876a5e7a1f2fb11286085069dbb732df`,
original final tensors
`33628ff23fbe56b9a2eaf4f4ea899afa5a572b43792d32a6caf3bf3a06d87d85`.
The orange reference is the original F32 body branch, not a screenshot from the
live Meta gallery or a full refined-hand reconstruction.

## Browser and normal tests

The final real-model browser run passes:

- invalid/oversized file rejection without showing a stale previous result;
- actual person-box dragging and explicit camera entry;
- native generation, visible stage/elapsed state, cancellation and worker recovery;
- orbit/front/side viewing and same-camera original/native overlays;
- finite body/camera values, nonzero draw calls, zero WebGL errors;
- GLB download, every-vertex/index verification and reloaded geometry rendering;
- settings edits detaching stale outputs, history restoring settings and results,
  and selection/result persistence after page reload;
- desktop sidebar scrolling, narrow mobile viewing and history scrolling,
  with no horizontal overflow or uncaught browser/network errors.

Actual screenshots were visually inspected: the reconstructed leaning torso,
raised arm and bent leg agree with the photograph, and original/native geometry
overlaps in front and side views. The initial renderer drew MHR's artificial
`body_world` → `root` connection as a long bone below the pelvis. This viewer-only
artifact was removed from both overlays; result coordinates were not changed.

The first QA attempt failed a decimal-camera restoration assertion: Go emits
the shortest decimal spelling of F32, whereas the assertion compared F64 decimal
values. The test now compares the actual F32 value, and complete packed-input
hash equality remains mandatory. The failed report is retained, not relabeled.

Normal native regression: **42/42** under ASan/UBSan/LeakSanitizer. Python:
**96 tests**. Go tests cover bounded image/result parsing, exact exports, queued
cancellation, restart recovery, origin checks, queue limits, serialized decoding
and static assets, and pass with Go's race detector. A short one-worker Go parser fuzz smoke passes; it is not
the existing 100,000-case native C API fuzz acceptance.

Local, ignored final evidence:

- `generated/demo-qa-final/report.json` — browser report
  (`de10ec52d5864c9b3b6f7b9fe46a80eceaae14543c064b747ff02ff613887b96`);
- `original-parity.json` — direct original comparison
  (`8376b39d0a7c3698a331ba16c29eb044a88e1d91a1edee1c575222dc220b4659`);
- `native-api-parity.json` — every-field accepted-native comparison
  (`2097e744a3d47c22cd18944512f8e657feebf290e53a851d282dd5fc4d9df09e`);
- eight screenshots from initial/upload, native front/oblique, original overlays,
  and mobile viewer/history states; `chrome.log` contains browser diagnostics.

`generated/demo-qa-handoff/report.json` repeats the browser checks against the
final running build using the completed real job, additionally exercising the
official-example button. It passes with zero browser errors. This is a
presentation rerun, not another inference or a substituted reference result.

Full CPU numerical acceptance, learned hand refinement and optimized upstream
CPU/CUDA versus native CPU/Vulkan performance remain separate unfinished gates.
