# Skeleton GLB export

The demo exports the native MHR LOD1 hierarchy from photos, offline video and
recorded webcam takes. `demo/skeleton.go` converts global MHR states to parent-local
TRS using the embedded 127-joint topology. The artificial `body_world` node is
retained, under one `SAM3D` group node. No orientation is inferred from joint
positions. Quaternion signs are made continuous between samples.

The output is glTF 2.0 GLB with 128 named nodes, no mesh/skin, and one `Take`
animation when at least two samples are selected. Each joint has translation,
rotation and scale channels; the group has a translation channel. LINEAR samplers
interpolate translations/scales and spherical-interpolate rotations. Source
sample intervals are preserved, including dropped live frames. The first selected
sample becomes time zero; its source timestamp is in `extras.sourceStartSeconds`.
A single pose has no animation or binary chunk. Importers must support node-only
scenes; a Blender armature or a retargeted character is not created automatically.

Native globals are `[127,8]` F32: XYZ centimetres, quaternion XYZW, uniform scale.
They are exposed in the public `joint_transforms` result tensor. Translation is
converted to metres; original MHR axes already match Y-up export coordinates.
Public `joints` instead use `diag(1,-1,-1)` and metric units. Camera translation
uses that same sign conversion when applied to the GLB group.

`in-place` holds the pelvis at its first position on all three axes for a sequence.
`camera` applies each frame's estimated camera translation. Neither estimates a
world frame or ground plane. Raw proportions/scales and pose-branch hand estimates
are retained; no filtering, retargeting, body-shape lock or refined hands are added.

## API

| Method and path | Behavior |
| --- | --- |
| `GET /api/jobs/ID/skeleton.glb` | Static skeleton from a completed photo result |
| `POST /api/tracks/LIVE_ID/record` | Start an independent saved take; JSON `{ "time": 1.25, "name": "Take" }`, name optional |
| `POST /api/tracks/LIVE_ID/record/stop` | Finish the active take while tracking continues |
| `GET /api/tracks/ID/skeleton.glb` | Export a finished video or saved take |
| `PATCH /api/tracks/ID` | Rename a saved video/take with JSON `{ "name": "Take" }` |
| `DELETE /api/tracks/ID` | Delete an inactive saved video/take and its files |

Both GLB routes accept `movement=in-place` (default) or `movement=camera`.
Track export also accepts inclusive `start` and `end` seconds in the saved
sample timeline. It selects samples without synthesizing trim-boundary poses.
An interval containing no sample is rejected. Export reads saved transforms and
never invokes inference. Legacy results without transforms require reprocessing.

The recording start time uses the same capture clock as live frame requests.
Frames submitted before recording, or captured before its start time, are excluded.
Only completed inference results persisted before Stop belong to the take.
Frame replies report `X-Take-ID`, `X-Take-Count` and `X-Take-State` when applicable.
Ending a live session finishes its take. Server restart marks unfinished saved
takes interrupted and retains completed poses. Each take is capped at 1,800
samples and shares the configured history/storage budgets.

## Saved skeleton frames

Offline videos retain their mesh `.bin` frames and add skeleton `.pose` sidecars.
Live takes save only `.pose` frames, a first-frame JPEG thumbnail and `track.json`.
The manifest has `mode: "take"`, `skeleton: true`, timestamped samples, inference
precision and model provenance. Live timestamps are relative to Record; offline
video timestamps refer to source-video seconds.

Every `.pose` file is exactly 4,092 bytes, little-endian with no padding:

| Bytes | Value |
| --- | --- |
| 0–7 | ASCII `S3DSKL01` |
| 8–15 | F64 sample timestamp in seconds |
| 16–4079 | F32 global MHR transforms `[127,8]` |
| 4080–4091 | F32 camera translation `[3]`, public body coordinates in metres |

## Verification

```sh
(cd demo && CGO_ENABLED=0 go test ./...)
node tests/test_tracking_math.mjs
node tests/test_live_presentation.mjs
node tests/test_frame_pipeline.mjs
# Chromium uses a simulated camera; no real camera or model is needed.
(cd demo && SAM3D_TEST_NODE=node SAM3D_TEST_CHROMIUM=chromium \
  CGO_ENABLED=0 go test -run TestSkeletonBrowser -v)
# Optional real current-runner capture; writes skeleton.glb alongside result.bin.
(cd demo && SAM3D_TEST_RESULT=/absolute/path/result.bin \
  CGO_ENABLED=0 go test -run TestSkeletonNativeCapture -v)
```

Tests independently reconstruct global matrices from serialized GLB nodes and
animation channels. They check positions, orientation, scales, timestamps,
quaternion continuity, recording boundaries, persistence, trimming and legacy
rejection. Browser QA records with a simulated camera, saves/reopens the take,
checks skeleton rendering and downloads the animation. Native capture QA compares
exported FK against public joint positions and global rotation matrices.
