# Body image and video demo

Upload a photograph, drag a box around one person, then select **Estimate 3D
body**. The photo/skeleton overlay and orbitable body mesh show the same native
result. Previous reconstructions restore the input image, box and camera settings.
Mesh GLB and OBJ downloads are static posed meshes in metres, Y-up.
**Skeleton GLB** exports the named 127-joint MHR hierarchy: a static photo pose
or an animation from a video/live take.

This is the working **body pose branch**, not the full hand-refined estimator.
There is no automatic person detection, camera estimation, texture generation
or learned temporal model. Video tracking applies the same estimator to sampled
frames of one selected person. The default focal length is the image diagonal;
known camera intrinsics can be entered explicitly. The model operates on its
own padded crop, so the selection rectangle is not a hard pixel mask.

## Offline video and live webcam

Select **Offline video**, choose a browser-decodable video (typically MP4/WebM),
select one person in the preview, set the source start time, sampling rate and duration, and start.
Frames are processed sequentially; offline processing does not discard samples
to keep up with playback. Completed poses persist in video history, including
partial sequences stopped by the user. Play, pause and scrub the saved sequence, then choose trim start/end times and
**Export skeleton GLB**.
The source video stays in the browser; only sampled JPEG frames are uploaded.
The current source can be scrubbed alongside the poses. After a history reload,
only the saved first-frame thumbnail is available, not the original video.

Select **Live webcam**, grant camera access, select the person, then start.
**A remote webcam requires HTTPS** (localhost is also allowed). An HTTP URL on
a LAN or Tailscale IP cannot normally request camera access. Put the demo behind
a trusted HTTPS reverse proxy; no browser security override is required or
recommended for users. Permission denial, unavailable codecs and disconnected
cameras are reported in the UI. Stop, changing modes, leaving the page or hiding
the tab releases the camera and cancels pending live work.

The configurable **maximum inference Hz** is a cap, not a throughput guarantee
(1–30 Hz, default 10). The browser permits two ordered live requests while
preparing frames in a dedicated encoding worker. The server can decode and pack
one frame while the preceding frame runs inference, with one bounded decoded
frame slot. There is still only one replaceable encoded frame and one active
encoding; live mode drops stale camera frames instead of building a backlog.
The server also enforces the live cap. The displayed recent Hz
uses the last 20 request intervals, not the entire session including cold load.
Live results start displaying on the next render, with a **25 ms blend from the
currently displayed pose** and no delayed playback timeline. Gaps snap to the
new result; nothing is predicted or extrapolated. This prioritizes reaction time
over perfectly even motion between estimates. Offline playback retains timestamped
interpolation. Raw estimates are never smoothed or rewritten on disk. Blending
is presentation only; skeleton exports use the raw recorded estimates.
The live photo preview is current; its pose overlay still trails by processing
latency, but there is no additional multi-frame playback buffer.

Optional **Follow projected person box** uses bounded, smoothed projected joint
extents for the next crop. Turn it off for a fixed selection. It is not an
identity tracker: use one visible person, and stop/reselect if tracking drifts,
the subject leaves the frame, or another person crosses them. Camera calibration
and monocular depth/shape jitter remain limitations of independent frame estimates.

Frames are resized in the browser to fit 960×960 without upscaling and JPEG
encoded at quality 0.94. Box/camera values refer to these resized pixels. The
worker uses transferable video frames where available, with bounded CPU pixel
transfer and main-thread canvas fallbacks. JPEG compression runs off the UI
thread on the worker path. When the browser exposes `MediaStreamTrackProcessor`
on the main thread, a one-frame camera reader supplies original decoded frames
before video-element rendering/readback. Other browsers retain video-element
capture. Superseded source frames are explicitly closed; stop also releases the
reader and its cloned camera track. Stop, mode changes and page exit terminate the worker;
capture/encoding never queues unlimited work. No WebRTC dependency is required.
In the matched standard-mode headless test, new results reach first render
submission in about **176 ms**, settling in **202 ms**, at **8.1 Hz**. With the
deployed approximate BF16 `fast384` mode and the two-request pipeline, a strict
10 Hz cap measures **9.85 Hz** (100.7 ms median request interval); raising the
cap to 30 Hz measures **14.25 Hz**. These exclude camera exposure and physical
display delay; see [latency evidence](../reference/BODY_LIVE_PERFORMANCE.md#follow-up-low-latency-presentation-and-camera-capture)
and the [Fast384 pipeline profile](../reference/BODY_LIVE_PERFORMANCE.md#fast384-two-request-pipeline).
The server accepts at most 1 MP / 2 MiB per frame. Offline sequences are capped at
1,800 samples and share the configured data budget. Live tracking keeps geometry
only while displaying it unless **Record take** is active.
One queue consumer owns native inference across still images and all video clients.
Busy replies provide backpressure; cold loading and inference failures are visible,
and cancellation discards/reaps the worker before another request can reuse it.

Offline data lives under `DATA/tracks/ID/`: `track.json` records timestamps,
per-frame box/intrinsics, precision and model hashes; `faces.bin` stores shared
uint32 triangle indices; `000000.bin`, etc. store raw sampled geometry. Each frame
is little-endian: 8-byte `S3DTRK01`, float64 source timestamp (seconds), then F32
vertices `[18439,3]`, joints `[127,3]`, camera translation `[3]` in native body
coordinates. Exactly 222,820 bytes per frame; no alignment padding. The browser
applies the same fixed Y/Z sign conversion as photo exports. These are independent
posed meshes. New sequences also save compact `000000.pose` skeleton sidecars
for animation export without rerunning inference.

HTTP endpoints: `POST /api/tracks` creates a session; `POST /api/tracks/ID/frame`
accepts a PNG/JPEG body with `time` and JSON `settings` query parameters;
`POST /api/tracks/ID/finish` ends it; `GET /api/tracks` lists videos and saved takes.
Frame replies use the binary layout above and append shared triangle indices
only on the first reply. Saved files are under `/tracks/ID/NAME`. Live sessions
are temporary and expire; there is no cross-client broadcast or authentication.

See [video QA](../reference/BODY_VIDEO.md) for actual browser evidence and limits.

## Skeleton export and recording

While live tracking is running, select **Record take**. **Stop recording** saves
that take while tracking continues; Record starts another take. History supports
replay, rename, delete and GLB download. Stopping tracking also saves its active
take. Saved poses survive restart; an interrupted recording retains completed
samples. Frames already in flight when recording starts, or unfinished when it
stops, are excluded. An empty take can be deleted but has no exportable pose.

The default **In place** export fixes the pelvis position on all three axes at
the first selected pose, retaining orientation and joint motion. **Estimated
camera-relative** adds the estimated camera translation; this is not recovered
world motion. Trim selects existing samples inclusively and rebases animation
time to the first selected sample. A single selected sample exports a static
pose. Changing trim or movement does not rerun inference.

The GLB contains named parent/child nodes with local translation, quaternion
rotation and scale, in metres and Y-up. It has no character mesh or skin and is
not automatically a Blender armature. Importers must preserve empty nodes and
node animation. Body proportions can vary between frames because exports retain
raw estimates. Retargeting, fixed proportions, temporal cleanup and refined
hands are outside this implementation. Older results without joint transforms
must be reprocessed to enable skeleton export. Rebuild both the native runner
and Go demo together for the expanded result schema.

See [skeleton export format and tests](../docs/SKELETON-EXPORT.md).

## Build and start on Linux

Build the native library and CLI using the root README. Install Go 1.23 or newer
to build the optional server. No third-party Go modules, Node build step,
Python inference runtime, CUDA runtime or sibling checkout is required.

```sh
# From the source root, with a Vulkan SDK installed.
cmake --preset vulkan-optimized
cmake --build --preset vulkan-optimized --target sam3d-body-infer -j2
(cd demo && CGO_ENABLED=0 go build -o ../build/sam3d-demo .)

./build/sam3d-demo \
  --listen 127.0.0.1:8097 \
  --data generated/demo \
  --runner build/vulkan-optimized/bin/sam3d-body-infer \
  --backend Vulkan --module /path/to/libggml-vulkan.so --device 0 \
  --backbone /path/to/body-dinov3-f32.gguf \
  --branch /path/to/body-pose-branch-f32.gguf \
  --mhr /path/to/mhr-lod1-f32.gguf
```

Use `--backend CPU` with the CPU module and an `optimized-sanitizers` CLI build
for ASan/UBSan CPU execution. Vulkan on the affected NVIDIA driver retains
UBSan; see the root README for that diagnosed ASan limitation. Strict F32 GGML
flags are set before initializing each native worker. Device/module selection remains
explicit; `--device-name` can require the exact GPU description. Configure the
Vulkan ICD through the usual environment if the host needs it; no machine-specific
paths are embedded in the application.

### BF16 image encoding on NVIDIA Vulkan

The optional `--precision bf16` mode uses BF16 image encoding with F32
decoder/MHR outputs. On Vulkan it requires NVIDIA cooperative-matrix-2 support
and our reviewable GGML build-copy patches. For the live demo, use the production
build (optimized, assertions retained, no sanitizers or fast-math):

```sh
cmake --preset vulkan-bf16-production
cmake --build --preset vulkan-bf16-production --target sam3d-body-infer -j2
```

In the launch command above, use `--precision bf16`,
`--runner build/vulkan-bf16-performance/bin/sam3d-body-infer` and
`--module build/vulkan-bf16-performance/bin/libggml-vulkan.so` (or the module's installed
location). No model reconversion is needed. The server selects the validated
precision flags, including precise class/register attention and bounded
pinned-transfer batching, exact SIMD CPU skinning, narrow F32 GEMM tiles,
BF16 affine/norm fusions and scalar image gathering,
before starting
either a persistent or one-shot worker. An incompatible backend fails explicitly;
there is no silent lower-precision fallback. CPU BF16 uses math attention.

The default remains `--precision f32`. The UI and saved jobs identify their
precision; selecting an older F32 history entry does not relabel it BF16.
The accepted BF16 candidate passes two official-image encoder/final-body policies;
one dancer hand-logit difference remains reported. This is not a claim of
full hand-refined estimation or universal image accuracy. The unsanitized warm
native benchmark is 85.8 ms, excluding uploads, model loading and exports.
Use `vulkan-bf16-cm2` for UBSan correctness checks, and CPU `debug` for
ASan/UBSan/LSan; only the production performance/deployment build disables them.
See [precision evidence](../reference/BODY_PRECISION.md) and
[live stage profiling](../reference/BODY_LIVE_PERFORMANCE.md).

The default bind is localhost. `--listen 0.0.0.0:8097` is an explicit trusted-LAN
deployment option. **There is no authentication**: do not expose this directly to
the public Internet. Uploaded photographs and derived bodies are private data;
protect the data directory and any reverse proxy. No images are sent to Meta.

### Optional faster body inference

`--body-inference` selects a fixed mode for both persistent and one-shot workers:

| Mode | Crop | Intermediate predictions | MHR correctives |
| --- | --- | --- | --- |
| `standard` (default) | 512 | 0,1,2,3,4 | on |
| `no-correctives` | 512 | 0,1,2,3,4 | off |
| `fast512` | 512 | 0,1,2 | off |
| `fast448` | 448 | 0,1,2 | off |
| `fast384` | 384 | 0,1,2 | off |

The fast presets also omit unused intermediate outputs. All six transformer
layers and the final full body/skeleton prediction still run. The setting combines
with `--precision f32` or `bf16` and requires no new weights. Restart the server
to change it. The UI model label and saved-job provenance identify approximate
modes; image, video and live jobs use the same setting.

These modes change the predicted pose. On two test images, BF16 `fast384` reduced
warm inference from 85.46 to 54.90 ms, but joint differences reached 201 mm.
See [the complete measurements and reproduction commands](../reference/BODY_FAST_INFERENCE.md).

## Worker, limits and progress

One native subprocess runs at a time, with a queue of two waiting jobs. The
default Linux/systemd worker scope has a **6 GiB hard RAM limit**, 5 GiB high
watermark and no swap. A snapshot check requires another 10 GiB of available
host RAM before starting. An unavailable memory guard fails the job rather than
silently running without limits. `--memory-mib` and `--reserve-mib` configure
these values. On a non-systemd host, impose an external memory cap first, then
explicitly use `--memory-mib 0`. Also bound the server itself in deployments;
our local QA uses a separate 1 GiB server cap. This does not reserve memory
against growth by unrelated applications.

Jobs show queue/running state, model loading, estimation, export and elapsed
wall time. These are actual stage messages, not simulated percentages. The
encoder/decoder call currently has no finer-grained public progress callback.
Cancel terminates the worker's entire scope; timeout defaults to ten minutes.
Interrupted jobs remain visibly failed after server restart and can be regenerated.
The default persistent worker reuses its model across images. Failed/cancelled
workers are discarded and restarted on the next request. `--worker-idle 2m`
releases idle RAM/VRAM within one additional timer interval (2–4 minutes with
the default); shutdown also reaps the worker. Use `--persistent-worker=false`
for a fresh-process job instead. One Vulkan model currently occupies about
4.4 GiB device memory; the cold example takes about 2.67 s, and a warm browser
job about 0.46 s including exports. This is not yet real-time or performance
parity with optimized upstream; see [profiles](../reference/BODY_PERFORMANCE.md).

Original photo uploads may be up to **128 MiB**. Before person selection,
`POST /api/prepare` checks format, dimensions and size. Ordinary PNG/JPEG images
within 20 MiB / 16 million pixels pass through unchanged. Otherwise the server
attempts local **FFmpeg** conversion, preserving aspect ratio and fitting within
4000×4000 without upscaling. If the resulting PNG exceeds 20 MiB, it retries at
2500×2500. Browser decoding/PNG-size failures also trigger a conversion attempt.
The converted preview becomes the actual input; box/camera coordinates refer to
that preview. A persistent UI notice records the conversion (and is saved with
generated jobs), or explains why FFmpeg failed. No model runs on a failed input.

Install FFmpeg separately, or set `--ffmpeg /path/to/ffmpeg`. The build must
support the seekable `fd:` protocol (verified with FFmpeg 8.1.2). Conversion is
optional: valid PNG/JPEG still works if FFmpeg is absent. Other raster formats
such as GIF, WebP, BMP and TIFF depend on the installed codecs; HEIF/AVIF support
also varies. Only the first image/frame is retained, not a video sequence.
Transparency is composited on black and orientation handling is reflected in
the preview. New uploads reset person selection and estimated intrinsics.

FFmpeg is executed directly with a fixed argument list, not through a shell.
Input uses an inherited descriptor; its [protocol whitelist](https://ffmpeg.org/ffmpeg-protocols.html#Protocol-Options)
disallows arbitrary file paths and network URLs, and a format whitelist excludes
playlists/scripts. It gets two codec threads, bounded output/diagnostics, a
45-second request deadline, and a separate 768 MiB/no-swap scope when systemd
memory guards are enabled. Original bytes are streamed to a private temporary
file and removed after the request. Disconnect/cancellation stops the scope.
With `--memory-mib 0`, external memory limits must cover conversion as well.
Photo preparation and image decoding are serialized.

The server defaults to 100 history entries and a 2 GiB data budget, reserving
space for queued results. It rejects new work when full instead of deleting
history. Archive old job directories while the server is stopped, then restart.
History/storage/worker budgets are parameterized; uploaded names are never used as disk paths.

Each random-ID job directory contains:

- `input.png`: exact normalized source photo;
- `image.input`: native RGB, box and camera request;
- `result.bin` / `result.json`: complete native public result;
- `body.glb` / `body.obj`: downloadable static body mesh;
- `job.json`: settings, state, timestamps and input/model/output hashes;
- `inference.log`: bounded native diagnostics.

The viewer and exports apply the same fixed `diag(1,-1,-1)` coordinate change
to the API's body-space vertices/joints. No scale normalization or mesh-origin
translation is baked into exports. The viewer camera frames the native body;
upstream overlays share that exact camera and origin. The non-anatomical MHR
`body_world` marker and its link to `root` are not drawn as a bone; all 127
joint coordinates remain untouched in the results. The ground is a visual
reference placed below the native mesh, not an inferred world plane.

## Optional original-upstream comparison

If you have the verified reference captures described in
[the branch evidence](../reference/TRAINED_BODY_BRANCH.md), prepare the optional
example without running a neural model:

```sh
python scripts/prepare_demo_reference.py \
  --reference generated/fixtures/body-trained-image-flow-cuda \
  --safe-state generated/extraction/body/body-other-state.safetensors \
  --output generated/demo-reference
```

Start the server with `--reference generated/demo-reference`. The official
dancing-example button loads a lossless PNG of the captured RGB and the same
person box/intrinsics. **You must still run native inference.** The orange
comparison is original PyTorch output, never a native result or inference input.
It becomes available only when the complete RGB/box/camera request hash matches.
Both outputs use the same scene/camera, with no independent alignment.

The optional fixtures/images are local, ignored artifacts; they are not bundled
or downloaded by the server. Their manifest identifies the official photograph,
upstream revision, capture and geometry. This compares the same F32 body branch,
not the full hand-refined pipeline or an unversioned Meta gallery screenshot.

## Verification

```sh
(cd demo && CGO_ENABLED=0 go test ./...)
# Real model-backed browser QA; creates a cancelled job and a complete result.
python scripts/qa_body_demo.py --reference generated/demo-reference \
  --output generated/demo-qa
python scripts/check_demo_result.py --job generated/demo/JOB_ID \
  --reference generated/demo-reference --report generated/demo-qa/parity.json
```

The browser test uses local Chromium and Python's standard library, with no
Playwright/Selenium dependency. It exercises file upload, bad/oversized input,
box selection, real inference/cancellation/recovery, 3D camera controls,
upstream overlays, GLB reload, history/settings restore and mobile scrolling.
It fails on browser errors, nonfinite geometry and mismatched artifact identity.
Screenshots must also be visually inspected. Use a fresh data directory for an
isolated run; `--reuse JOB_ID` can inspect a completed real job without inference.
Numerical checking uses the development NumPy/safetensors environment; it is
not part of the server or native runtime.

`scripts/qa_photo_prepare.py --url URL --output generated/photo-prepare-qa`
separately exercises real FFmpeg conversion and errors through the browser,
without neural inference. The conversion notice is saved as `preparation_note`
in each generated job. Selected history entries are retained across reloads in
the browser tab's session storage.

## CLI wire formats

`sam3d-body-infer` is also a minimal standalone consumer of the opaque public
C API. Run it without arguments for usage. Its interchange is little-endian:

- Request: `S3DIMG01`, three U32 values (width, height, RGB row stride), four
  F32 XYXY box coordinates, four F32 `fx,fy,cx,cy` values, then tight RGB U8.
- Result: `S3DOUT01`, U32 tensor count, then for each tensor: U32 ASCII-name
  length, name bytes, U32 dtype, U32 rank, U64 element count, U64 dimensions,
  and four-byte F32/I32 elements. See [API](../docs/API.md) for the exact schema.

The server rejects truncated/extra bytes, missing/duplicate fields, incorrect
shapes/types, nonfinite values and out-of-range mesh indices. Direct library
consumers should generally use the C API rather than this demo-specific format.
