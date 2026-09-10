# Production live pipeline profile

2026-09-10. The live demo now runs the `vulkan-bf16-production` preset:
optimized native code, assertions retained, sanitizers disabled, no fast-math.
The existing CPU ASan/UBSan/LSan and Vulkan UBSan correctness configurations
remain separate. BF16 is the encoder policy; decoder and MHR remain F32.

The server enables the already validated affine/norm fusions and scalar image
gather, alongside the previously accepted precision/transfer/skinning settings.
No new model approximation or acceptance tolerance was introduced.

## Native validation

Five warmups and twenty timed alternating official dancer/rider inputs:
**85.79 ms median, 86.17 ms p95**. All 25 complete outputs are byte-identical
to the accepted `bf16-host-performance-v3` baseline. Dynamic dependencies of
the deployed native library contain neither ASan nor UBSan. Native timing
excludes model loading, HTTP, image serialization and result file output.

Artifacts: `generated/diagnostics/live-production-native-v1/` and
`live-profile-build-v2*`. These are ignored local diagnostics, not release assets.

## Actual browser profile

The table below is the **pre-pipeline baseline**. The later sections record the
worker/prefetch experiment and the now-deployed low-latency implementation.

Chrome 151 virtual webcam, 960×540 input, JPEG quality 0.94, fixed person box,
10 Hz cap, one inference in flight, the real native Vulkan backend and a private
HTTPS reverse proxy. No insecure-origin override. Five warmups then forty
measured requests per run; no fake inference responses or persisted generations.

Two runs with explicit SwiftShader rendering agree: **4.48 / 4.53 Hz**.
The browser and inference run on the same host, so this is not a benchmark of a
remote hardware-accelerated browser or physical camera. Median times from the
second stable run:

| Stage | ms |
| --- | ---: |
| Browser canvas capture submission | 0.2 |
| Await browser `canvas.toBlob()` | 81.5 |
| Entire HTTP request through geometry decode | 139.1 |
| ↳ Server request until reply ready | 130.7 |
| ↳ Server JPEG decode | 7.6 |
| ↳ Server RGB input packing | 18.4 |
| ↳ Native inference | 98.2 |
| ↳ Native call, file output and worker protocol combined | 98.8 |
| ↳ Server result parsing/validation | 3.2 |
| Browser response-body read | 2.1 |
| Browser geometry decode | 1.3 |
| Total capture through decoded result | 221.6 |

Nested stages overlap: do not add every row. In particular native inference is
included in worker, server and HTTP request timings. The HTTP-minus-server gap
includes browser/proxy scheduling and transfer, not just wire latency. Capture
submission does not necessarily complete deferred canvas pixel work.

The browser's JPEG timer measures API completion latency, **not pure JPEG CPU
compression time**. As a control, after stopping inference, the same frozen
pixels/quality in the same browser gave median 27.1 ms for `toBlob()`, 17.5 ms
for synchronous `toDataURL()` and **6.4 ms for `OffscreenCanvas.convertToBlob()`**.
Each control has three warmups and twelve timed calls. This supports capture,
canvas readback and scheduling as important contributors; it does not identify
their individual costs or establish the gain of replacing the live path.
No encoding path or JPEG quality was changed for users in this profiling pass.

The follow-up below tests worker capture/encoding. Server input packing remains
a separate optimization opportunity; no JPEG quality or model precision change
is needed to investigate it.

Both stable runs have no browser errors, finite bounded geometry, one request
in flight and visible interpolated mesh/skeleton playback. The screenshot also
shows the documented lag between the current video and buffered pose overlay;
this is not frame-synchronized overlay QA or 3D ground-truth validation.
The JS render-update timer excludes the Three.js renderer/GPU draw itself.

Artifacts: `generated/diagnostics/live-production-https-v1/` and `-v3/` contain
raw samples, summaries and screenshots. Their browser cgroups peaked around
0.50 GB with no memory-high, max or OOM events. A separate default-renderer
attempt (`-v2`) still fell back to SwiftShader and hit its memory-high guard
(5,442 events); its 0.46 Hz result is **not a valid unthrottled performance
measurement or a hardware-WebGL comparison**. It caused no OOM.

## Follow-up: bounded worker upload pipeline

The demo now overlaps frame preparation with the previous inference request.
One dedicated worker owns encoding, one completed-frame slot replaces older
frames, and only one HTTP request is in flight. Both capture and dispatch respect
the requested Hz cap. A latency-aware prefetch target uses recent warm request
and encoding durations; an unexpectedly slow request still refreshes the slot.
Stale frames are discarded. Offline sampling remains sequential and does not
discard samples. Resolution, JPEG quality, native model and precision are unchanged.

The final approach uses transferable `VideoFrame` objects where available,
RGBA copying in the worker for unscaled frames, and worker canvas drawing for
resized/oriented frames. Other inputs use bounded CPU pixel transfer; browsers
without worker/OffscreenCanvas support retain a private canvas fallback.
Encoding failures are reported and timeout/stop terminates the worker. No Pion,
WebRTC, video server decoder or extra networking port was needed for this pass.

The matched latency-aware run (`live-pipeline-https-v7`) measures **7.20 Hz**,
up from **4.53 Hz** (about **59% more completed estimates per second**).
Median capture-to-result latency is **221.0 ms** versus **221.6 ms** before;
p95 is **232.3 ms** versus **241.7 ms**. The median completed-frame queue wait
is 11.4 ms. This is improved throughput without simply increasing frame age.
The UI now reports a rolling 20-request rate separately from the cold-start
session rate and labels capture-to-result time as frame latency.

The final rebuilt demo repeats at **7.25 Hz**, **223.6 ms** median frame
latency and **235.3 ms** p95 (`live-pipeline-https-final`). Its extended browser
QA also verifies exact decoded JPEG pixels for the direct RGBA VideoFrame-copy
fixture. Peak browser memory is 643 MB, with zero memory-high/max/OOM events.

The initial fixed-cadence pipeline reached 7.31 Hz but increased median latency
to 248.8 ms; it is not the selected scheduling policy. Explicit main-thread
pixel readback exposed ~52 ms capture cost and only ~4.8 ms JPEG compression,
but blocked the UI and reduced throughput to 5.90 Hz; live video therefore uses
the transferable-frame path. That moves readback/conversion off the UI thread;
it **does not eliminate** its ~50–65 ms cost in this SwiftShader test. JPEG
compression remains around 5 ms. Native/server work and headless canvas/video
readback—not JPEG compression alone—remain performance costs.

Regression checks:

- Real worker/fallback JPEG decoded pixels match exactly for a deterministic
  frozen image at original and resized dimensions.
- Headless Chrome uses actual native inference for stop/restart, a 3 Hz cap,
  missing-worker fallback and a deliberately delayed outgoing request path.
- Under delayed uploads, old prepared frames are replaced: never more than
  one prepared frame or one request in flight. Geometry remains finite and
  no runtime errors occur. QA creates no persistent generations.
- JavaScript ownership tests cover replacement, cancellation, encode errors,
  transfer failure and frame-resource cleanup; Go race tests pass.
- `--qa-pipeline` on `profile_live_demo.py` reproduces the extra browser checks.
  If Python does not inherit the system HTTPS trust store, set `SSL_CERT_FILE`
  to its CA bundle; do not disable certificate verification.

All diagnostics remain ignored under `generated/diagnostics/live-pipeline-*`.
The browser profiles use bounded cgroups with no OOM; inspect their recorded
memory-high events before interpreting any timings. As with the baseline,
these are virtual-camera/headless measurements, not a guarantee of performance
on a particular user's browser or a new upstream 3D numerical parity test.

## Follow-up: low-latency presentation and camera capture

The preceding upload improvements left a major source of user-visible delay:
live playback targeted `processing latency + 1.5 * sample interval`, about
430 ms behind live time at 7.2 Hz. Sampling/camera/display delays could bring
perceived reactions near half a second. The earlier ~222 ms number measured
capture through response decode, **not the displayed reaction**.

The production demo now:

- Starts displaying each new result on the next render, blending from the
  currently displayed pose for only 25 ms. It never waits for a subsequent
  inference result. A new result interrupts an unfinished blend continuously;
  large gaps snap, and completed blends exactly copy the native target.
  Offline playback still uses timestamp interpolation.
- Uses original camera `VideoFrame`s from a feature-detected
  `MediaStreamTrackProcessor`, before HTML video rendering. Its internal queue
  is capped at one; a reader retains only the latest source frame and closes
  every replaced frame. A consumer owns one explicit clone. Stop closes the
  reader, cloned camera track and encoding worker. Browsers without this API
  on the main thread keep the video-element capture path.
- Packs JPEG/PNG pixels with typed image access and a single output allocation,
  preserving the exact original wire bytes and black-alpha compositing.
- Avoids reallocating the preview canvas every render, repainting the same
  camera frame repeatedly, or recomputing unchanged live mesh geometry after
  a blend has settled.

Matched HTTPS/SwiftShader/virtual-camera profile, five warmups and forty measured
requests, 10 Hz cap, same source/crop and production native model:

| Stage | Median | p95 |
| --- | ---: | ---: |
| Source frame age when selected | 17.2 ms | — |
| Worker camera conversion/copy | 1.8 ms | — |
| JPEG compression | 5.0 ms | — |
| Prepared-frame wait | 13.7 ms | — |
| Server input packing | 4.8 ms | — |
| Native inference | 98.7 ms | — |
| Source frame available → decoded response | 161.1 ms | — |
| Source frame available → first render submission | **176.2 ms** | **197.4 ms** |
| Source frame available → settled render submission | **201.6 ms** | **224.8 ms** |

Throughput is **8.06 Hz**; the preceding repeat gives 8.17 Hz, 172.0 ms first
render and 196.8 ms settled render. Camera conversion had cost about 60 ms when
capturing from HTMLVideoElement; direct source access removes that rendering
readback from the fast path. Input packing fell from ~18 to ~4.8 ms in the demo.
The isolated Go benchmark improves 13.4–13.5 ms to 5.14–5.16 ms and 518,406
allocations to one for a 960×540 YCbCr image.

Timing scope matters: source arrival is recorded when the browser reader receives
the frame. The latency includes its age before selection and the final Three.js
render submission. It does **not** measure camera exposure, an event occurring
between inference samples, GPU completion, compositor timing or physical scanout.
These results establish removal of the artificial playback delay, not a measured
176 ms physical motion-to-photon guarantee. A physical test needs synchronized
camera/display observation. Fallback capture can remain slower.

Verification:

- All 16,777,216 YCbCr combinations give exactly the original 16-bit-RGBA-to-byte
  result; no changed color rounding or numerical tolerance. Whole packed buffers
  match for six chroma subsampling modes, odd/negative origins, subimages/strides,
  RGBA/NRGBA/Gray, 16-bit and palette fallback images, all alpha/channel pairs
  and an actual JPEG encode/decode. Native model arithmetic is untouched.
- Deterministic step-response tests show movement starts at the next sample of
  the display clock and reaches the exact target within 25 ms independently of
  inference cadence. Interrupted blends, gaps, reset and no extrapolation pass.
- Real browser responses—not substitutes—are decoded independently and compared
  element-by-element with the settled display arrays, including signed zero.
  Worker/processor fallback, stop/restart, low Hz cap and delayed transport pass.
- Go race and JS tests pass. Browser peak memory is 616 MB, with no memory-high,
  max or OOM events. No QA generations were added to the live history.

Artifacts: `generated/diagnostics/live-latency-https-v1/` (presentation-only),
`live-latency-https-v2/`, `live-latency-https-final/`,
`live-latency-pack-tests-v1*` and `live-latency-final-tests*`.
Browser diagnostics now include `first_render_ms`, `response_to_render_ms`,
`settled_render_ms`, `source_age_ms` and the actual frame-source path. They are
bounded, like the existing timing buffers.

## Moving-video source

Use actual moving frames from Meta's related SAM 3 repository, not the earlier
animated-still smoke-test input:

- [Official SAM 3 video frames](https://github.com/facebookresearch/sam3/tree/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/assets/videos/0001),
  files `0.jpg` through `59.jpg`.
- Revision: `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`.
- Resized to 960×540, YUV420p Y4M, **chosen test playback rate 30 fps**;
  this is not a claim about the source recording's frame rate.
- Test crop: `[710,30,960,535]`, right-hand dancer, automatic box following off.
- Y4M SHA256: `8ed2f2de3d976f6e1721d3446d1e319603efa099cc2cc03a875366f7197cb4dd`.

This is a segmentation sample from SAM 3, **not SAM 3D Body ground truth**.
No upstream executable code was run to use these media frames.

## Reproducing and inspecting timings

Start the production demo as described in [the demo README](../demo/README.md).
Run a bounded browser profile on a private demo URL (supply your own Y4M path):

```sh
python scripts/run_bounded.py --memory-mib 1024 --report /tmp/live-budget.json -- \
  python scripts/profile_live_demo.py --url https://YOUR-PRIVATE-DEMO \
  --camera /path/to/camera.y4m --box 710 30 960 535 \
  --frames 40 --warmup 5 --hz 10 --output /tmp/live-profile
```

The report/output paths must be new. The guard retains 10 GiB host headroom,
disables swap and serializes heavy verification. Increase the browser budget
only with sufficient host headroom; inspect cgroup events before interpreting
timings. The deployed worker has its own 2 GiB hard cap and the Go server 1 GiB.

Frame HTTP responses now provide `Server-Timing`. In browser DevTools,
`window.sam3dQA.tracking.timings` holds the last 240 completed frame timings,
including JPEG wait, upload bytes, HTTP/read/decode time and server stages.
The native worker sends its inference duration immediately before `DONE` on
the **same pipe**, avoiding cross-pipe attribution races. Tests reject malformed,
negative and nonfinite durations and verify correct per-frame attribution.
