# Frame-wise Body video tracking

2026-09-10. Implemented in the optional Go/browser demo, using the existing
public-C-API resident native worker. There is no new model, temporal network,
person detector or tracker dependency. No model arithmetic or numerical
acceptance policy changed for video. Initial QA used a UBSan Vulkan worker;
the live demo now uses the separately validated production BF16 build.
CPU native regression uses ASan/UBSan/LSan and Vulkan correctness retains UBSan.
See [production live profiling](BODY_LIVE_PERFORMANCE.md) for real moving-video
input, HTTPS QA and stage-level measurements.

The live upload path now uses a bounded worker/prefetch pipeline: one active
encode, one replaceable prepared frame and one inference request. The measured
headless rate now reaches about 8.1 Hz. Live presentation no longer uses the
delayed source timeline: each result starts a 25 ms blend immediately. Direct
camera-track capture and exact typed image packing bring median first render
submission to 176 ms and settled render to 202 ms (excluding physical camera/
display latency). `scripts/profile_live_demo.py --qa-pipeline` additionally checks
worker/camera-processor fallbacks, cancellation, restart, slow-transport replacement
and exact settled-display agreement with the real native response.
Native arithmetic and the saved frame format are unchanged.

## What was tested

- Go race tests cover valid/invalid session creation, format and frame-size
  admission, rate limiting, no-video-backlog behavior, timestamp ordering,
  persistence/reload, first-only topology, live no-history behavior,
  cancellation/reaping and exact complete-native-output → saved-frame mapping.
- JavaScript tests cover binary schema, nonfinite/extreme values, topology,
  interpolation, clamping/no extrapolation and bounded next-frame crop changes.
- Real Chrome 151 uploads and decodes video, performs real NVIDIA Vulkan
  inference, renders meshes/skeletons, plays/scrubs saved sequences, reloads
  history, cancels and restarts webcam inference, and checks finite geometry.
  Mobile layout screenshots are included. No browser/runtime errors remain.
- Chromium's virtual webcam delivers ordinary video pixels through
  `getUserMedia`; inference responses are **not mocked**. This does not test a
  physical webcam, real permission interaction, network webcam drivers or an
  HTTPS deployment. QA's explicit insecure-origin override is test-only.
- A second real photo job consumes the exact first video JPEG and box/camera.
  Its final vertices, joints, camera translation and triangle indices match
  the saved video data byte-for-byte. Display interpolation never changes those
  stored estimates.

Artifacts (local ignored data):

- `generated/diagnostics/video-go-v4*`: complete Go race tests/build.
- `generated/diagnostics/video-browser-v1/`: constant official dancer image
  encoded as a short video; offline + virtual-webcam workflow passes.
- `generated/diagnostics/video-browser-v3/`: the official dancer image with
  small image-plane rotation over time; offline + virtual-webcam workflow and
  exact final-output comparison pass. `report.json` and desktop/mobile
  screenshots record the actual run.
- `tests/test_tracking_math.mjs`: small dependency-free Node regression test.
- `generated/diagnostics/video-photo-regression-v3/`: original photo viewer,
  upstream overlay, orbit controls, history/reload, desktop/mobile scrolling,
  official-example selection and exact GLB export/reload pass, reusing an
  accepted real native result. The new video/photo comparison above separately
  runs fresh inference. The preceding browser-only v2 run hit its 1 GiB soft
  memory watermark during reload (no OOM); v3 used 1.5 GiB with the unchanged
  10 GiB host reserve. No inference settings were relaxed.
- `generated/diagnostics/bf16-image-gather-cpu-test-final*`: all 49 normal native
  tests pass under ASan/UBSan/LSan after the preceding optimization cleanup.

The preceding `video-browser-v2` comparison failed because Python parsed JSON
`-0` as an integer and lost its IEEE sign. Direct native binary inspection
confirmed exact equality in all three geometry fields. The checker now preserves
signed zero; no tolerance or inference output was changed to pass it.

## Rate, scope and limitations

The first live browser test used a 5 Hz cap and delivered about 4.4 Hz. The final
changing-frame test used a 10 Hz cap and delivered about **4.8 Hz** (eight-frame
observation, roughly 4.74 Hz including its ninth frame before stopping). These
are short functional QA runs with UBSan and software-rendered headless Chrome,
not representative sustained camera benchmarks or unsanitized native timings.
The UI displays actual request latency and achieved rate. One request is in
flight; a slow model cannot cause an unbounded queued-video delay.

Both clips derive from a known still image, so they test transport, changing
pixels, model reuse, playback and numerical safety—not accuracy on real moving
people, occlusion, fast gestures, multiple identities or temporal bone-length
stability. That quality evaluation remains to be done with representative
real-motion clips. Independent estimates can jitter in depth/shape. Linear
mesh/joint display interpolation is not a constrained skeletal animation and
does not create additional model estimates. Long live gaps reset interpolation
history; normal buffering uses a monotonic time cursor and no extrapolation.

Raw inference uses native body coordinates, with camera translation separate.
The viewer uses the same fixed axis change as the photo demo. It does not claim
world-space motion recovery. Offline history stores only the first source
thumbnail; complete source video is not retained by the server.

## Reproduce

Build/run as described in [the demo guide](../demo/README.md), then:

```sh
(cd demo && go test -race ./...)
node tests/test_tracking_math.mjs
python scripts/qa_body_video.py --url http://localhost:8097 \
  --video /path/to/short-person.webm --camera /path/to/same-video.y4m \
  --live-hz 10 --output generated/diagnostics/video-qa --chrome chromium
```

Use `scripts/run_bounded.py` for the browser test on memory-constrained hosts,
with the native worker separately bounded. The existing complete single-image
Chrome QA remains applicable. See [demo details](../demo/README.md) for the
binary sequence format, limits, camera HTTPS requirement and API routes.
