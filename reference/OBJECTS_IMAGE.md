# Objects RGBA image/mask preparation

`src/objects_image.cpp` ports the original default Objects image/mask path:
RGBA pixels -> crop and masked object view -> resized object/full-image tensors.
The opaque C API in `sam3d.h` exposes this stage without Python, image decoding,
model loading or public struct layouts. It runs on CPU, including before future
Vulkan inference, matching upstream preprocessing before its device transfer.
**This is not learned Objects reconstruction or completion of O1.** Pointmap
processing as a complete image-to-condition pipeline, MoGe and checkpoint-specific
configurations remain unfinished. [SSI normalization](OBJECTS_SSI.md) and the
[point-window transformer](OBJECTS_POINTPATCH.md) now have separate component
tests. Default-source behavior does not prove which preprocessing
options the still-gated published model configuration selects.

## Original behavior retained

- Convert bytes using NumPy's original F64 division by 255 then F32 conversion.
  Alpha greater than zero becomes a binary object mask; it is not soft opacity.
- Bounds use maximum minus minimum indices, **without adding one**. Either
  axis spanning fewer than three pixel centers is rejected. The native library
  also returns a checked error for empty masks rather than upstream's later
  zero-size resize exception.
- Expand a square about the box center with Python's truncation-toward-zero and
  integer-half conventions, preserving off-image zero padding. Then square-pad
  and add `int(side * padding_factor)` black pixels on each edge.
- Mask the cropped RGB view, but keep the separate full RGB view unmasked.
- Center-pad the full image. Resize RGB with float antialiased bicubic (Keys
  coefficient -0.5, separable horizontal/vertical passes) and masks with nearest
  neighbor. Do not clamp bicubic overshoot or add ImageNet normalization here.
  The official example's cropped RGB minimum is approximately -0.08168; clamping
  it would silently change the condition tensor.

Default output side is 518, box factor 1.0 and padding 0.1. Explicit supported
options are documented in the C header and invalid setters preserve prior values.
All input/intermediate allocations are bounded; input dimensions are <=4096 and
intermediate images <=16M pixels. Large diagnostic tap retention is not intended
as a memory/performance-optimized model implementation.

## Independent original capture

`reference/capture_objects_image.py` extracts unchanged, hash-verified original
AST definitions for `InferencePipeline.preprocess_image`, its helper methods,
`get_default_preprocessor`, `PreProcessor` and relevant image/mask transforms.
This avoids unrelated model/rendering imports, not their numerical behavior.
Source-frame observers capture outputs without replacing operations. Each case
has three identical uninstrumented runs and an exactly equal observed run.

Five small cases cover up/down/equal-size resizing, odd dimensions, border
padding, fractional box factors, disconnected support and nonbinary alpha.
The sixth uses the original `notebook/images/kid_box/image.png` and `0.png` mask,
with default options and 518x518 outputs. Original image/mask files remain outside
Git. Their SHA-256 identities are:

- Image: `0d4d55dcebda52e7be9fd60e6fe68ce3fe52fdee5f7fe086bffda7520466e9ed`.
- Mask: `811eada8e42b6870a403bf63cbf0ba2ec06bf585c993833cc6b4a13585ec77b8`.

Use the reviewed offline/read-only reference container described in
[MHR.md](MHR.md), mounting a dedicated output directory. This capture needs no
GPU or model weights. Its Python entry arguments are:

```sh
/work/reference/capture_objects_image.py \
  --upstream /work/reference/upstream/sam-3d-objects --output /output
```

The pinned original Objects revision is
`f91db411c50efee93d8db7aeb323885650f6f722`. Manifests record every executed source
file's hash, the script and environment versions, raw input artifacts, rules,
observed outputs and independent full outputs. `S3DOIM01` native test inputs
contain only raw RGBA pixels/stride and explicit options, not computed crops.

```sh
uv run --project reference/python --frozen python scripts/run_objects_image.py \
  --reference generated/fixtures/objects-image-cpu \
  --runner build/debug/bin/sam3d-objects-image-capture \
  --output generated/fixtures/objects-image-native-cpu
```

## Results and regression

All **114 comparisons pass**: 90 original boundaries and 24 independent final
outputs. Of the boundary checks, 78 require exact equality. RGB resize limits
were frozen before native implementation at max-absolute and relative-L2 1e-6;
observed worst errors are 8.345e-7 and 7.056e-7 respectively. No tolerance was
changed to obtain a pass. These are CPU preprocessing results, not claimed
Vulkan graph or learned-model parity.

| Artifact | SHA-256 |
| --- | --- |
| Original boundaries | `b98de5250f807dd28889e5597f969f6f659aef8f52109cc160abdfc0d7a34f7a` |
| Original full outputs | `10c1a333fe973cde0ffe7762b0b659de9adf5e0f4fd5df28b1a65a163493df7a` |
| Native boundaries | `e19abd5795d2b1a2b0f8d178394bb05c15b1ef070d055fc50e362fcd4951dc12` |
| Small text fixture | `f59846fe4df9ebdf5685cccd15be7bf2628c3855907856da6bd6d71067f3c663` |

`tests/fixtures/objects-image.txt` includes all 75 boundaries of the five small
cases, not just selected pixels. Normal CTest also compares the public C API's
four fields with the verified native outputs and tests invalid options, strides,
shapes, masks, error buffers and lifetime independence. The pure-C ownership test
also passes when compiled against a separately installed header/library.
All 27 native ASan/UBSan/LSan tests and 37 Python tests pass. The Python suite
checks the original small fixture and capture-source identities as well.

`sam3d-objects-image-fuzz` completed 100,000 cases with seed 60124, ASan/UBSan/LSan
enabled, 444 MB peak RSS and no findings. It exercises both successful and
rejected C API paths; GGUF loading is excluded. Reproduce with:

```sh
cmake --preset fuzz
cmake --build --preset fuzz
build/fuzz/bin/sam3d-objects-image-fuzz \
  -runs=100000 -seed=60124 -max_len=128 -rss_limit_mb=2048
```

An initial direct launch hit the guard after reporting 3101 MB before the first
input. A genuinely forked child started at 33 MB and passed under the same limit,
consistent with the earlier inherited-launcher high-water issue documented in
the [reference guide](README.md). Sanitizers and memory limits were not weakened.
