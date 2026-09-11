# Body pose head, real MHR and output mapping

The native composition is now:

```text
pose token + optional initial estimate
  -> original-contract FFN / pose, hand and scale decoding
  -> real converted MHR: skeleton, blendshapes, correctives, skinning
  -> meters / joint rotations / 308-point mapping / 70-point selection / axes
```

`src/body_output.cpp` connects the existing pose prefix and real MHR GGUF runtime;
it does not accept injected MHR results in the composed entry point. This is
**composition/contract parity with synthetic SAM head state**, not trained SAM
image-to-body parity. The MHR asset is real and byte-verified. The FFN, scale/hand
PCA, hand index buffers and 308-point mapping are deliberately synthetic and
explicit in the tests, because the trained SAM checkpoint remains gated. Do not
ship them as model defaults, semantic keypoint definitions or inference weights.

## Exact source semantics

The pinned original `MHRHead` constructor and complete `forward`/`mhr_forward`
methods run in the reviewed container. `MOMENTUM_ENABLED=0` deliberately selects
the verified released TorchScript through the original loader; the newer Momentum
Python model is not substituted. No original mathematical functions are rewritten
for the full-head oracle. Original source and RoMa wheel hashes are checked before
import; only the exact official MHR byte identity is deserialized.

- Geometry and skeleton come from the native pose parameters, not from a fixture.
- Vertices and joint coordinates are divided by 100 before keypoint regression.
  Native CPU preserves direct division; the CUDA-reference Vulkan mode preserves
  F32 reciprocal multiplication. The arithmetic choice is explicit internally.
- The mapping input is `[vertices, joint coordinates]`, then permute/flatten from
  `[B,N,3]` to `[N,B*3]`. The full 308 rows are projected before selecting 70.
  GGML performs the projection; its output layout is checked explicitly.
- RoMa `unitquat_to_rotmat` uses XYZW and **does not normalize**. Its diagonal
  formula uses all four squares, not the unit-only `1 - 2*(...)` shortcut.
- Only vertex/keypoint/joint **positions** get the Y/Z sign flips in the final
  body outputs. `joint_global_rots` remains in MHR's original rotation convention;
  applying a seemingly consistent camera-axis change there would differ upstream.
- `do_pcblend` is accepted by the original head but never forwarded to `self.mhr`.
  This body path therefore keeps MHR's default `apply_correctives=True`; both
  default and `do_pcblend=False` original outputs are compared explicitly.
- The wrist-centric hand model, hand refinement and its extra transforms are
  separate unfinished paths. `slim_keypoints` is also unused by this original
  body-mode forward; do not invent different output selection semantics.

## Capture and compare

Use the isolated container procedure in [MHR.md](MHR.md), an output directory such
as `generated/fixtures/body-output-cpu`, and these Python entry-point arguments:

```sh
/work/reference/capture_body_output.py \
  --model /work/generated/models/mhr-public/assets/mhr_model.pt \
  --upstream /work/reference/upstream/sam-3d-body \
  --roma-wheel /work/generated/wheels/roma-1.6.1-py3-none-any.whl \
  --output /output --device cpu
```

Capture CUDA separately with the documented GPU exposure and `--device cuda`.
The two cases exercise batch two/no initial estimate and batch one/1024-wide
two-layer FFN with an initial estimate. Head weights are small perturbations of
the original zero-pose initialization, and the nontrivial synthetic mapping has
signed weights using both vertices and joints. Every required state array is
passed explicitly to native inference.

Non-replacing operation observation and source-frame locals capture 23 boundaries
per case. Observation is checked against an uninstrumented complete head call:
CPU is exact; CUDA worst difference is 1.49e-8 on these inputs. The original final
tensor outputs are retained independently and compared natively too.

```sh
uv run --project reference/python --frozen python scripts/run_body_output.py \
  --reference generated/fixtures/body-output-cpu \
  --runner build/debug/bin/sam3d-body-output-capture \
  --module build/debug/bin/libggml-cpu.so \
  --gguf generated/models/mhr-public/mhr-lod1-f32.gguf \
  --output generated/fixtures/body-output-native-cpu --threads 12
```

For NVIDIA Vulkan use that build's runner and `libggml-vulkan.so`, the CUDA
reference, `--backend Vulkan`, and your device index/description. Driver paths are
caller configuration, not hardcoded. The runner sets the same strict-F32 Vulkan
options as the preceding component captures. CPU retains ASan/UBSan/LSan; the
documented NVIDIA ASan/ICD exception uses the separate Vulkan UBSan build.

Both backends pass 70 comparisons (46 boundaries plus 24 uninstrumented final
tensor comparisons) at max-abs 1e-4 / relative-L2 2e-5. Native CPU final vertices
are within 4.77e-7 **meters** and keypoints within 2.39e-7 m. Vulkan/CUDA final
vertices are within 5.97e-7 m and keypoints within 3.58e-7 m. Worst intermediate
max-abs, including raw MHR centimeters, is 4.58e-5 on CPU and 5.35e-5 on Vulkan.
These are correctness gates, not performance measurements.

The normal sanitizer regression `body_output_mapping_upstream` covers 12 mapping
boundaries plus malformed shapes/nonfinite values, axis signs and non-normalizing
quaternion behavior. Its 268KB fixture compresses the first case to 16 vertices:
only identically zero mapping columns are omitted, with all joints retained.
The expected outputs still come from the original complete head. This fixture is
explicitly an **isolated mapping test**, not the full-head test with real GGUF.
It needs no model download. Adjacent JSON records full-capture provenance.

| Tensor artifact | SHA-256 |
| --- | --- |
| Original CPU observed head | `48c9948fb6ea97b68e5bc034b6c1397684dd9a3affe005f6d87015c105d7bfee` |
| Original CUDA observed head | `a958c882e77d66a9a1c1c3a605e4654cfa673197a72dda4162c4adc4e6f7ca09` |
| Native CPU | `2d2d8e2156c3a1ea21a6ad334b2f4e86edffe201522e929b80415c251c54dfde` |
| Native NVIDIA Vulkan | `b693270880c6b354380ff285325ab22271c50769f7b8cf42fb36ab0194b086c8` |
| Normal mapping fixture | `773ceef7f6eaf60c5c97b9bc0f71001f23d84482fc00ac390908515d9dfc9cdc` |

The full per-layer decoder geometry/camera feedback composition is now separately
tested with synthetic SAM state; see [BODY_FLOW.md](BODY_FLOW.md).
Remaining acceptance includes trained checkpoint buffers/semantic keypoints,
full-width decoder and backbone/decoder composition,
hand refinement, the public model/session API, image-to-mesh demo/Chrome QA,
Objects and optimized end-to-end performance. This stage does not satisfy those.
