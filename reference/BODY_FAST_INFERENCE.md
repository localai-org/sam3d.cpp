# Optional Fast-SAM-inspired native inference

Implemented in the native Body pose branch using the existing GGUF weights.
The default stays at 512 pixels, every intermediate prediction, with MHR
correctives. Approximate modes are available through the C API, native CLI and
`--body-inference` demo startup setting. Final mesh, joints, keypoints, camera and
skeleton transforms retain the existing public result contract.

## What changed

- Select intermediate pose/MHR/camera predictions after decoder layers 0–4.
  All six transformer layers and the final prediction always execute. The Fast-SAM
  schedule selects 0,1,2. Subsequent layers still update feedback using the latest
  prediction; before the first selected prediction there is no pose feedback.
  Cached predictions are local to one inference call.
- Optionally disable MHR pose correctives, including the final mesh. Correctives
  affect feedback keypoints as well as vertices, so disabling them can change
  the eventual skeleton even though they do not directly alter MHR joints.
- Support 448- and 384-pixel crops throughout preprocessing, rays, image encoding
  and decoder conditioning. The same weights are used without retraining.
- Omit unused intermediate output copies and rotation matrices. Mesh skinning and
  keypoint mapping still run because feedback needs surface-derived landmarks.
  Final output and diagnostic boundary captures remain complete.

The implementation follows the scheduling semantics in
[Fast-SAM-3D-Body at 808b53c](https://github.com/yangtiming/Fast-SAM-3D-Body/tree/808b53c7d9c26a7e511d31144f1e5efb058e15c9).
Credit to Timing Yang, Sicheng He, Hongyi Jing, Jiawei Yang, Zhijian Liu,
Chuhang Zou and Yue Wang for this research. See [NOTICE](../NOTICE) and their
[MIT license](../LICENSES/Fast-SAM-3D-Body-MIT.txt).
It covers changes relevant to our existing body-only pipeline. It does not add
that project's hand-refinement pipeline, GPU detector/cropping, PyTorch compile,
TensorRT or SMPL export.

## Native measurements, 2026-09-11

Compared with the preserved runner and shared library from commit
`9befb479e0872ed706dbe5efe3f9399622da0c63`, using identical models, input files,
backend modules, precision flags and thread counts. Hardware: NVIDIA RTX 5070 Ti
and AMD Ryzen 9 7900. Production Vulkan build, no sanitizers, one host inference
thread, 6 GiB process memory cap. Compilation was restricted to eight CPUs and
`-j8`.

Two RGB/box/intrinsics inputs were alternated in each persistent worker. BF16 used
four warm-up requests and twenty timed requests (ten per image). Timings cover
native image preprocessing through the final body result; they exclude startup,
model loading, file serialization, web requests and exports. Each variant ran
sequentially, without randomized ordering, so small differences may be noise.

Distances below compare predictions directly with our original implementation;
**they are not errors against ground truth**. Joint distances include the 126
anatomical MHR joints and exclude the artificial `body_world` joint. Coordinates
are in metres, converted to millimetres, with no pelvis alignment or camera
translation added. Mean/max are shown separately for each input.

| BF16 variant | Median ms | Speedup | Image 0 joints mean/max mm | Image 1 joints mean/max mm |
| --- | ---: | ---: | ---: | ---: |
| Original implementation | 85.46 | 1.00× | 0 / 0 | 0 / 0 |
| New default | 85.35 | 1.00× | 0 / 0 | 0 / 0 |
| Slim intermediate outputs only | 85.03 | 1.00× | 0 / 0 | 0 / 0 |
| Intermediate predictions 0,1,2 only | 80.24 | 1.06× | 13.04 / 34.10 | 37.40 / 196.12 |
| Correctives off only | 80.33 | 1.06× | 1.86 / 6.56 | 2.60 / 8.76 |
| Crop 448 only | 73.21 | 1.17× | 10.53 / 30.48 | 69.65 / 258.74 |
| Crop 384 only | 64.87 | 1.32× | 24.88 / 50.04 | 60.08 / 183.40 |
| Combined `fast512` | 76.64 | 1.12× | 13.35 / 33.74 | 38.31 / 197.42 |
| Combined `fast448` | 64.40 | 1.33× | 17.59 / 43.38 | 75.36 / 269.78 |
| Combined `fast384` | 54.90 | 1.56× | 26.31 / 47.47 | 66.02 / 200.96 |
| Final prediction only | 68.03 | 1.26× | 60.08 / 127.83 | 68.23 / 289.69 |

Combined presets select predictions 0,1,2, disable correctives and use slim
intermediate outputs. These differences warrant broader evaluation before any
approximate preset becomes a default. Slim is exact on these inputs but has no
convincing standalone speedup.

F32 Vulkan used two warm-ups and eight timed requests (four per image), with
cooperative matrices disabled as required by the existing F32 acceptance path:

| F32 variant | Median ms | Speedup | Image 0 joints mean/max mm | Image 1 joints mean/max mm |
| --- | ---: | ---: | ---: | ---: |
| Original implementation | 325.52 | 1.00× | 0 / 0 | 0 / 0 |
| New default | 327.21 | 0.99× | 0 / 0 | 0 / 0 |
| Slim only | 326.08 | 1.00× | 0 / 0 | 0 / 0 |
| Predictions 0,1,2 | 320.92 | 1.01× | 12.98 / 33.50 | 36.94 / 193.41 |
| Correctives off | 322.06 | 1.01× | 1.86 / 6.56 | 2.11 / 5.39 |
| Combined `fast448` | 237.41 | 1.37× | 17.50 / 43.09 | 72.53 / 253.62 |

The default and slim variants match the original **byte for byte across all 20
public output fields** in both precisions. All tested variants repeated exactly
across alternating requests. Topology, result shapes/types and finiteness passed.
The [machine-readable evidence](fast-body-native-v1.json) also includes vertices,
keypoints, pixel projections, camera-relative and pelvis-aligned joints, joint
rotation angles, per-field differences, raw timings and artifact hashes.

A CPU F32 smoke comparison used eight inference threads, one warm-up and only
two timed requests per variant. Original/default/`fast448` medians were
7394.98/7570.10/6125.99 ms. The default remained byte-identical to the original;
`fast448` joint differences match the F32 Vulkan figures above. This short run
checks CPU functionality, not a stable CPU performance estimate.

Seven affected native regression tests passed with ASan/UBSan/LSan, including
the C API, result ownership, geometry mapping, decoder norm, preprocessing,
feedback and backbone. Slim/full keypoint mapping is checked for both body and
hand fixtures. The Go race suite and three comparison-metric regressions passed.
Production and debug builds completed with at most eight compiler jobs. The final
rebuild adds CLI help and finite-output validation; a separate smoke run checks
its BF16 outputs against the binaries used for the main timing sweep.

## What the paper published

[Fast SAM 3D Body, arXiv:2603.15603v1](https://arxiv.org/html/2603.15603v1)
reports benchmark accuracy against ground truth, rather than pairwise differences
between its predictions and the original predictions. Its ablation tables report:

- Table 3: selecting body intermediate layers 0,1,2 changes MPJPE from
  58.89 to 58.96 mm and PVE from 69.21 to 69.28 mm.
- Table 4: reducing 512 to 448 pixels changes MPJPE from 58.96 to 58.95 mm,
  and PVE from 69.28 to 69.50 mm; 384 pixels gives 59.01 and 70.22 mm.
- Table 6: disabling correctives changes MPJPE from 58.04 to 58.96 mm,
  and PVE from 69.26 to 69.28 mm.

These are dataset averages with their complete pipeline and ablation settings.
They do not imply that individual joints move by only those small differences,
that all ablations add independently, or that our two-image differences should
match them. Their hardware and full pipeline timings also differ from ours.
We have not measured parity against their implementation or ground-truth accuracy.

## Reproduction

Use the production build and the environment flags documented in
[BODY_LIVE_PERFORMANCE.md](BODY_LIVE_PERFORMANCE.md) / [API.md](../docs/API.md).
Keep precision and environment identical within each baseline/candidate pair.
Before modifying/rebuilding, preserve the old executable and `libsam3d.so`; the
`--library` argument overrides the runtime library search path. A preserved
executable alone may otherwise load the newly built library through its RUNPATH.

```sh
cmake --build build/vulkan-bf16-performance --target sam3d-body-infer -j8

# Run each invocation under the same external memory cap.
python3 scripts/benchmark_fast_body.py \
  --runner build/fast-body-baseline/sam3d-body-infer \
  --library build/fast-body-baseline/libsam3d.so \
  --module build/vulkan-bf16-performance/bin/libggml-vulkan.so \
  --backbone generated/models/sam-3d-body-dinov3/body-dinov3-f32.gguf \
  --branch generated/models/sam-3d-body-dinov3/body-pose-branch-f32.gguf \
  --mhr generated/models/mhr-public/mhr-lod1-f32.gguf \
  --input generated/demo-reference/image.input \
  --input generated/benchmarks/body-yoga-bf16-eager/image.input \
  --precision bf16 --variant baseline --output generated/diagnostics/original
```

Repeat with the new runner/library, a fresh output directory and selected
`--variant` arguments (`baseline`, `slim`, `schedule`, `no-correctives`, `crop448`,
`crop384`, `fast512`, `fast448`, `fast384`, `final-only`). Output directories must
not exist. Inputs use the CLI format documented in the demo README; their hashes
in the evidence identify the exact test inputs. Then, in the reference Python
environment with NumPy and safetensors:

```sh
python3 scripts/compare_body_variants.py \
  --baseline generated/diagnostics/original \
  --candidate generated/diagnostics/candidate \
  --output generated/diagnostics/candidate/divergence.json \
  --require-exact baseline --require-exact slim
python3 scripts/test_compare_body_variants.py
```

The benchmark records nearest-rank p95 and median wall-clock inference times.
The comparison rejects different input/model hashes or runtime configurations,
changed topology, incompatible shapes/types and nonfinite output. The exact gates
apply only to variants expected to preserve predictions; approximate modes record
all divergences without an invented accuracy acceptance threshold.
