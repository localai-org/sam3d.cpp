# Raw RGB to Body geometry composition

This test composes native image preparation, the full 32-block DINOv3 backbone,
conditioning, six decoder layers and their own MHR/camera/keypoint feedback.
It uses the official Body `notebook/images/dancing.jpg` photograph, **synthetic
SAM neural parameters and real released MHR geometry**. It is not trained pose
estimation, visual reconstruction acceptance, hand refinement or demo acceptance.

The image SHA-256 is
`0112b0a32ea5860db6a6fc700804751528341a02bf07fed0329a0c2610f905d3`.
The explicit box is `[600, 80, 1330, 1250]` in the 1920x1280 image; it is manually
selected, not an upstream detector prediction. Intrinsics use the original
default focal-length formula and image center. These choices are saved in the
capture manifest rather than hidden in the native inference code.

## Independent original reference

`capture_body_flow.py --image-pipeline --full-width` uses
`body_image_pipeline.py` to execute original `prepare_batch`, transforms,
`BaseModel.data_preprocess`, the DINOv3 factory and Body backbone wrapper, and
the unchanged `SAM3DBody.forward_pose_branch`. The existing decoder observers
retain the independently captured, uninstrumented per-layer outputs. The
instrumented and uninstrumented final outputs must agree exactly.

Two reference-only adaptations are explicit:

- Backbone pre/post hooks load and release each original block's parameters
  to bound memory. They do not replace neural operations. Parameters are reused
  from a hash-verified original synthetic backbone capture. Its old input image
  is skipped and never used in this run.
- Original `get_ray_condition` hardcodes CUDA placement. The wrapper passes its
  metadata on CUDA and moves the resulting rays to the selected neural device.
  Therefore even the CPU-neural reference requires CUDA for ray construction.
  Native CPU inference does not require CUDA. This reference is **not a pure-CPU
  or optimized performance baseline**.

Run the reviewed isolated container procedure from [MHR.md](MHR.md), exposing
the GPU for either reference device and using these entry arguments:

```sh
/work/reference/capture_body_flow.py \
  --upstream /work/reference/upstream/sam-3d-body \
  --model /work/generated/models/mhr-public/assets/mhr_model.pt \
  --roma-wheel /work/generated/wheels/roma-1.6.1-py3-none-any.whl \
  --output /output --device cpu --full-width --image-pipeline \
  --dino-upstream /work/reference/upstream/dinov3 \
  --backbone-reference /work/generated/fixtures/dino-backbone-full-cuda
```

Use distinct output directories for `--device cpu` and `--device cuda`.
Both captures retain 320 intermediate tensors and 115 independent full-output
checks. Sources, image, model state, scripts and artifacts are hash-identified.

## Native composition, not intermediate replay

The `S3DRGB01` diagnostic input contains raw RGB bytes, a box, intrinsics,
prompt/optional previous-estimate inputs and model parameters. It contains no
saved normalized image, rays, backbone features, decoder tokens, camera
predictions or meshes. The native runner calls `body_from_rgb` and uses only
native-produced intermediates. Its second input supplies backbone parameters;
the standalone fixture's original image is explicitly skipped.

```sh
uv run --project reference/python --frozen python scripts/run_body_flow.py \
  --reference generated/fixtures/body-image-flow-cpu \
  --runner build/debug/bin/sam3d-body-flow-capture \
  --module build/debug/bin/libggml-cpu.so \
  --gguf generated/models/mhr-public/mhr-lod1-f32.gguf \
  --backbone-input generated/fixtures/dino-backbone-full-cuda/case.0000.input \
  --output generated/fixtures/body-image-flow-native-cpu --threads 12
```

For Vulkan, select its runner/module and the CUDA reference and add
`--backend Vulkan` plus the desired device index/description. Device/driver paths
are caller configuration. The runner enforces the existing strict F32 settings.
CPU retains ASan/UBSan/LSan; the diagnosed NVIDIA driver-initialization exception
uses the separate Vulkan UBSan build.

The comparison requires the exact expected sets of all diagnostic and full
outputs, validates input/artifact hashes, and applies the existing per-unit
limits and reference-calibrated Euler policy from [BODY_FLOW.md](BODY_FLOW.md).
Passing synthetic composition does not establish learned Body parity or remove
the official checkpoint-access prerequisite. Objects and the actual browser
workflow remain separate unfinished deliverables.

## Results

CPU versus the original CPU-neural reference and NVIDIA Vulkan versus original
CUDA each pass **435/435 checks**: 320 diagnostic boundaries plus 115 independent
uninstrumented outputs. Observed/unobserved original outputs are byte-identical
on both devices. No acceptance limit or inference math was changed for this run.

| Worst absolute error across layers | CPU | NVIDIA Vulkan |
| --- | ---: | ---: |
| Preparation (all nine fields) | 0 | 0 |
| Backbone boundaries | 1.51e-5 | 2.58e-5 |
| Normalized decoder tokens | 1.81e-5 | 1.32e-5 |
| Mesh vertices, meters | 7.16e-7 | 5.97e-7 |
| 3D keypoints, meters | 5.97e-7 | 3.58e-7 |
| Projected mesh vertices, pixels | 6.11e-4 | 6.72e-4 |

| Tensor artifact | SHA-256 |
| --- | --- |
| Original CPU-neural taps | `1cdbe1f0361f7bc055f618d28d521751f5c84a7318329b6472be5a4608a18c3d` |
| Original CUDA taps | `cbac6e6842b9c7508f4fbdd3cbf5ce87b83a0a55c35eca94dd2b6a30f8d25c2c` |
| Original CPU-neural full outputs | `be17bf73ce69a2f44113c6592ae3c9d0d0462db648254376e0bb198ebd2d9535` |
| Original CUDA full outputs | `a0a63ade6e78123031174c9f08621ec62c2e3cdd2e8b549a46e24c968b0586f0` |
| Native CPU | `5e86028e47eabb79fdc5d88d05c6fe449051010907e8e147128f995a97005cf0` |
| Native Vulkan | `5a758faa0daff03a2c72ea4da8d77caf667aed9be86756dc9b296dab9100dc40` |

Reports are retained under `generated/fixtures/body-image-flow-native-{cpu,vulkan}`.
Normal regression remains 25 native ASan/UBSan/LSan tests and 35 Python tests,
all passing without model downloads. This large image-composition test is a
separate local acceptance run requiring the verified parameter and MHR assets.
