---
license: other
license_name: sam-license
license_link: LICENSE
base_model: facebook/sam-3d-body-dinov3
library_name: gguf
pipeline_tag: image-to-3d
inference: false
tags:
  - gguf
  - ggml
  - sam-3d-body
  - human-pose-estimation
  - 3d-human-mesh-recovery
---

# SAM 3D Body DINOv3 — GGUF

F32 GGUF conversions of Meta's
[facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
for **sam3d.cpp**, a C++23/GGML CPU/Vulkan implementation. Given an image and
a selected person's bounding box, the supported Body pose branch estimates a
3D body mesh, joints, pose parameters and camera translation.

This is a format conversion and component selection, **not a fine-tune or
low-bit quantization**. GGUF names a container format: these custom SAM3D
architectures require sam3d.cpp, not llama.cpp or a generic LLM runner.
There is no Python, PyTorch or CUDA inference dependency in the native runtime.

## Bundle

Download all three files together. The runtime validates companion identities.

| File | Content | Bytes |
| --- | --- | ---: |
| `body-dinov3-f32.gguf` | Complete DINOv3 H+ image backbone, F32 | 3,363,058,592 |
| `body-pose-branch-f32.gguf` | Body decoder/pose heads and topology, F32/I32 | 231,353,792 |
| `mhr-lod1-f32.gguf` | MHR LOD1 geometry, skinning and correctives, F32/I32 | 693,111,264 |

The BF16 runtime mode converts selected weights/activations at load/execution;
it uses these same files. There are no Q4/Q8 variants in this release.
`MANIFEST.json` and `SHA256SUMS` identify the exact artifacts and conversion
source commit. The model source, metadata and license files are included in the
checksum manifest; no training checkpoints or reference photographs are bundled.

## Download and run

Use the Hugging Face CLI after this repository has been published:

```sh
hf download LocalAI-io/sam-3d-body-dinov3-GGUF --local-dir models/sam3d-body
(cd models/sam3d-body && sha256sum -c SHA256SUMS)
```

Build sam3d.cpp with CMake and an installed C++23 compiler; add a Vulkan SDK for
GPU inference. Its optional Go web demo accepts these paths:

```sh
./build/sam3d-demo \
  --listen 127.0.0.1:8097 --data generated/demo \
  --runner build/debug/bin/sam3d-body-infer \
  --backend CPU --module build/debug/bin/libggml-cpu.so \
  --backbone models/sam3d-body/body-dinov3-f32.gguf \
  --branch models/sam3d-body/body-pose-branch-f32.gguf \
  --mhr models/sam3d-body/mhr-lod1-f32.gguf
```

The demo uses the same opaque C API available to library consumers. CPU debug
builds prioritize sanitizer checks; use an optimized build for performance.
See sam3d.cpp's README, API guide and demo README for build, Vulkan/BF16 and
resource-limit configuration. It neither downloads models nor guesses a person
detector. Remote webcam access requires HTTPS and a trusted deployment.

## Supported scope and limitations

- One selected person per request, with an explicit box and camera intrinsics
  (the demo offers an approximate focal-length default).
- Body pose-branch inference and MHR mesh decoding are implemented. **Detailed
  hand-crop refinement is not implemented.** These files do not reproduce the
  entire upstream hand-refined estimator.
- Video uses independent frame estimates, not a learned temporal tracker.
  Monocular depth, camera calibration, occlusions and small/partial people can
  produce ambiguity or jitter. Outputs are estimates, not measurements or
  medical/biometric identification results.
- Native F32 Vulkan has passed 646 recorded trained comparisons for the tested
  Body branch. Six strict F32 CPU geometry/projection checks remain open; BF16
  has separate numerical acceptance policies and known hand-logit differences.
  This is not universal accuracy or full-pipeline parity across all inputs.
- SAM 3D Objects, automatic detection, textures and animation-ready rig export
  are not supplied by this bundle. Demo mesh exports are static posed geometry.

## Provenance and changes

- Original weights: `facebook/sam-3d-body-dinov3`, revision
  `11aaa346c7204874a1cbafe3d39a979080b2c55a`.
- Original Body code: `facebookresearch/sam-3d-body`, revision
  `b5c765a0d89d789985e186d396315e7590887b94`.
- DINOv3 code: `facebookresearch/dinov3`, revision
  `6876159a11b4df116f30f667f8c9888617df0751`.
- Geometry: independently public
  [MHR v1.0.1](https://github.com/facebookresearch/MHR/releases/tag/v1.0.1),
  matching the upstream Body companion asset.
- Legacy checkpoints were extracted only in an isolated trusted reference
  environment. Normal converters accept verified safetensors. Floating values
  are stored as F32, bounded topology indices as I32. Names/layout descriptors
  are converted for GGML; the unused DINO mask token is synthesized as zero,
  as required by the unmasked evaluation path. No retraining is performed.
- The Body archive selects the pose branch, omitting hand-refinement weights;
  the MHR archive omits unused solver state. Source/checkpoint/config hashes
  are recorded in `MANIFEST.json` and embedded component metadata.

## License and responsible use

**The SAM/DINO weights are not Apache licensed.** SAM material and its
derivatives remain under the [SAM License](LICENSE). DINOv3 material additionally
retains the [DINOv3 License](LICENSES/DINOv3.md). The MHR companion and topology
retain [MHR Apache-2.0](LICENSES/MHR-Apache-2.0.txt), with applicable
[Momentum MIT](LICENSES/Momentum.txt) notices. See [NOTICE](NOTICE).
The Apache grant for original sam3d.cpp code/documentation does not relicense
the model or third-party adaptations.

SAM and DINO permit modification/distribution subject to their complete terms,
including same-license redistribution, supplied license copies and use/legal/
trade-control restrictions. Do not use this conversion to evade upstream terms.
Only process images you are authorized to use, obtain appropriate consent, and
protect photographs and inferred body data. No endorsement by Meta is implied.

The original [model card](https://huggingface.co/facebook/sam-3d-body-dinov3)
is authoritative for training data, intended use and research claims. Please
acknowledge and cite **SAM 3D Body: Robust Full-Body Human Mesh Recovery** and
MHR using the citations supplied by their original authors. These conversions
make no independent training-data or benchmark claims.
