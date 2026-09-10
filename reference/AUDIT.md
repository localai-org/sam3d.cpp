# Reference and reusable-code audit

Date: 2026-09-08. Scope: selected source, build scripts, interfaces, upstream
integration code and public repository/file metadata. This is not a complete
security audit and no numerical/performance results have been independently
reproduced. No third-party engine, installer, build system or model was run.
No checkpoints were downloaded. Only a small YAML config was downloaded and
hashed locally; weight hashes below are Hugging Face's published LFS metadata.

The implementation decisions and future gates are in [DESIGN.md](../DESIGN.md).

## Official references

| Repository | Pinned source revision | Role |
| --- | --- | --- |
| [facebookresearch/sam-3d-body][body] | `b5c765a0d89d789985e186d396315e7590887b94` | Authoritative Body model and preprocessing/decoding |
| [facebookresearch/sam-3d-objects][objects] | `f91db411c50efee93d8db7aeb323885650f6f722` | Authoritative Objects pipeline, flow models and decoders |
| [NVlabs/GEM-X][gem] | `32992550dba114c62243fb55e361311972dce8f9` | Actual downstream Body/ViTPose feature contract and SOMA video estimation |

GEM's git tree pins `third_party/sam-3d-body` to the Body revision above,
`third_party/soma` to `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`, and
`third_party/soma-retargeter` to `b12d9a3eeff6ea64d7029684e47d1e92b9a60c2c`.
The latter two are dependency identities, not completed source audits.

Official model repositories and audited metadata revisions:

| Repository | HF revision | Access reported by public metadata |
| --- | --- | --- |
| [facebook/sam-3d-body-dinov3][hf-body] | `11aaa346c7204874a1cbafe3d39a979080b2c55a` | Manually gated |
| [facebook/sam-3d-objects][hf-objects] | `2e73555018d2741ccd486e56c24fac41155a1dc6` | Manually gated |
| [nvidia/GEM-X][hf-gem] | `5ccf5ca3746c3620aa4016114f069a5f6ae399cd` | Not gated; that is not a waiver of model/asset licenses |

[sources.json](sources.json) records the principal file sizes and SHA-256s.
The Body checkpoint (2,109,129,346 bytes) and MHR asset (696,110,248 bytes)
have identical published hashes in Meta's and NVIDIA's repositories.
The locally hashed GEM `model_config.yaml` specifies `dinov3_vith16plus`,
512×512, decoder dimension 1024/depth 6 and BF16 training/mixed-precision options.
Checkpoint tensor dtype/count and actual inference precision remain unverified.

Objects has multiple checkpoints: SS generator alone is 6,690,136,964 bytes,
SLat generator 4,906,537,684 bytes, plus decoders, conditioners and MoGe assets.
These serialized sizes are not an inference parameter count or RAM/VRAM budget.
Do not download every sibling: metadata includes encoders not necessarily used
by inference, a zero-byte safetensors entry and alternate decoder formats.
The exact `pipeline.yaml` and its complete dependency closure still need to be
resolved after access is established.

### Integration traps established from source

- [Body's loader][body-loader] uses `torch.load(..., weights_only=False)`,
  loads state dictionaries non-strictly and offers an unpinned HF snapshot
  convenience path. Isolate legacy loading; validate actual missing/unexpected
  keys and pin downloads instead of adopting these defaults.
- [GEM's SAM extractor][gem-sam] sets `DO_HAND_DETECT_TOKENS=False`, wraps
  the decoder to recover its pose token, takes its first token and returns
  `pred_cam_t`. If the token is missing, it can pad/truncate MHR parameters
  into a 1024-value replacement. That is not the same learned feature. Reference
  capture must prove the real token path ran; our validated path must fail
  clearly rather than silently use the fallback.
- [GEM's ViTPose extractor][gem-vitpose] constructs DINOv3 with width 1280,
  depth 32, 20 heads, SwiGLU and a 77-channel deconvolution heatmap head.
  It fetches DINO code through `torch.hub` and uses flipped-image inference
  with a SOMA77 permutation. It is not an interchangeable standard ViTPose
  checkpoint or the local TRELLIS ViT-L/16 backbone.
- MHR model parameters/keypoints are not SOMA rotations. Preserve camera and
  coordinate contracts at each boundary. The [upstream webcam bridge][sonic]
  explicitly lists unforwarded wrist/finger tracking as a limitation; do not
  interpret that as SAM/GEM being incapable of estimating arms or wrists.

## Official MHR geometry reference

Subsequent geometry work obtained MHR from Meta's own independently public
[v1.0.1 release](https://github.com/facebookresearch/MHR/releases/tag/v1.0.1).
Its model bytes match the MHR identity above. This does not authorize the denied
SAM neural checkpoints. The exact released TorchScript is now the standalone
geometry oracle; see [MHR.md](MHR.md) for hashes, isolation and source/asset
differences. No prior-art C++ engine was executed to obtain this reference.

## Candidate: AmmarkoV/SAM3DBody-cpp

Audited revision: `198002ede17892648d27d5c0a0085860a9e63bcd`.
Read README, CMake, selected engine implementation, public C header and license.

Useful source leads are image preprocessing, MHR parameter conversion/native
body decoding and skeleton/BVH export. They require their own comparison with
Meta, particularly rotation order, scale and pose/shape correctives. The root
license declares MIT for this project; copied upstream/asset terms still need
file-level accounting.

Observed reasons not to build/adopt the engine wholesale:

- [The implementation][candidate-body-engine] instantiates ONNX Runtime
  sessions. GGML use does not make the full inference stack GGML-only.
- [CMake][candidate-body-cmake] fetches GGML at `master`, not an immutable
  revision. It can download prebuilt ONNX Runtime when not installed; that
  `file(DOWNLOAD)` call has no `EXPECTED_HASH` argument. This is a reproducibility
  and dependency-verification gap, not evidence that the downloaded file is bad.
- Model downloads at configure time are **off by default**, but CMake explicitly
  documents first-run binary model fetching. Do not run the quickstart or binary
  just to inspect it.
- [Its C API][candidate-body-api] exposes `FsbConfig` and `FsbResult` layouts,
  unlike the opaque-handle/setter/getter interface needed here.

Decision: no execution needed to establish the mismatch. Keep selected decoding
code as prior art; re-audit any files actually proposed for copying.

## Candidate: Asher-1/sam-3d-objects-ggml

Audited revision: `1c14b7c3c3e8d9109b943ddc83a0a39c73744246`.
Read runtime README, CMake, public interface, GGUF loader, converter excerpts,
patch helper and stage-capture excerpts. Enumerated graph sources without
claiming a line-by-line audit of every graph or shader.

This is substantial useful prior art, not just a converter: the repository
contains DINOv2/PointPatch, SS/SLat flow, sparse operations, Gaussian decoding
and a native MoGe graph. Its [README][candidate-objects-readme] explicitly
distinguishes native conditioned generation from its hybrid raw-image workflow.

Observed limits and review items:

- Raw-image orchestration still uses Python MoGe preprocessing. A separate
  native MoGe neural graph does not prove native preprocessing equivalence.
- Its native GLB exporter consumes **already decoded** mesh tensors. It does
  not implement native mesh decoding, repair, UV mapping or texture baking.
- Despite a C API comment, [the public header][candidate-objects-api] uses
  a C++ namespace, `std::string` and `std::vector`. Its input is a captured
  condition directory, not a usable flat image/mask C ABI.
- [CMake][candidate-objects-cmake] automatically runs a
  [patch helper][candidate-objects-patches] that modifies the GGML submodule
  worktree. It is pinned and checks patch state, but we need the local
  build-copy approach instead. Review needed patches individually, including
  custom operations/backend support; do not transplant the combined patch blind.
- [The GGUF loader][candidate-objects-loader] casts 64-bit integer metadata
  to 32 bits and accepts defaults for missing/wrong-typed keys. Its string-array
  accessor checks array kind but not element kind before the GGUF string
  getter. It allocates device weights before an application-level architecture
  schema/budget check. Do not reuse these helpers unchanged; this inspection
  did not establish an exploit or audit GGUF internals.
- [The converter][candidate-objects-converter] directly loads the MoGe
  checkpoint with `weights_only=True`. That is better than unrestricted
  pickle, but does not meet our safe-intermediate-only normal converter contract.
- [Stage capture][candidate-objects-fixtures] invokes upstream modules but
  patches loading/configuration and manually orchestrates stages/noise. Useful
  diagnostics, not independently sufficient evidence that outputs match the
  untouched public pipeline. Validate instrumentation/staging first.
- The README describes a failing release signal until its complete required
  quality/latency matrix passes. Do not reuse its performance claims or render
  thresholds as our F32 tensor-parity evidence.

Decision: worthwhile source-level reuse candidate for the Objects phase, not
a ready replacement for this project's reference process or runtime contract.
No need to execute it now. Before code adoption, inspect the selected graph,
patch and license boundaries and test it against our own official fixtures.

## What remains unknown

No official forward pass has been performed in this task. Device support,
container/package compatibility, weight tensor inventories, peak memory,
oracle repeatability and CPU/Vulkan errors are all **unmeasured**. No candidate
has been declared numerically correct, fast, memory-safe or release-ready.

Remaining pins include Objects' resolved auxiliary checkpoint/config closure
and the actual DINO source used by GEM's dynamic feature loader. Licensing and
access checks must cover model code, weights, geometry and copied code separately.
No weights or HF repository should be published merely because another port
already distributes a conversion.

[body]: https://github.com/facebookresearch/sam-3d-body/tree/b5c765a0d89d789985e186d396315e7590887b94
[objects]: https://github.com/facebookresearch/sam-3d-objects/tree/f91db411c50efee93d8db7aeb323885650f6f722
[gem]: https://github.com/NVlabs/GEM-X/tree/32992550dba114c62243fb55e361311972dce8f9
[hf-body]: https://huggingface.co/facebook/sam-3d-body-dinov3/tree/11aaa346c7204874a1cbafe3d39a979080b2c55a
[hf-objects]: https://huggingface.co/facebook/sam-3d-objects/tree/2e73555018d2741ccd486e56c24fac41155a1dc6
[hf-gem]: https://huggingface.co/nvidia/GEM-X/tree/5ccf5ca3746c3620aa4016114f069a5f6ae399cd
[body-loader]: https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/build_models.py
[gem-sam]: https://github.com/NVlabs/GEM-X/blob/32992550dba114c62243fb55e361311972dce8f9/gem/utils/sam3db_extractor.py
[gem-vitpose]: https://github.com/NVlabs/GEM-X/blob/32992550dba114c62243fb55e361311972dce8f9/gem/utils/vitpose_extractor.py
[sonic]: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/live_camera_teleop.html#limitations
[candidate-body-engine]: https://github.com/AmmarkoV/SAM3DBody-cpp/blob/198002ede17892648d27d5c0a0085860a9e63bcd/src/fast_sam_3dbody.cpp
[candidate-body-cmake]: https://github.com/AmmarkoV/SAM3DBody-cpp/blob/198002ede17892648d27d5c0a0085860a9e63bcd/CMakeLists.txt
[candidate-body-api]: https://github.com/AmmarkoV/SAM3DBody-cpp/blob/198002ede17892648d27d5c0a0085860a9e63bcd/src/fast_sam_3dbody_capi.h
[candidate-objects-readme]: https://github.com/Asher-1/sam-3d-objects-ggml/blob/1c14b7c3c3e8d9109b943ddc83a0a39c73744246/cpp_ggml/README.md
[candidate-objects-api]: https://github.com/Asher-1/sam-3d-objects-ggml/blob/1c14b7c3c3e8d9109b943ddc83a0a39c73744246/cpp_ggml/include/sam3dggml.h
[candidate-objects-cmake]: https://github.com/Asher-1/sam-3d-objects-ggml/blob/1c14b7c3c3e8d9109b943ddc83a0a39c73744246/cpp_ggml/CMakeLists.txt
[candidate-objects-patches]: https://github.com/Asher-1/sam-3d-objects-ggml/blob/1c14b7c3c3e8d9109b943ddc83a0a39c73744246/cpp_ggml/scripts/apply_ggml_patches.sh
[candidate-objects-loader]: https://github.com/Asher-1/sam-3d-objects-ggml/blob/1c14b7c3c3e8d9109b943ddc83a0a39c73744246/cpp_ggml/src/gguf_loader.cpp
[candidate-objects-converter]: https://github.com/Asher-1/sam-3d-objects-ggml/blob/1c14b7c3c3e8d9109b943ddc83a0a39c73744246/cpp_ggml/scripts/convert_sam3d_to_gguf.py
[candidate-objects-fixtures]: https://github.com/Asher-1/sam-3d-objects-ggml/blob/1c14b7c3c3e8d9109b943ddc83a0a39c73744246/cpp_ggml/scripts/dump_e2e_stages.py
