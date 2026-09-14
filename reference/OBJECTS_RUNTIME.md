# SAM 3D Objects native runtime evidence

This report records the initial numerical and end-to-end evidence for the
experimental native SAM 3D Objects runtime. It does not mark the official
Gaussian-render visual gate complete. The separate native geometry decoder has
now reached numerical, structural, topology and rendered-image parity; see the
[geometry parity report](OBJECTS_MESH_PARITY.md).

## Revisions and exemplar

- Numerical authority: Meta `facebookresearch/sam-3d-objects` at
  `f91db411c50efee93d8db7aeb323885650f6f722`.
- Credited GGML implementation basis: Asher-1 `sam-3d-objects-ggml` at
  `1c14b7c3c3e8d9109b943ddc83a0a39c73744246`.
- Depth/point-map model: Microsoft MoGe `moge-vitl`.
- Scene: upstream
  `notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png`, SHA-256
  `72785412200dcb9f5c6bc4eaa3018172293f67836627708e6bce78150109af15`.
- Object mask: upstream mask `14.png`, SHA-256
  `7a79c2f537a20bfd70c001df42dd8d916d4c5f7fce4ea7ee99738aa178b29c13`.
- Sampler seed: 42. Native and upstream arithmetic comparisons replay the exact
  stored initial noise rather than relying on seed-equivalent RNGs.

The native demo input was bounded to 1536 by 1024 pixels and encoded as the
`S3DOBJ01` tight RGBA container. Its SHA-256 was
`fa638ef2316e4a6de7d5195263be4c179d4b5f142a5b3ab8e37d9d9c5d91d84a`.
Alpha is the exact binary selection mask.

## Vulkan configuration

The run used F16 GGUF weights with F32 graph inputs, outputs and accumulation on
the Vulkan backend. Strict precision settings disabled F16 storage and arithmetic
shortcuts. The `vulkan-optimized` preset applies the reviewed GGML build-copy
patch series by default. Patch 0013 fixes incomplete K32/K64 scalar convolution
tiles; without it, MoGe output channels after the first four were unwritten on
the tested NVIDIA Vulkan device.

The model-free Vulkan regression covers complete K32 and K64 convolution output
tiles. It passes in the patched build. A later legacy F32 matmul case in the full
precision executable requires the separate BF16 cooperative-matrix handshake,
so that unrelated case is not evidence for or against the Objects convolution
fix.

## Numerical boundary results

The corrected native MoGe point map, compared element-for-element with the
pinned PyTorch result, measured:

| Output | MAE | Relative L2 | Correlation |
| --- | ---: | ---: | ---: |
| Point map | 0.0001057347 | 0.0001157610 | 0.99999999698 |
| Validity mask | 0.0000108 | 0.0000128 | — |

PointPatch preprocessing now replaces non-finite XYZ values with zero and emits
the corresponding exact invalid-token mask. The previous all-valid upload let
NaNs enter the condition sequence and invalidated every later comparison.

The pinned PyTorch SS generator then consumed the exact condition tensor and
initial states dumped by Vulkan. At the first matched solver step:

| Quantity | Maximum absolute error | Relative L2 | Correlation |
| --- | ---: | ---: | ---: |
| Conditional shape velocity | 0.00075221062 | 0.00012513957 | 0.99999999 |
| Unconditional shape velocity | 0.00051139295 | 0.00010642343 | — |
| Integrated shape state | 0.000084325671 | 0.0000087861286 | — |

The other modality velocities have relative L2 error from approximately
`1e-4` to `3.4e-4`; their integrated states are approximately `2e-6` to `5e-6`.
These differences are consistent with accumulated floating-point evaluation
differences at this boundary and do not indicate changed support or semantics.
The replay command is documented in [`scripts/objects/README.md`](../scripts/objects/README.md).

## Complete native run

The corrected 25-step SS trajectory decoded 27,485 occupied cells out of
262,144. Surface pruning retained 26,960 cells. The 25-step SLat trajectory and
Gaussian decoder completed and wrote 862,720 Gaussian records to a 58,665,396
byte binary PLY. The artifact SHA-256 for this run was
`bb0c720361084324ee2416800642ef8bdf46ef882b2b7a6a30c840db63b6b980`.
A clean rebuild with patch fingerprint
`5ba9b9c5d6eab0186b1ebe45719013b81994ef48ff6b31cbeaf1d78a2369b527`
produced the same file byte for byte and reported 22.4 seconds of native staged
compute on the tested NVIDIA GeForce RTX 5070 Ti.

The demo accepts a scene image and painted mask, runs that native path through
the single bounded worker, saves both canonical inputs and hashes, restores jobs
from history, and exposes the full PLY. Its Three.js display samples Gaussian
centers and degree-zero color for interaction. That view is useful workflow QA,
but it is not a Gaussian splat renderer.

A real headless Chromium reload restored an Object history entry, its scene and
mask, parsed its binary PLY, rendered the point preview with two draw calls and
exposed the PLY download. Browser diagnostics reported no page exceptions. This
workflow test used a 10,000-record bounded fixture; the full 862,720-record file
was validated by the native exporter, byte extent and deterministic hash above.

## Open acceptance work

Full visual parity still requires the official sparse SLat/decoder path to replay
the same native support, noise and conditions and export an upstream PLY. Both
PLYs must then be rendered with the same official Gaussian renderer, camera and
background. CUDA reference execution is now available through the host's NVIDIA
container device, and it was used to complete the raw-geometry comparison. The
official Gaussian-render comparison has not yet been produced. The numerical
results above and the plausible nonempty native PLY do not replace that remaining
Gaussian visual gate.
