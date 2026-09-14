# SAM 3D Objects geometry parity

This report records the first accepted native geometry-decoder baseline. It
ports the released SLat mesh head and inference-mode FlexiCubes extraction to
C++/GGML, then compares it with the pinned upstream implementation. Mesh repair,
UV generation and appearance baking are outside this baseline.

## Authority and fixed inputs

- Meta SAM 3D Objects source: `f91db411c50efee93d8db7aeb323885650f6f722`.
- Released `slat_decoder_mesh.ckpt`; the local F16 GGUF SHA-256 is
  `cf025bbf1039fa85e1fdd1284cc707578cee836fe50a0ec4c9b40704866904ca`.
- CUDA oracle: PyTorch 2.7.1+cu128, spconv 2.3.8, TF32 disabled, NVIDIA GeForce
  RTX 5070 Ti.
- Native candidate: GGML Vulkan on the same RTX 5070 Ti, F16 weights and F32
  graph inputs/outputs.
- Small development fixture: the deterministic 3x3x3 window-boundary input
  produced by `capture_mesh_reference.py`.
- Real exemplar: upstream kids-room image and mask 14, sampler seed 42. The
  1536x1024 `S3DOBJ01` input SHA-256 is
  `b4cd3e30b681f7fb55e47076fba3c4f68e423f8f73e69563928808765bbe59b1`.
  Both decoders consume the exact native SLat tensor and coordinates, whose
  SHA-256 values are `69ff57b5da1381a97881f7c422e39a86366af81fa984dba45ed71ba88f503f35`
  and `196e74138b3f8db9ecf9e97e69a3181e2f9478d31ea86e3f978dc56c8be353c4`.
  The compact CUDA capture manifest SHA-256 is
  `b033f220e4a5d42641e3e26e5c151283ca2d866dd88127c1b3707af2ea333b7b`.

The CUDA capture executes the original transformer, spconv layers, sparse
aggregation, NVIDIA FlexiCubes tables and private extraction helpers. The native
test does not inject upstream intermediates into its end-to-end candidate.

## Numerical and structural result

The native neural stages pass the frozen real-exemplar gates:

| Boundary | Maximum absolute error | Relative L2 | Sign agreement |
| --- | ---: | ---: | ---: |
| Transformer block 11 | 0.828947 | 0.00222134 | 0.999177 |
| Upsample 0 | 0.375109 | 0.00295865 | 0.999001 |
| Upsample 1 | 0.127745 | 0.00531099 | 0.997709 |
| Raw 101-channel decoder output | 0.219571 | 0.00548420 | 0.989734 |
| Aggregated vertex attributes | 0.0406201 | 0.00327857 | 0.999404 |

When the C++ extractor receives the exact upstream raw tensor, aggregate
coordinates, surface cubes, cases, surface edges and all 977,164 triangle
indices match exactly. Vertex and learned-attribute maximum errors are
`1.19e-7` and `2.38e-7`. This isolates the ported extraction algorithm from
neural arithmetic differences.

End to end, 780 of 2,010,196 aggregated SDF values change sign. Every changed
decision is close to the zero isosurface: the largest upstream and native
margins are 0.000312 and 0.000397. Keying dual vertices by source cube avoids
mistaking later index shifts for geometry error. It matches 487,747 dual
vertices; 1,604 keys are unmatched across both outputs. Matched vertices have
relative L2 `0.000185` and maximum error `0.003877`, less than one cell at the
256 extraction resolution. Matched learned attributes have relative L2
`0.000665`. The canonical triangle-set symmetric difference is 33,896; this
includes locally changed surface cases and near-tied quad diagonal choices.

The upstream mesh has 488,534 vertices and 977,164 faces. Native has 488,564
vertices and 977,232 faces, differences of 0.0061% and 0.0070%. The acceptance
gate reports exact discrete differences as diagnostics, then bounds SDF margins,
source-cube structure, matched geometry/attributes and canonical topology. This
policy accounts for accumulated F16/F32 backend arithmetic without hiding an
axis, layout, table or extraction error. It is calibrated on the small fixture
and this fixed real exemplar; a broader object suite remains useful future work.

## Visual result

Both raw vertex-colour GLBs were rendered by the checked-in Three.js comparison
page with the same parser, camera, scale, lighting, material and framing. There
is no independent alignment.

| View | RGB MAE | Pixels with RGB delta > 24 | Silhouette IoU |
| --- | ---: | ---: | ---: |
| Front | 0.00011738 | 0.01488% | 0.9999106 |
| Side | 0.00007192 | 0.00420% | 0.9999681 |
| Oblique | 0.00011440 | 0.01373% | 0.9999377 |

![Pinned upstream CUDA and native Vulkan geometry comparison](OBJECTS_MESH_PARITY.png)

The native GLB is 23,453,288 bytes with SHA-256
`427522d1a74b5f84840b021b5e8e7d8d55b54325e90c9463ab44a2f64c37f2c5`.
The upstream-oracle GLB written by the same exporter is 23,451,752 bytes with
SHA-256 `4f741b43730757de0779b112f04970fbac8bfd807edaba16d7330c8e182b484b`.

## Post-parity host optimization

The first optimization pass preserves the accepted arithmetic and output while
removing avoidable ordered-tree work. Sparse subdivision now indexes parents
with a bounded dense table when practical and a hash table otherwise. It looks
up each parent's 27 neighbours once, then derives the rows of all eight children
from precomputed bit transitions. FlexiCubes aggregation and edge counting use
hash tables followed by compact sorted key lists where upstream ordering is
observable. Deformed positions and sigmoid colours are computed once per grid
vertex, and production extraction skips the capture-only aggregate sort.

On the fixed real parity workload, the complete Vulkan parity executable fell
from 23.59 to 15.90 seconds wall time (1.48x, 32.6% less). The four Vulkan graphs
remained approximately 1.7 seconds, confirming that the reduction is in host
table construction and extraction. Both CPU and Vulkan still pass every gate,
and a full optimized image-to-GLB run produced the exact native GLB hash above.
The level-two graph still reserves 4,675.36 MiB of Vulkan scratch memory; reducing
that allocation requires a separately gated staged or fused sparse convolution.

## Reproduction

Generate the small oracle or pass production SAMT tensors:

```sh
python scripts/objects/capture_mesh_reference.py \
  --upstream reference/upstream/sam-3d-objects \
  --checkpoint generated/models/sam-3d-objects/checkpoints/slat_decoder_mesh.ckpt \
  --features /path/e2e_slat_feats.samt \
  --coords /path/e2e_slat_coords.samt \
  --output /path/upstream-mesh --extract-mesh --compact
```

Run the native gate with no more than eight CPU threads:

```sh
build/vulkan-optimized/bin/sam3d-objects-mesh-decoder-parity \
  build/vulkan-optimized/bin/libggml-vulkan.so Vulkan 0 \
  generated/models/objects-gguf/slat_decoder_mesh-f16.gguf \
  /path/upstream-mesh 8
```

Serve `mesh_visual_compare.html`, the vendored Three.js files and the two GLBs
from one local origin to reproduce the three fixed render metrics.
