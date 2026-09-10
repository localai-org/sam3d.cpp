# Objects composed pointmap preprocessing

This milestone ports the **complete original `preprocess_image` method for an
explicit supported transform configuration**, with raw RGBA and supplied CHW XYZ
as inputs. It is CPU preprocessing, not MoGe inference, the published checkpoint's
verified transform selection, learned conditioning, or image-to-object parity.

## Contract and original reference

`src/objects_preprocess.cpp` composes native operations in the original order:

1. Convert RGBA with NumPy-compatible F64 division followed by F32, retaining
   soft alpha. Compute object pointmap normalization and moments.
2. Align XYZ to RGB, preserving the original NaN-mask resize behavior; crop,
   pad and remove background jointly. RGB/mask padding and XYZ padding differ.
3. Square-pad and resize cropped RGB/mask/XYZ, then full-image RGB/mask.
4. Normalize the original XYZ again using the **resized full-image mask** and
   the first normalizer's own moments, subject to each normalizer's override
   rules. With normalization disabled, moments are still computed, overrides
   are not forwarded, and both pointmap paths retain unnormalized XYZ.
5. Transform the full normalized and unnormalized pointmaps and return all 11
   original fields. No captured intermediate is supplied to native inference.

The supported explicit configuration uses `resize_all_to_same_size`,
`crop_around_mask_with_padding`, `rembg`, centered square padding, bicubic-AA RGB
resize and nearest mask/XYZ resize. Full-image joint transforms are identity.
Image and pointmap sides, crop factor/padding, XYZ square-padding value, both
normalizers and normalization enabled/disabled are explicit options. These are
not asserted to be the inaccessible model configuration's defaults.

`capture_objects_preprocess.py` runs hash-verified, unchanged original
`InferencePipelinePointMap.preprocess_image` and `PreProcessor` methods, together
with the original SSI/PyTorch3D and image transforms already validated separately.
It extracts original AST definitions to avoid unrelated model imports, without
rewriting numerical expressions. The additional pipeline source SHA-256 is
`55b69917ff0bb8b5ca6e918f516d75e9e7561a6c9396120abde863bf5757aa72`.
All source identities are recorded in `tests/fixtures/objects-preprocess.json`.

Eight small cases cover all eight normalizer types across the object/full paths,
override enabled/disabled, normalization disabled, soft masks, partial-coordinate
NaNs, equal/different XYZ dimensions and zero/NaN square padding. A ninth case
uses the unresized official **4096x2160 kid_box image and mask**, with explicitly
synthetic 257x259 XYZ; it is not an estimated pointmap. Three uninstrumented
original runs and the observed run must produce bit-identical final tensors.
Original outputs and native returned buffers are captured independently from
intermediate diagnostic tensors. Large tensors stream to individual files to
bound capture/comparison memory and remain outside Git.

## Reproduction and acceptance

Use the reviewed isolated reference container described in the other Objects
reference documents, with the repository read-only, output directory writable,
no network, no privileges and a 5 GiB memory limit:

```sh
python reference/capture_objects_preprocess.py \
  --upstream reference/upstream/sam-3d-objects \
  --pytorch3d reference/upstream/pytorch3d-selected \
  --output generated/fixtures/objects-preprocess-cpu

cmake --build --preset debug
python scripts/run_objects_preprocess.py \
  --reference generated/fixtures/objects-preprocess-cpu \
  --output generated/fixtures/objects-preprocess-native-cpu \
  --runner build/debug/bin/sam3d-objects-preprocess-capture
ctest --preset debug --output-on-failure
python -m unittest discover -s tests -p 'test_*.py'
```

All **536 checks pass**: 437 original boundaries and 99 independently returned
fields. These preserve existing component limits: RGB maximum absolute error
and relative L2 <=1e-6, pointmap/normalizer maximum absolute error <=1e-4 and
relative L2 <=2e-5, exact masks/bounds, exact nonfinite categories for permitted
XYZ tensors, and finite normalization parameters. No tolerance was relaxed.
Worst absolute error is 8.345e-7 and relative L2 3.642e-7, both in bicubic RGB
resize. This is a CPU-to-CPU reference claim, not native Vulkan preprocessing.

The normal sanitizer-enabled CTest contains 388 small original boundaries and
checks all 11 returned buffers, observer/output-only equality and input
rejections. Python tests check fixture/source identity, case coverage, corrupted
mask/pointmap comparisons and nonfinite parameter rejection. The complete suite
passes 31 native tests and 55 Python tests without model downloads.

`sam3d-objects-preprocess-fuzz` passed 100,000 ASan/UBSan/LSan cases, seed 61203,
28 seconds, peak RSS 450 MB, without findings. It exercises the internal composed
API, buffer/options validation and returned-field invariants; it excludes GGUF
loading. Its binary SHA-256 is
`d79feec11e6ae491a84671b41f2d119418eec13cab798634ad09c3f1cbd1c893`.
The small fixture SHA-256 is
`345842d6f6e32664bd9be0d7c31279df5d0938121985b3d0b4b487bdb54e4940`.
The final native report SHA-256 is
`f4e0651152120b9b25487f4f25c9458ab29296b13b1898a4b29b2023ca787f56`;
it records each emitted file's hash, the runner and current comparator identity.

Remaining composition work includes consuming these native fields in PointPatch
and image conditioners, verifying trained configuration/state, MoGe, and the
subsequent generation/geometry stages. None of the web-demo or performance gates
is satisfied by these preprocessing tests.
