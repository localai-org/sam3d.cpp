# Objects pointmap-aware joint transforms

`objects_prepare_pointmap_joint` in `src/objects_image.cpp` implements an explicit
original joint-transform sequence:

```text
raw RGBA + supplied XYZ
    -> preserve soft alpha
    -> align XYZ resolution to RGB
    -> crop RGB/mask/XYZ together
    -> square/extended padding
    -> remove background from RGB and XYZ
```

This is native CPU preprocessing with no Python inference dependency. It is
also intended for the Vulkan pipeline's preprocessing. It does **not** include
SSI normalization, individual final image/pointmap transforms, MoGe inference,
or learned conditioning. [SSI](OBJECTS_SSI.md) and
[PointPatchEmbed](OBJECTS_POINTPATCH.md) have separate tests. The interface is
currently internal C++; the full model/session C API remains unfinished.
Alignment currently uses the original default target (RGB dimensions); the
optional explicit target-size branch that also resizes RGB is not implemented.

## Upstream details that affect alignment

- Alpha remains its original F32 value in [0,1], not a binary mask. Cropping and
  background removal use nonzero support; RGB is multiplied by soft alpha.
  This differs from the earlier default RGBA entry point, which binarizes alpha.
- If XYZ has different dimensions from RGB, original `resize_all_to_same_size`
  clears NaNs **per coordinate**, resizes the cleaned XYZ with antialiased
  bilinear interpolation, resizes the original per-pixel any-NaN mask with nearest
  interpolation, and restores all-three-coordinate NaNs where that mask is true.
  Infinity is not silently cleared. An equal-sized pointmap bypasses this logic,
  retaining any original partial-coordinate NaNs.
- The mask bbox uses max-minus-min, without adding one, and original Python
  integer truncation/half-size conventions. Empty or degenerate masks are rejected.
- Torchvision crop pads off-image regions with zeros, **including XYZ**. Subsequent
  square/extension padding uses zero for RGB/mask but NaN for XYZ. Background
  removal then makes masked-out XYZ NaN. Making all padding NaN prematurely
  would differ at the crop boundary even if the final masked result looked right.

RGB input dimensions are bounded at 4096; supplied pointmap dimensions at 2048.
Crop/intermediate images are bounded to 16M pixels. Buffer extents/stride, box
factor and padding are checked. The official image initially revealed a too-small
2048 RGB limit; support was extended to 4096 rather than shrinking the test image.

## Reference capture

`reference/capture_objects_joint.py` uses hash-verified original Meta methods at
Objects revision `f91db411c50efee93d8db7aeb323885650f6f722`. It reuses the existing
audited image-reference loader and additionally extracts the unchanged alignment
function. Original `PreProcessor._preprocess_image_mask_pointmap` runs the sequence
`resize_all_to_same_size -> crop_around_mask_with_padding -> rembg`. A distinct
legacy dual-transform list is also configured, so triple-transform priority is
exercised. The sequence and options are explicit test inputs, **not an assertion
that the still-gated checkpoint configuration selects them**.

Three uninstrumented executions are bit-identical; the observed execution is
also bit-identical. Source-frame observers capture actual NaN masks, cleaned and
resized XYZ, aligned RGB/mask/XYZ, bbox, padded crops and final outputs. Native
input contains only raw RGBA with stride, supplied XYZ and crop options. No
computed reference intermediate is injected into native execution.

The seven cases include six small resize/soft-mask/padding/NaN/Inf edge cases
and the official `kid_box` image/mask at **4096x2160**, paired with an explicitly
synthetic 257x259 XYZ map. The photograph is real upstream data; the XYZ is **not
an upstream-estimated pointmap or a reconstruction-quality result**. Original
image/mask hashes and every source/input/output hash are in the capture manifest.

Run inside the reviewed offline/read-only container described in [MHR.md](MHR.md):

```sh
/work/reference/capture_objects_joint.py \
  --upstream /work/reference/upstream/sam-3d-objects --output /output
```

The reference is CPU F32 PyTorch 2.7.0/torchvision 0.22, one thread, deterministic
algorithms. This stage is not a CUDA/Vulkan neural-graph comparison.

```sh
uv run --frozen python scripts/run_objects_joint.py \
  --reference generated/fixtures/objects-joint-cpu \
  --runner build/debug/bin/sam3d-objects-joint-capture \
  --output generated/fixtures/objects-joint-native-cpu
```

The comparison reads original tensors individually to avoid retaining a second
full-resolution capture while native ASan inference runs. Generated large
captures stay outside Git; no image or model downloads occur in normal tests.

## Results and regression

All **129 checks pass**: 108 original intermediate boundaries and 21 independent
final RGB/mask/XYZ outputs. RGB, alpha/masks and bbox values require exact equality.
Pointmap comparisons require matching NaN/+Inf/-Inf categories and both maximum
absolute error <=1e-5 and relative L2 <=2e-5 on finite elements. The limits were
declared before native testing and were not relaxed. Worst errors are 9.537e-7
absolute and 4.560e-8 relative L2, at the full-resolution aligned pointmap.

Artifact SHA-256s:

- Native report: `3a33d4390ac4346eabe494190cbc7de248221a492330f715d1870bf5e580429d`.
- Native tensors: `f9bc2ecdc96f9ca035ef672594f2c168dfe922b203bf5ba57167bdbf6637e395`.
- Small text fixture: `b58ab38c14c500371687f77200deb449b80d6ed645bf9921eeeeb1ca9e1fdc79`.
- Returned result buffers: `4851dfd1f71d2804265095383e1ef709f0f95ae017abde6fad4209c89c168b7d`.

The final comparison captures the function's actual returned RGB, mask and XYZ
buffers separately from diagnostic taps; normal tests also require those buffers
to agree with the observed final stage. A correct diagnostic copy cannot conceal
an incorrect returned result.

Normal CTest compares all 92 boundaries of the six small cases, checks final
buffers and observation/output-only equality, and rejects invalid dimensions,
extents, strides, factors and degenerate masks. Python tests verify source/fixture
identity and reject altered soft alpha or nonfinite classifications. The complete
30-test native sanitizer suite and 52 Python tests pass. All 114 original default
RGBA comparisons were repeated after changing the shared image helpers and pass.
Their native tensors remain byte-identical to the previous implementation:
`e19abd5795d2b1a2b0f8d178394bb05c15b1ef070d055fc50e362fcd4951dc12`.

The new `sam3d-objects-joint-fuzz` exercises this internal API; the existing
`sam3d-objects-image-fuzz` covers the public image C API. Neither includes GGUF
loading. Both completed 100,000 ASan/UBSan/LSan cases with seed 61103 and no
findings. Joint fuzzing took 312 seconds, peak RSS 425 MB; the repeated public
image API run took 112 seconds, peak RSS 431 MB. Binary identities:

- Joint fuzzer: `c811bd89cf839adbc9c8b7f05af11b7538ce2192ff7be8083cdfd9376e2a4b4a`.
- Image API fuzzer: `20d9079032e2e909d22defefa6d0689757474694fcf98069d75703fc1419524f`.
- Fuzz shared library: `dba4d9bb6b6913a0236faf5401bd55aa4cb164f70ab55e3f9ac27b4429af963f`.

This component is not complete image-to-condition or learned-model parity,
browser acceptance, or performance parity.
