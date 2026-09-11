# Objects pointmap scale/shift normalization

`src/objects_ssi.cpp` implements all eight normalization variants exposed by the
four original SSI classes, plus their own denormalization. It consumes supplied
F32 XYZ pointmaps and soft masks. This is **CPU preprocessing**, including for
future Vulkan inference, not a Vulkan neural graph. It has no Python/PyTorch3D
runtime dependency. The interface is currently internal C++.

This does **not** establish MoGe inference, selected-checkpoint configuration,
the complete pointmap image/crop path, or learned Objects reconstruction. It is
one component of O1. The point-window conditioner is separately covered in
[OBJECTS_POINTPATCH.md](OBJECTS_POINTPATCH.md).

## Behavior retained

| Variant | Shift | Scale |
| --- | --- | --- |
| Basic SSI | Scene median Z, zero X/Y | Mean absolute centered coordinate |
| Object-centric scene | Per-axis masked median | Scene median of maximum absolute coordinate |
| Object-centric quantile | Per-axis masked median | Twice the selected object-radius quantile range |
| Object-centric norm median | Per-axis masked median | Median object radius |
| Apparent-size scene/object | Zero | Scene/object median Z |
| Disparity scene | Median log Z plus configured offset, zero X/Y | One |
| Disparity object | Per-axis masked median of X/Z, Y/Z, log Z | One |

The three object-centric variants and the two apparent-size variants additionally
apply the configured scale factor. Masks resize with nearest interpolation and
select values strictly greater than 0.5. `nanmedian` uses the lower middle element
for an even count, not an average. NaNs are ignored where upstream uses nan
reductions; ordinary max/norm operations retain their NaN propagation.

The original override rules differ and are preserved:

- Basic, apparent-size and disparity modes use overrides only when both scale
  and shift are supplied; supplying only one causes both to be recomputed.
- Object-centric modes compute statistics first and independently apply any
  supplied overrides only when `allow_override` is enabled.

Clipping differs too: object-centric tests 3D radius, apparent-size tests positive
Z, and disparity tests absolute log-space Z. Clipped points become NaN triples.
Basic SSI has no clipping option in upstream. Denormalization consumes the
normalized map's own scale/shift; it does not recover clipped or invalid points.

Normalization follows upstream's composed **4x4 homogeneous row-vector
transform**, with inverse scale and translation. It is not simplified to
independent `(coordinate - shift) / scale` expressions. In particular, a NaN or
infinity in one input coordinate can propagate through zero matrix entries and
the homogeneous denominator to all three output coordinates. Disparity mode
also preserves remapping before the transform and inverse remapping afterwards.

## Supported inputs and checked failures

Pointmap inputs are CHW with dimensions 1..2048; mask dimensions are 1..4096 to
support original-resolution RGB masks, including the 4096x2160 official example.
Masks must be finite soft values in [0,1]. Overrides have either zero or three
values. Scale must be positive, finite and have a finite reciprocal; shift and
the composed transform must be finite. Quantile drop is in [0,0.5), scale factor
is positive, and clip is nonnegative. Unsupported modes, malformed extents and
invalid options return checked C++ errors.

An empty selected object mask is rejected, as the original object-centric max
on an empty tensor raises. All-invalid selected points use the original
scale-factor/zero-shift fallback or raise when configured. Invalid computed
scale/shift and overflow are rejected instead of silently feeding a poisoned
transform into subsequent inference. No epsilon or guessed fallback scale is
introduced. These explicit validity restrictions are not a claim that upstream
rejects every malformed input in exactly the same way.

## Original reference

The Objects source is `f91db411c50efee93d8db7aeb323885650f6f722`. PyTorch3D is
`75ebeeaea0908c5527e7b1e305fbc7681382db47`, the exact dependency pin in upstream
Objects. Selected source files were downloaded from the official repository and
reviewed before execution. Their hashes are recorded in
`reference/original_objects_ssi.py` and the fixture manifest. Attribution and
installed license text are in `NOTICE` and `LICENSES/PyTorch3D.txt`.

The loader executes unchanged selected original class/function ASTs inside the
reviewed offline/read-only container. Postponed annotations avoid importing
unused pose/rotation definitions; numerical methods are not rewritten. Original
PyTorch3D `Transform3d`, `Scale`, `Translate` and device helpers perform the actual
transform/inverse operations. Source-frame observers capture actual statistics,
matrices, homogeneous inputs and bmm results. No rewritten mathematical oracle
supplies expected intermediate tensors.

Each case has three bit-identical uninstrumented normalization/denormalization
runs and an exactly equal observed run. Native input contains only raw XYZ,
mask, options and requested overrides. Native normalization and denormalization
use their own intermediates throughout.

Reference settings are F32 PyTorch 2.7.0+cu128, one CPU thread, TF32 disabled.
CPU and CUDA are captured separately. Strict deterministic mode initially
**rejected original CUDA nanmedian-with-indices**. The normalizers use only its
values, not its tie indices. CUDA capture therefore enables deterministic
**warnings**, retains the original operation, and requires bit-identical actual
values across repeats and instrumentation. This policy is recorded in the
manifest, not silently hidden or presented as strict deterministic CUDA support.

Reference arguments, inside the reviewed container described in [MHR.md](MHR.md):

```sh
/work/reference/capture_objects_ssi.py \
  --upstream /work/reference/upstream/sam-3d-objects \
  --pytorch3d /work/reference/upstream/pytorch3d-selected \
  --output /output --device cpu
```

Use `--device cuda` with the explicit NVIDIA device and deterministic cuBLAS
workspace configuration for the CUDA capture. The reviewed PyTorch3D subset
contains `pytorch3d/transforms/transform3d.py` and `pytorch3d/common/datatypes.py`;
there is no C++ extension build or PyTorch3D installation.

```sh
uv run --project reference/python --frozen python scripts/run_objects_ssi.py \
  --reference generated/fixtures/objects-ssi-cpu \
  --runner build/debug/bin/sam3d-objects-ssi-capture \
  --output generated/fixtures/objects-ssi-native-cpu
```

Change only the reference directory to compare the same native CPU preprocessing
with the CUDA reference. This must not be labelled a native Vulkan test.

## Results

All **475 comparisons pass against each reference device**: 383 operation
boundaries plus 92 independently captured final fields (normalized map, scale,
shift and own denormalized map). The 23 cases cover all eight variants, soft-mask
resizing, overrides, clipping, partial nonfinite triples, all-invalid fallback,
and all eight variants at 257x259 input resolution.

Rules were fixed before native comparisons: maximum absolute error <=1e-4 and
relative L2 <=2e-5. Resized masks require exact equality. NaN/+Inf/-Inf categories
must match exactly at explicitly allowed point/statistic boundaries; finite
elements still face both error assertions. Matrices, scale and shift must be
finite. Matching a few selected finite entries cannot pass a changed invalidity
mask. No thresholds were increased to obtain a pass.

| Comparison | Worst maximum absolute error | Worst relative L2 |
| --- | --- | --- |
| Native CPU / original CPU | 9.537e-7 | 5.530e-8 |
| Native CPU / original CUDA | 1.431e-6 | 6.402e-8 |

Report SHA-256s:

- CPU: `a06717d79b6cdafe776ebd54eedd07ecda191a96131067e086e009f310df9135`.
- CUDA-reference comparison: `668083b1ca98851498a0680dc887705954c33cba9027de65af9edaaa0a3c951d`.

The normal CTest fixture contains all 249 boundaries of the 15 small cases, not
a subset of vertices. Its SHA-256 is
`878f073704d1254ba48a32fcaf98f8090757c6f128601c7533db58f5bc7f62a2`.
Native tests also check malformed input, invalid homogeneous transforms,
observation/output-only equality and cross-coordinate NaN propagation. Python
tests verify fixture/source identities and reject truncated, duplicate or
oversized keyed captures. All 29 native sanitizer tests and 49 Python tests pass.

`sam3d-objects-ssi-fuzz` exercises the internal normalizer/denormalizer with
ASan/UBSan/LSan; **100,000 cases pass**, seed 61002, in 17 seconds with 432 MB
peak RSS and no sanitizer finding. Fuzzer binary SHA-256:
`4d13c155ef2522ec4a57fe169291d2ca4748a464e1e6e9e4333e4f50ed19154c`.
GGUF loading is excluded. This is not yet public SSI C API fuzz
coverage. Full learned models, browser acceptance and performance parity remain
separate unfinished gates.
