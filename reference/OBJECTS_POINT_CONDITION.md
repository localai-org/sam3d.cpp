# Composed point conditioning

`src/objects_point_condition.cpp` connects raw RGBA and supplied CHW XYZ through
native preprocessing, a shared PointPatch encoder and the native condition fuser.
The native runner accepts raw input bytes, explicit options and model state;
it does **not** receive saved preprocessed tensors or encoder embeddings.

This is an explicit **point-only configuration with synthetic neural state**.
It is not the checkpoint's verified image/point conditioner, MoGe pointmap
estimation, learned reconstruction, browser acceptance or performance parity.
The image inputs determine crop/mask/normalization, but no image neural encoder
participates yet. Official model access was granted on 2026-09-09; these captures
remain synthetic and do not become trained-model evidence through that approval.

## Reference and coverage

The oracle is a segmented original pipeline: hash-verified outputs from the
unchanged upstream preprocessing method, followed by unchanged upstream
`PointPatchEmbed` and `EmbedderFuser` classes. The preprocessing manifest and
every selected raw-input/output artifact are checked before use. This retains
the original pipeline's own intermediates while avoiding repeated capture of
already-verified large preprocessing diagnostics. It is not a separately run
unmodified complete published inference pipeline; that remains a later gate.

The neural reference runs three identical uninstrumented passes plus an observed
pass in the isolated container. It uses actual PointPatch encoders in the fuser,
not the supplied-embedding boundary used in the standalone fusion tests.
Separate CPU/CUDA captures share the exact CPU preprocessing fixture. TF32 is
disabled and F32 math SDPA is selected. No config or neural checkpoint defaults
are inferred from these deliberately explicit tests.

The composition compares all 11 preprocessing fields, nine module boundaries per
encoder invocation, every fuser boundary, independently returned encoder buffers
and the final conditioning tensor. Finer per-operation SDPA/remapping and SSI
diagnostics remain in the standalone tests described in
[PointPatch](OBJECTS_POINTPATCH.md), [preprocessing](OBJECTS_PREPROCESS.md) and
[fusion](OBJECTS_FUSER.md). These connected tests add own-input propagation checks;
they do not replace those operation-level tests.

Two small cases cover object/full-image modalities, distinct preprocessing and
encoder resolutions, shared weights, positional groups and forced scene drop.
The full-size case uses the official 4096x2160 kid_box image/mask with supplied
257x259 synthetic XYZ, 518x518 RGB, 256x256 point fields, and PointPatch's
256/8/768 architecture on both point fields.

## An actual configuration-domain failure

The initial second case paired object-centric normalization with the `exp`
point remapper. Its normalized full-image depth included values below -1.
Original `PointRemapper.forward` then applies `log1p(z)`: four nonfinite values
first appeared there despite a completely finite safe input. The final original
conditioning contained 64 nonfinite values. Forced modality dropout did not
repair them because it multiplies by zero; NaN times zero remains NaN.

This failed the reference repeatability gate, rather than being counted as a
native parity failure or silently clamped away. The incompatible pairing is
retained as a negative control in both CPU/CUDA manifests. Native CPU and Vulkan
must reject it cleanly with `nonfinite PointPatch parameter/output`, exit code 1,
not an abort. The valid second test explicitly selects the original `sinh`
remapper (which computes `asinh`) for that normalization mode. The first and
full-size tests retain `exp` with compatible apparent-size normalization.
This is **not evidence that the published configuration has this problem**;
that configuration has not been downloaded or validated yet.

## Reproduction

Run reference capture only inside the reviewed isolated container, with read-only
source and a dedicated writable output directory:

```sh
python reference/capture_objects_point_condition.py \
  --upstream reference/upstream/sam-3d-objects \
  --timm-wheel generated/wheels/timm-0.9.16-py3-none-any.whl \
  --preprocessing generated/fixtures/objects-preprocess-cpu \
  --output generated/fixtures/objects-point-condition-cpu --device cpu
```

Use `--device cuda` and a separate output directory for the CUDA capture.
Add `--full-size` for the full original-photo-derived conditioning case.

```sh
python scripts/run_objects_point_condition.py \
  --reference generated/fixtures/objects-point-condition-cpu \
  --preprocessing generated/fixtures/objects-preprocess-cpu \
  --output generated/fixtures/objects-point-condition-native-cpu \
  --runner build/debug/bin/sam3d-point-condition-capture \
  --module build/debug/bin/libggml-cpu.so
```

For NVIDIA, select the Vulkan UBSan runner/module, matching CUDA reference,
`--backend Vulkan`, device index and expected description. The runner preserves
the existing strict-F32 flags. CPU verification keeps ASan/UBSan/LSan enabled.

Small-case validation passes **108 comparisons plus the negative configuration
check per backend**. Neural comparisons require finite values, maximum absolute
error <=1e-4 and relative L2 <=2e-5; prepared fields retain their stricter existing
RGB/mask and nonfinite-category rules. No tolerances were relaxed.

Normal CTest packages two small original raw-input/synthetic-state fixtures,
about 548 KB combined, using `scripts/make_point_condition_regression.py`.
Their hashes and original provenance are recorded in
`tests/fixtures/point-condition.json`; they contain no learned model weights or
official photographs. Tests compare all 14 returned fields, exact observed versus
output-only neural results, invalid remapping rejection and incompatible shape
rejection. Python tests enforce provenance and finite-output comparison rules.
The complete normal suite passes 35 native sanitizer tests and 66 Python tests.

The full-size Vulkan comparison additionally passes **54 checks**, bringing its
total to 162. Its worst intermediate error is 1.908e-5 absolute and 1.718e-6
relative L2; final conditioning error is 6.676e-6 absolute and 1.688e-6 relative
L2. The CPU full-size sanitizer run now also passes all **54 checks** (162 total),
with worst absolute error 8.584e-6 and worst relative L2 8.159e-7.

Current report SHA-256s:

- Small CPU: `086a2b51ea95b8cf6388f9a5a3b4c6d53f580c61f118ca5744f94c35def21add`.
- Small Vulkan: `8b057b20191f3290ece5eaba454a3c6d3c87c28a5c6a4e81cbc9bacc9daff18a`.
- Full Vulkan: `8ee3c48c1429eb9f925df0e34daffb6648979672952c05992ea4c78ec8bad771`.
- Full CPU: `d2060e4511e1beb4efff048a35eef9a060bca4e84faa2ba9dffad9a01ca3410d`.

The reports retain per-tensor metrics, artifact, native runner/module and
comparator identities. Trained-model, browser and performance gates remain open.
