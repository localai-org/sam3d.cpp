# Objects point-window transformer

`src/objects_pointpatch.cpp` implements evaluation of upstream `PointPatchEmbed`:
supplied XYZ pointmap -> nearest resize -> validity/remapping -> point projection
-> learned invalid-point replacement -> window CLS/position tokens -> one timm
transformer block -> CLS extraction/patch position -> optional forced dropout.
Projection, normalization, attention and MLP execute using F32 GGML operations.
Coordinate remapping, window assembly and final positional additions are native
CPU operations. No Python inference dependency is introduced.

**This is synthetic-state component parity, not trained Objects reconstruction.**
MoGe pointmap estimation, SSI normalization, image conditioning and the downstream
generators/decoders are not covered. The default source architecture is tested;
the gated published configuration has not yet been verified. See
[OBJECTS_IMAGE.md](OBJECTS_IMAGE.md) for the separately tested RGBA path.

## Contract

The default architecture uses 256x256 resized points, 8x8 windows, 768 channels,
16 attention heads and an MLP expansion of two. Each window has 64 points plus
a CLS token. The block uses epsilon 1e-6 LayerNorm, packed biased QKV, scaled
dot-product attention and exact-erf GELU. There is no RoPE or LayerScale here.

All five original coordinate-remapping modes are implemented. NaN/Inf input
triples select a learned invalid-point token unless a post-resize explicit binary
valid mask is supplied. A point explicitly marked valid must have finite values
and a valid remapping domain; malformed inputs receive a checked error instead
of poisoning attention. Invalid zeros can become nonfinite under disparity
remapping, as in upstream, but are replaced before the transformer.

Evaluation with a trained dropout token is supported, including forced dropout
which returns that token plus patch position. Stochastic training is not.
Inputs, parameters, architectures, mask extents and chunk sizes are validated.
The current interface is internal C++; it is not a public model/session API.

Windows are independent. Native execution uses bounded chunks (default 32),
with streamed, window-major diagnostic tensors and checked offset/total lengths.
Reference inference processes the whole input without native chunking. Weights
are currently uploaded for each chunk and captures read back each stage; this
is not a resident or performance-optimized implementation. Full-size diagnostic
artifacts consume about 4.6 GiB per reference/native capture, outside Git.

## Original reference, not a rewritten oracle

The Objects source pin is `f91db411c50efee93d8db7aeb323885650f6f722`.
`reference/original_pointpatch.py` hash-verifies the original `PointPatchEmbed`
and `PointRemapper` files and the official PyPI **timm 0.9.16** wheel, the exact
dependency pinned by Objects. Wheel SHA-256:
`bf5704014476ab011589d3c14172ee4c901fd18f9110a928019cac5be2945914`.
It executes unchanged selected class/function ASTs in the reviewed isolated
container, avoiding unrelated model registry imports. Original timm config
defaults and its own attention selector are retained. No third-party C++ engine
is executed or used as an oracle. Attribution is in `NOTICE`.

`capture_objects_pointpatch.py` runs the original uninstrumented module twice
with identical results, then observes a third run and requires exact equality.
Module/source-frame hooks capture actual original values. A dispatch observer
captures the actual SDPA bmm/softmax operations; logits/probabilities are not
computed by a separate replacement attention formula. Captured layout changes
only reshape/permute original values into a canonical window-major order.

The reference is PyTorch 2.7.0+cu128, F32, eval, one CPU thread, deterministic
algorithms, SDPA MATH and TF32 disabled. CPU and CUDA references are separate.
Synthetic parameters/input values are stored, so native execution never depends
on cross-framework random-seed equivalence or injected computed intermediates.

Run the reference script in the reviewed offline/read-only container described
in [MHR.md](MHR.md), with an explicit output mount and memory limit:

```sh
/work/reference/capture_objects_pointpatch.py \
  --upstream /work/reference/upstream/sam-3d-objects \
  --timm-wheel /work/generated/wheels/timm-0.9.16-py3-none-any.whl \
  --output /output --device cpu
```

Use `--device cuda` with the NVIDIA container device and deterministic cuBLAS
workspace configuration for the GPU reference. Add `--full-size` for the default
256/8/768 architecture with a 257x259 input and invalid points. The ordinary
capture has six cases covering batch/layout, all remappers, explicit validity,
forced/unforced dropout and width 768. A 5 GiB reference-container memory bound
was used for full-size capture. The original modules are not memory-rewritten.

```sh
uv run --project reference/python --frozen python scripts/run_objects_pointpatch.py \
  --reference generated/fixtures/objects-pointpatch-cpu \
  --runner build/debug/bin/sam3d-pointpatch-capture \
  --module build/debug/bin/libggml-cpu.so \
  --output generated/fixtures/objects-pointpatch-native-cpu \
  --threads 12 --chunk 7
```

For GPU select the Vulkan-UBSan build/module, `--backend Vulkan`, its explicit
device/description, and the CUDA reference. The runner disables F16 and both
cooperative-matrix paths to avoid hidden operand downcasting. An explicit Vulkan
ICD path may be needed on multi-vendor systems. ASan/UBSan/LSan stays enabled for
CPU; the separate Vulkan build uses UBSan because of the previously diagnosed
NVIDIA ICD/ASan startup incompatibility. This is not a general sanitizer waiver.

## Acceptance rules and regression

Each case compares all **27 operation boundaries plus the independently captured
uninstrumented final output**, using native-produced intermediates throughout.
Four boundaries require exact equality: resized input, validity, safe input and
patch position. Other boundaries require both maximum absolute error <=1e-4
and relative L2 <=2e-5, frozen before running native tests.

Only resized input, remapped points and raw point projection permit nonfinite
values. Their NaN, positive-infinity and negative-infinity masks must match
exactly; finite elements still face both error assertions. All later tensors,
including the final output, must be finite. Python negative controls reject
changed invalid categories, hidden finite errors and nonfinite final outputs.

The normal weight-free CTest consumes every tensor of the first original small
case in `tests/fixtures/objects-pointpatch.txt`; its adjacent JSON retains the
capture provenance. The fixture SHA-256 is
`3470733875914d39ddf54a241233656ba608c9d7fd68657b6c2da9e576bfd52a`.
Tests compare chunk sizes 1, 7 and 32, require complete non-overlapping tap
coverage, and reject malformed shapes, parameter sets, masks, remapping domains
and chunk lengths. No downloads occur in normal tests.

The six-case run passes **168 checks per backend**. Report SHA-256s:

| Report | SHA-256 |
| --- | --- |
| Native CPU / original CPU | `27bd014651fa3952c9d8e079b073d75643b59d2cce2fe9848d98438c5224ec1f` |
| Native Vulkan / original CUDA | `90586bab5aef45a8b220944109da7361fbf61c1a80e473f2dba97c0374ccfc15` |

The full default-size run additionally passes **28 checks per backend**, for
**196 per backend in total**. Both original observer/non-observer comparisons
are exact. Native CPU uses ASan/UBSan/LSan and 12 threads; Vulkan uses the NVIDIA
device with UBSan. Both use 32-window chunks, producing all 1024 output tokens.

| Full-size result | CPU / PyTorch CPU | Vulkan / PyTorch CUDA |
| --- | --- | --- |
| Worst operation maximum absolute error | 3.577e-6 | 9.060e-6 |
| Worst operation relative L2 | 5.144e-7 | 1.074e-6 |
| Independent final maximum absolute error | 2.146e-6 | 5.603e-6 |
| Independent final relative L2 | 4.385e-7 | 8.408e-7 |

Full-size report SHA-256s:

- CPU: `be020cc3aca5a2986509ecabf0dbd0c951dc77c5bd231f464c941a82ce39b7ff`.
- Vulkan: `5b3d1d7e3a3bbd52ba556e604dcd4ea5cbe60f67a1050f6fa8ac8eea51daf6ff`.

All 28 normal native sanitizer tests and 43 Python tests pass after adding this
component. Thresholds were not changed to obtain these results.

No trained model, visual reconstruction, browser QA or performance-parity claim
follows from these component tests.
