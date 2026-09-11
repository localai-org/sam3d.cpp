# Exact official MHR geometry reference

MHR is a geometry model, not the SAM image estimator. Its independently public
[official v1.0.1 release](https://github.com/facebookresearch/MHR/releases/tag/v1.0.1)
includes an Apache-2.0 license and the **same** MHR TorchScript byte identity
listed for the SAM 3D Body companion. No access to the separate gated SAM neural
checkpoints has been granted by this download.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `assets.zip` | 198,943,157 | `e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94` |
| `assets/mhr_model.pt` | 696,110,248 | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` |
| `assets/LICENSE.txt` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

All hashes were verified locally. Only the model and license were extracted,
not the other LOD assets (4.77GB uncompressed altogether). The official demo/source
checkout is pinned to `e412e12c9d7287a598f00edf19242b476b440211`.
The **released TorchScript**, not a rewrite or newer Python class, is the geometry
numerical authority.

## Obtain and inspect without loading code

```sh
mkdir -p generated/models/mhr-public
curl --fail --location --output generated/models/mhr-public/assets-v1.0.1.zip \
  https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
sha256sum generated/models/mhr-public/assets-v1.0.1.zip
unzip -l generated/models/mhr-public/assets-v1.0.1.zip
```

Require the archive hash above before extracting the two explicit members:

```sh
unzip -n generated/models/mhr-public/assets-v1.0.1.zip \
  assets/mhr_model.pt assets/LICENSE.txt -d generated/models/mhr-public
sha256sum generated/models/mhr-public/assets/mhr_model.pt
```

Do not load TorchScript/pickle on the host. `capture_mhr.py` verifies the exact
asset size/hash before loading in the reviewed offline container. It also
verifies the original demo input-generator source hash and executes its unchanged
function AST. Source assets must remain immutable during this process.

## Isolated full geometry reference

Use the reference image documented in [README.md](README.md):

```sh
mkdir -p generated/fixtures/mhr-cpu
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --memory 8g --pids-limit 128 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  -v "$PWD:/work:ro" -v "$PWD/generated/fixtures/mhr-cpu:/output:rw" \
  --entrypoint python sam3d-reference-camera /work/reference/capture_mhr.py \
  --model /work/generated/models/mhr-public/assets/mhr_model.pt \
  --upstream /work/reference/upstream/MHR --output /output --export-state
```

Capture CUDA with the documented NVIDIA device exposure,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, a separate output directory and `--device cuda`.
Exporting state is only needed once. No packages/models are downloaded at runtime.

This executes actual original `forward`, with/without correctives, on two official
demo parameter samples. Results are `[2,18439,3]` vertices in centimeters and
`[2,127,8]` skeleton states (translation, XYZW quaternion, scale). Safe export
includes state inventory/hashes, safetensors, names, prefix schedule, embedded
code hashes and inlined graph. This is full standalone MHR geometry, **not** the
complete SAM image-to-body entry point or native parity.

Three fresh CPU processes repeat byte-for-byte. Three CUDA processes repeat
skeletons and non-corrective vertices exactly; corrected vertices vary at 1–7
coordinates by at most 1.91e-6 cm. Original CPU/CUDA vertex max-abs differences are
6.10e-5 cm without correctives and 4.58e-5 cm with correctives. One-shot timings
are diagnostic, not optimized performance baselines.

`trace_mhr_repeat.py` observes original ATen results without replacing operations.
With the same container/model arguments, supply `--reference` to the CUDA fixture
directory and a fresh `--output` directory. It traces 380 initialized floating
boundaries after two original warm-up calls. First variation: `aten.matmul.default`,
sparse COO `[3000,750]` by dense `[750,2]`, with repeated max-abs 1.49e-8.
Observed versus unobserved geometry differs by at most 9.54e-7 cm, with exact
skeletons: bounded instrumentation equivalence, not an exact-equality claim.
Deterministic PyTorch poisons uninitialized scratch with NaNs; identified
allocation/view results are recorded separately, while non-finite arithmetic
remains an error. The initial naive scratch comparison is retained in ignored
generated artifacts, not treated as a numerical divergence.

## Preserve the released computation

Current Python source is not identical to the released asset:

- The asset separately projects identity and expression blendshapes and pads 204
  parameters with **45** zeros for its `[889,249]` parameter transform. Current
  Python combines these assets and uses different padding; do not transplant it.
- The first corrective projection is sparse COO, not current Python's dense
  tracing workaround: 750 features → 3000 activations → ReLU → `[55317,3000]`
  corrective projection.
- Local skeleton state is F32, but prefix FK accumulates in **F64**, then returns
  F32. Preserve stored prefix groups `[65,56,62,83]`, quaternion normalization
  and multiplication order.
- Skinning multiplies global and inverse-bind states, transforms selected vertices
  and performs weighted `index_add`. Joint and vertex indices have distinct domains.

The native stages below preserve this prefix schedule and include identity,
expression and pose blendshapes plus skinning. The original released forward
remains the numerical authority, not a rewritten Python oracle.

## Native skeleton operation parity

`capture_mhr_skeleton.py` executes the original released parameter/local/FK methods
on the original two demo samples, observing their outputs with a non-replacing
dispatch tracer. `torch.jit.optimized_execution(False)` exposes the unfused local
arithmetic. Equivalence against an uninstrumented warmed full-model call is
measured: CPU is exact; CUDA differs at max-abs 1.53e-5, relative-L2 5.15e-8.
This is bounded CUDA equivalence, not bit-identical observation. Both the original
full-model skeleton and the observed operations are retained and checked natively.

Using the same offline container and mounts above, change the entry point to
`/work/reference/capture_mhr_skeleton.py`, use a fresh output directory, and pass:

```sh
--model /work/generated/models/mhr-public/assets/mhr_model.pt \
--reference /work/generated/fixtures/mhr-cpu --output /output --device cpu
```

For CUDA use its reference directory, GPU exposure and `--device cuda`.
Then run the native comparison on the host (no host PyTorch):

```sh
uv run --project reference/python --frozen python scripts/run_mhr_skeleton.py \
  --reference generated/fixtures/mhr-skeleton-cpu \
  --runner build/debug/bin/sam3d-mhr-skeleton-capture \
  --module build/debug/bin/libggml-cpu.so \
  --gguf generated/models/mhr-public/mhr-lod1-f32.gguf \
  --output generated/fixtures/mhr-skeleton-native-cpu --threads 12
```

For Vulkan choose that build's executable and `libggml-vulkan.so`, the CUDA
reference directory, `--backend Vulkan`, and the desired device index/description.
The driver configuration remains a caller choice. The runner explicitly disables
F16 operand downcasts and cooperative-matrix paths for this strict-F32 comparison.
Use the documented UBSan-only Vulkan build only for the diagnosed ASan/ICD issue.

The native path pads 204 parameters with 45 zeros and performs the real GGUF
`[889,249]` projection in GGML. Its own result feeds Euler trig, quaternion
prerotation, local translation/scale and all four F64 prefix passes. Recorded taps
include normalization, cross products, state products, pass outputs and final F32
skeleton. Indices, parent topology, group uniqueness and the complete prefix
schedule are validated before use; in-range but topologically incorrect schedules
are rejected too. CPU trigonometry/F64 geometry is explicit even on Vulkan.

Both CPU and NVIDIA pass all 46 operation taps and the separate full-model
skeleton check, with frozen max-abs 1e-4 / relative-L2 2e-5 gates. CPU worst
max-abs is 7.63e-6; Vulkan/CUDA worst is 1.53e-5. No upstream joint parameters or
local/FK results are injected in this composed test. This does **not** establish
final mesh, SAM image-to-body, fully GPU-resident or performance parity.

The normal sanitizer test `mhr_local_skeleton_upstream` separately exercises local
transforms/FK using a small original fixture, plus malformed shape/value/parent/
schedule rejection. Its inputs deliberately start at joint parameters; this
isolated test is not evidence for the GGUF parameter projection. The 484KB fixture
contains small Apache-licensed geometry constants, not learned matrices. Recreate
it to a new path with `run_mhr_skeleton.py --reference … --fixture …`.

| Evidence | SHA-256 |
| --- | --- |
| Original CPU operation tensors | `a39adb8a4ede22d1fa4ff1a2eed9b3e816cc944f533c24bd2d376d52f3fd3bcf` |
| Original CUDA operation tensors | `8992c9c65d1f2f0e6630c32a1f331522765bba1bd2cde919fdd576cab08c1583` |
| Native CPU tensors | `dba17da32cb48b745b9edfea82e4b9543b571d0b9bc12704c9dbf3ddbd5163a1` |
| Native NVIDIA Vulkan tensors | `f8e0fc31e62198a8f015432c5eff98c619b6fab9fceba8c521577a2a069841e7` |
| Normal local/FK fixture | `3b08436b0506f85a1f98acf0d4796d9440d14ab58fbc7af655a6996a48dd24e9` |

## Complete native MHR geometry

`src/mhr_geometry.cpp` now implements the released model's complete forward:
identity projection/base shape, separate expression projection, own parameter
transform/local/F64 skeleton, pose features, both corrective projections and ReLU,
inverse-bind composition, twice-normalized point quaternion transforms and weighted
vertex accumulation. It returns all 18,439 vertices in **centimeters**, before SAM
Body's later unit/axis/keypoint mapping. No MHR intermediate comes from upstream
in the composed native run; only the original demo inputs and converted GGUF do.

Neural projections run on the requested GGML backend in F32. The first corrective
COO matrix is materialized densely (3000×750, 9MB) with all exact original nonzeros
for GGML matmul. This is a native storage/operation choice, **not** replacement of
the sparse upstream oracle. CPU/native sparse-projection max-abs is 4.92e-7,
relative-L2 3.23e-7. Trigonometry, small state transforms and weighted skinning
currently run on CPU; weights are loaded/projected in stages, not retained in a
persistent whole-model GPU session. This is not a performance or GPU-residency
claim. Those remain later gates.

`capture_mhr_geometry.py` observes an original full `MHRDemo.forward`, with
correctives both off and on. It records operation-level outputs without replacing
operators, verifies the final tap is the returned mesh, and retains an independent
uninstrumented optimized full-model result. A warmed CUDA model retains cached
fused execution plans even under `optimized_execution(False)`: using it for taps
initially omitted blendshape operations and correctly failed capture. Observation
now uses a fresh load with its first call unoptimized. CPU observation is exact;
CUDA observed/uninstrumented vertex max-abs is 4.58e-5 cm, relative-L2 1.32e-7.
Thus CUDA observation equivalence is bounded, not bit-exact.

Using the same isolated container/model/mount arguments as above, capture with:

```sh
# Arguments to the container's Python entry point:
/work/reference/capture_mhr_geometry.py \
  --model /work/generated/models/mhr-public/assets/mhr_model.pt \
  --reference /work/generated/fixtures/mhr-cpu --output /output --device cpu
```

Use a fresh output directory (for example `generated/fixtures/mhr-geometry-cpu`).
Capture CUDA separately with its reference inputs, GPU exposure and `--device cuda`.
The host runner needs NumPy/safetensors, **not PyTorch**:

```sh
uv run --project reference/python --frozen python scripts/run_mhr_geometry.py \
  --reference generated/fixtures/mhr-geometry-cpu \
  --runner build/debug/bin/sam3d-mhr-geometry-capture \
  --module build/debug/bin/libggml-cpu.so \
  --gguf generated/models/mhr-public/mhr-lod1-f32.gguf \
  --output generated/fixtures/mhr-geometry-native-cpu --threads 12
```

For Vulkan select that build's executable/backend module and the CUDA fixture,
add `--backend Vulkan`, and select the device as in the skeleton example.
Both native runs pass all 48 checks: 19 operation taps without correctives, 25
with correctives, plus two independent final outputs per mode. Fixed gates remain
max-abs 1e-4 and relative-L2 2e-5. Final vertex comparisons on the two official
demo samples:

| Backend / original reference | Correctives off max-abs (cm) | Correctives on max-abs (cm) |
| --- | ---: | ---: |
| Native CPU / PyTorch CPU | 4.58e-5 | 4.58e-5 |
| Native NVIDIA Vulkan / PyTorch CUDA | 6.10e-5 | 4.58e-5 |

CPU final-vertex relative-L2 is at most 5.41e-8; Vulkan/CUDA at most 1.15e-7.
CPU tests retain ASan/UBSan/LSan. Vulkan retains UBSan with the documented
ASan/driver exception. Timing includes diagnostic transfers/allocation and is not
optimized performance acceptance.

Normal CTest additionally runs `mhr_skinning_upstream` on 16 selected actual
vertices (46 influences), with 12 original skinning boundaries and malformed
shape/index/weight/state rejection. `capture_mhr_skin_fixture.py` restricts only
the original module's vertex/influence buffers, preserving their order, and checks
the result against those same vertices of the original full mesh: CPU is exact.
The 96KB fixture deliberately starts from supplied skeleton/unposed inputs, so it
is an isolated LBS regression, not a substitute for the full GGUF test. Its small
asset constants and provenance are distributed under the original asset terms.

| Correctives-on tensor artifact | SHA-256 |
| --- | --- |
| Original CPU | `81a4e53fb79a49a9dc20d4868fbfa42db6622f6f448c7faf9e52b01c789760a3` |
| Original CUDA | `8e90aa7e72a8e0450dadc57fda3225fc866fad20f77060ee1a2cce7e85b66437` |
| Native CPU | `5474d212944bb5a081d6181c10edc866ffda51d75849b3bb97a0645ce4d9cb4d` |
| Native NVIDIA Vulkan | `cd4ec99a5ca8167f06c3f3cf1e4f99981a23d4644e6148a79efb1a90bfdc2300` |

This completes standalone released-MHR geometry on the captured original demo
inputs. The subsequent [Body mapping/pose composition](BODY_OUTPUT.md) is tested
separately with synthetic SAM head state. Neither stage completes trained SAM
image estimation, hand refinement, full decoder feedback, the public inference
C API, image-to-mesh browser QA, Objects or optimized performance acceptance.

## Safe GGUF conversion

```sh
uv run --project reference/python --frozen python scripts/convert_mhr_gguf.py \
  --input generated/fixtures/mhr-cpu/state.safetensors \
  --manifest generated/fixtures/mhr-cpu/manifest.json \
  --geometry generated/fixtures/mhr-cpu/geometry.json \
  --output generated/models/mhr-public/mhr-lod1-f32.gguf
build/debug/bin/sam3d-mhr-gguf-verify \
  generated/models/mhr-public/mhr-lod1-f32.gguf
```

Architecture `sam3d.mhr.lod1`, schema 1: 18 required tensors, 173,275,904 elements,
693,111,264 bytes. Floats remain F32; bounded indices become I32, never floating
indices. Names, units, original identities and mixed-precision FK semantics are
metadata. Unused solver-limit state is omitted. Current GGUF SHA-256:
`d52ab772628fb6d851550381b428185a32da6398793f0ff70299bad1999ba8f8`.

The converter accepts safetensors only, checks hashes/layouts/names/index bounds
and publishes without overwriting. The bounded native reader validates all real
tensors with ASan/UBSan/LSan. Tiny deterministic rejection tests need no weights;
GGUF loading is not fuzzed. Metadata hashes are provenance, not authentication:
trusted extraction and immutable inputs remain necessary. Successful conversion/
loading is not geometry inference or model parity.
