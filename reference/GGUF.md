# GGUF component conversion and validation

Three component formats exist. The initial **Body DINOv3 H+ backbone** format is architecture
`sam3d.body.dinov3.vith16plus`, schema version 1. It does not include the Body
decoder, MHR, or any Objects model. The real Body backbone has now been extracted,
byte-verified and converted; see [trained checkpoint evidence](TRAINED_BODY.md).
Its 36 trained stage and 704 operation checks pass on CPU/Vulkan under frozen
reference-derived F32 policies; full Body gates remain open. The separate
`sam3d.mhr.lod1` format contains a verified
conversion of the independently public official MHR geometry asset; see
[MHR.md](MHR.md). Native standalone MHR geometry now passes CPU/Vulkan operation
and final-vertex checks on the original demo inputs. Body output mapping is now
connected to the trained pose-branch companion, `sam3d.body.pose_branch`.
Its GGUF-only Vulkan run passes 646 trained comparisons; CPU and full Body
inference remain incomplete. See [trained branch evidence](TRAINED_BODY_BRANCH.md).

## Trained pose-branch companion

This schema contains 316 tensors (57,832,436 stored elements): 313 F32 tensors
for conditioning, six decoder layers, pose/camera/hand-box heads and mapping,
plus three I32 hand-index/topology tensors. It excludes hand-crop refinement
and mask conditioning. The learned **no-mask embedding is included**.

After the verified restricted Body extraction, convert without Torch:

```sh
uv run --frozen python scripts/convert_body_branch.py \
  --input models/body-other-state.safetensors \
  --manifest models/extraction.json \
  --output models/body-pose-branch-f32.gguf
build/debug/bin/sam3d-body-branch-verify models/body-pose-branch-f32.gguf
```

The converter requires the pinned safe-state, source/config hashes and Body
revision. It preserves every F32 payload byte, bounds-checks I64-to-I32 indices,
checks handedness partitions, and reads back the entire output before atomic
publication without overwrite. The converted companion is 231,353,792 bytes,
SHA-256 `eebd51ac66bab764b52c7c51a47671bfa6152980a5bc7da6ea7fe20f6ae61431`.
Metadata declares required backbone and geometry identities and its limited
output scope. As with other components, native metadata validation is not
cryptographic authentication of an externally supplied model.

## Backbone schema

The schema contains 552 F32 state tensors: 840,633,600 learned parameters plus
RoPE periods and key-bias masks. Shapes were checked against the original
`dinov3_vith16plus(pretrained=False)` factory on PyTorch's `meta` device, without
allocating model storage or loading weights. Both the Python converter and C++
reader match every original shape. This says nothing about learned tensor values.

## Safe input boundary

`scripts/convert_gguf.py` accepts only a verified F32 `.safetensors` intermediate.
It never imports PyTorch, deserializes pickle, downloads models or runs remote
code. Official checkpoint deserialization and any BF16/F16-to-F32 upcasting must
happen separately in the reviewed, isolated reference environment. That extraction
step is implemented for the official Body checkpoint in
`reference/extract_body_checkpoint.py`, with independent safe-reader byte checks.
Do not manufacture a provenance manifest as a substitute for source verification.

The extraction manifest must contain exactly these fields:

- `schema_version`: integer `1`.
- `architecture`: `sam3d.body.dinov3.vith16plus`.
- `body_revision`, `dinov3_revision`: the pinned revisions in `scripts/gguf_schema.py`.
- `checkpoint_sha256`, `config_sha256`: the pinned identities from the same file.
- `safetensors_sha256`: the actual safe intermediate's SHA-256.
- `source_precision`: unique source dtypes, for example `["BF16", "F32"]`.

The converter verifies the safe intermediate's hash before and after streaming,
exact tensor names/shapes/F32 type, finite values, key masks and positive RoPE
periods. It rejects duplicate JSON keys, malformed metadata and unexpected model
identities. It writes a temporary file and publishes it atomically without
overwriting an existing destination, including a dangling symlink.

Once a verified safe intermediate and manifest exist:

```sh
uv run --frozen python scripts/convert_gguf.py \
  --input models/body-dinov3-f32.safetensors \
  --manifest models/body-dinov3-manifest.json \
  --output models/body-dinov3-f32.gguf
build/debug/bin/sam3d-gguf-verify models/body-dinov3-f32.gguf
```

Tensor bytes retain PyTorch contiguous order. Only the dimension list is reversed
for GGML; no tensor data transpose occurs. GGUF v3, alignment 32, F32 data and the
exact component metadata contract are required. This first format intentionally
rejects arbitrary GGUF variants, quantization and other model revisions.

The native reader checks bounded metadata before handing the same checked header
snapshot to GGML. It verifies file/data sizes, rank, dimensions, offsets and
identities, then reads tensors on demand with a caller-supplied byte budget.
The verifier checks all tensors sequentially; it does not execute a neural graph.
Model files must remain immutable and reside in trusted directories. Metadata
hash strings provide provenance, **not authentication**: the native reader does
not recompute source hashes or guarantee the origin of externally supplied bytes.

## Reproduce the architecture and format checks

The small, checked-in `tests/fixtures/dino-schema.json` is the original meta-factory
report, including source hashes and PyTorch version. Regenerate it using the
reviewed reference image from [README.md](README.md), offline:

```sh
mkdir -p generated/fixtures/dino-schema
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --memory 4g --pids-limit 128 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  -v "$PWD:/work:ro" -v "$PWD/generated/fixtures/dino-schema:/output:rw" \
  --entrypoint python sam3d-reference-image /work/reference/capture_dino_schema.py \
  --upstream /work/reference/upstream/dinov3 --output /output
uv run --frozen python scripts/check_gguf_contract.py \
  --reference-schema generated/fixtures/dino-schema/schema.json \
  --verifier build/debug/bin/sam3d-gguf-verify \
  --archive-test build/debug/bin/sam3d-archive-test
```

The last command checks both complete schemas against upstream, converts two
tiny synthetic tensors, reads them using the native/GGML path, and confirms the
full-backbone verifier rejects that intentionally incomplete test archive.
It needs no learned weights and removes its temporary files automatically.
The checked-in schema can be used instead of regenerating it for regression.

Normal CTest also verifies a file written by GGML itself and deterministic
malformed-input cases under ASan/UBSan/LSan. Python unit tests cover converter
rejections and compare its full schema with the original report. GGUF loading
is deliberately excluded from fuzzing, as specified in the design; other public
C API/parser surfaces remain sanitizer/fuzzer targets.
