# SAM 3D Objects conversion and reference tools

These scripts support the native experimental Objects path. They are offline
development tools and are not runtime dependencies.

The implementation is based on Meta's official `sam-3d-objects` source pinned
at `f91db411c50efee93d8db7aeb323885650f6f722`, with selected conversion and
GGML graph work adapted from Asher-1's `sam-3d-objects-ggml` revision
`1c14b7c3c3e8d9109b943ddc83a0a39c73744246`. MoGe weights and architecture
come from Microsoft's MoGe. See the root `NOTICE` for licenses and attribution.

## Convert the released checkpoints

Use a Python environment containing PyTorch, NumPy and the GGUF Python package.
The checkpoint directory is the released SAM 3D Objects `checkpoints` directory.
Convert the model family to the directory expected by the demo:

```sh
python scripts/objects/convert_sam3d_to_gguf.py \
  --checkpoint-dir generated/models/sam-3d-objects/checkpoints \
  --output generated/models/objects-gguf \
  --model all --dtype f16

python scripts/objects/convert_sam3d_to_gguf.py \
  --output generated/models/objects-gguf \
  --model moge_vitl --dtype f16 \
  --moge-checkpoint /path/to/moge-vitl/model.pt
```

The current end-to-end Vulkan path expects:

- `moge_vitl-f16.gguf`
- `ss_generator-f16.gguf`
- `ss_decoder-f16.gguf`
- `slat_generator-f16.gguf`
- `slat_decoder_gs-f16.gguf`
- `slat_decoder_mesh-f16.gguf`

The converter also contains development-only quantization modes. F16 weights
with F32 compute are the current numerical baseline; quantized Objects models
have a separate experimental status and are not selected by the demo.

## Capture and compare the mesh decoder

`capture_mesh_reference.py` runs the pinned upstream transformer, spconv
upsampling and FlexiCubes extractor on CUDA. Its default synthetic input is a
small window-boundary regression. Supply `--features` and `--coords` for a real
SLat, add `--extract-mesh`, and use `--compact` to retain stage boundaries and
topology taps without every large internal activation. The native comparison
executable accepts at most eight CPU threads. The accepted commands and metrics
are recorded in [the geometry parity report](../../reference/OBJECTS_MESH_PARITY.md).

## Replay an identical SS trajectory upstream

Set `SAM3D_OBJECTS_DUMP_DIR` on a native run to retain the initial random values,
condition tokens, per-step velocities and states. Then run the pinned official
SS generator over those exact values:

```sh
python scripts/objects/ss_trajectory_ref.py \
  --e2e-dir /tmp/native-objects-dump \
  --out-dir /tmp/upstream-ss-replay \
  --checkpoint generated/models/sam-3d-objects/checkpoints/ss_generator.ckpt \
  --config generated/models/sam-3d-objects/checkpoints/ss_generator.yaml \
  --device cpu --threads 8
```

This comparison deliberately reuses the native noise instead of assuming that
PyTorch and the C++ standard library produce the same random sequence from the
same seed. The measured exemplar and current acceptance boundary are recorded
in [`reference/OBJECTS_RUNTIME.md`](../../reference/OBJECTS_RUNTIME.md).
