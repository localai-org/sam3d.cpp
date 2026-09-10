# Objects condition fusion

This milestone implements the original `EmbedderFuser` **after its encoders**.
It accepts supplied `[B,N,D]` embedding arrays and explicit synthetic/trained
parameters; it does not yet run the image/pointmap encoders inside the fuser or
establish learned-model conditioning parity.

## Implemented contract

`src/objects_fuser.cpp` uses GGML F32 operations for:

- Per-encoder shared projection networks: optional LayerNorm (epsilon 1e-5),
  bias-free W1/W3, SiLU gating and W2. Hidden width uses the original two-thirds
  reduction followed by rounding up to a multiple of 256, not simply `4*D`.
- Ordered modalities and first-encounter positional groups, including repeated
  groups and no-position inputs. Learned and random position layouts have the
  same flattened values but different original tensor ranks.
- Deterministic forced drops **after** projection and positional addition.
  Training-only random modality dropout is not applied during eval.
- Token-axis concatenation, or channel-axis concatenation followed by the
  original compression projector. The latter requires compatible widths and
  equal token counts. Invalid projected/compression width combinations are
  rejected before constructing a GGML graph.

The caller's embedding arrays are not modified. The graph shares each encoder's
projection weights across its modalities. Optional diagnostics and output-only
execution must produce exactly equal final output. The implementation currently
uploads parameters per invocation; it is not the final persistent model session.

## Reference provenance

The numerical oracle is the unchanged original Meta `EmbedderFuser` and
`FeedForward` class definitions at Objects revision
`f91db411c50efee93d8db7aeb323885650f6f722`, not rewritten PyTorch expressions.
AST extraction avoids unrelated model/package imports. Source SHA-256s:

- `model/backbone/dit/embedder/embedder_fuser.py`:
  `a4d2ef27bddbff542550a76ad5a46facf0a653bc8dbc71fa6e92676c805d3c29`.
- `model/layers/llama3/ff.py`:
  `f3d6b9e6b60e71e00db34f9f3d6bbbf06e67e538e64a5ad796bf923af2c6e05d`.

The test boundary substitutes clone-only encoders for supplied embeddings.
Cloning is important: upstream adds positions in place, whereas real encoder
outputs are freshly computed on each call. The first harness used identities;
the unprojected positional case then failed the three-run repeatability check
because it accumulated positions into the reused input. No native comparison
was accepted from that capture. Clone-only boundaries and immediate diagnostic
copies restore correct test ownership without changing the original fuser.
This deliberately isolated boundary is not whole-encoder or raw-image parity.

Captures run in the reviewed, no-network reference container using PyTorch 2.7,
one CPU thread, deterministic algorithms and TF32 disabled. Separate CPU and
CUDA references each require three identical uninstrumented runs and an exactly
matching observed run. Hooks and TorchDispatch record actual norm, linear, SiLU,
gated-product, position, drop and concatenation tensors. Final outputs are also
saved from the uninstrumented baseline; native output-only execution is compared
against them independently.

## Tests and reproduction

Seven cases cover batch sizes one/two, shared/different-width encoders, repeated
position groups, absent positions, random/learned positions, pre-norm enabled/
disabled, forced drops, projection disabled and compression. The seventh uses
768/1024-channel inputs with 1024 tokens each and the original projection
multiplier 4. These dimensions test full-width arithmetic but do not assert the
inaccessible checkpoint's exact configuration.

Run the reference commands **inside the isolated container**, with only the
output directory writable:

```sh
python reference/capture_objects_fuser.py \
  --upstream reference/upstream/sam-3d-objects \
  --output generated/fixtures/objects-fuser-cpu --device cpu
python reference/capture_objects_fuser.py \
  --upstream reference/upstream/sam-3d-objects \
  --output generated/fixtures/objects-fuser-cuda --device cuda
```

On the host, after building:

```sh
python scripts/run_objects_fuser.py \
  --reference generated/fixtures/objects-fuser-cpu \
  --output generated/fixtures/objects-fuser-native-cpu \
  --runner build/debug/bin/sam3d-objects-fuser-capture \
  --module build/debug/bin/libggml-cpu.so
python scripts/run_objects_fuser.py \
  --reference generated/fixtures/objects-fuser-cuda \
  --output generated/fixtures/objects-fuser-native-vulkan \
  --runner build/vulkan-ubsan/bin/sam3d-objects-fuser-capture \
  --module build/vulkan-ubsan/bin/libggml-vulkan.so --backend Vulkan \
  --device 0 --description 'NVIDIA GeForce RTX 5070 Ti'
```

Select the actual intended Vulkan ICD/device on the test system. The runner
sets strict-F32 Vulkan flags as documented in the earlier neural milestones.
CPU runs with ASan/UBSan/LSan; NVIDIA uses the separately documented UBSan build
because of the diagnosed ASan/ICD initialization conflict.

All **140 checks per backend pass**: 133 operation tensors plus seven independent
final outputs. Input echoes require exact equality; all values must be finite;
other tensors require maximum absolute error <=1e-4 and relative L2 <=2e-5.
Limits were fixed before comparison and unchanged. Worst CPU errors are
7.630e-6 absolute / 5.255e-7 relative L2; Vulkan errors are 1.956e-5 / 1.370e-6.

Report SHA-256s:

- CPU: `5b96cf34f319adfb1577743fe42377b4cf614c47221ff999c6f3a9de83cc85f2`.
- Vulkan: `adfe940609edb2b54df22d08eca297b81ef3dcab65947b7f140ab1c110550f8a`.
- Small fixture: `7325ac82d1f8fd9c6c3b4d4a57ce4d260450d0545a322cec5be43fa6a26406e4`.

Normal CTest includes all 111 small-case boundaries, output-only equality,
caller input ownership, forced-drop checks and malformed shape/parameter/data
rejections. Python tests validate provenance, finite/error rules, and the bounded
capture-reader extension needed for the 33-tensor first case. The existing
32-tensor default remains unchanged for other component readers.

Neither timing these synthetic tests nor their numerical success establishes
full reconstruction, browser QA or performance parity.
