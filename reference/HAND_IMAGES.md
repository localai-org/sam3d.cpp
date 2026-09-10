# Learned hand-image branches

This is a component composition for Body refinement, not the completed full
estimator. The input is the official dancing RGB image, camera and an explicit
hand ROI from the verified original crop fixture. The left case mirrors the
full image and uses the already mirrored ROI; camera intrinsics are unchanged,
as upstream requires. Native and original pipelines each compute their own
padding-0.9 crop, normalized pixels, camera conditions, DINOv3 features,
no-mask embedding, hand decoder, MHR/camera feedback and shared box heads.

The initial ROI is deliberately supplied to both pipelines. This does **not**
test native body-derived hand-box selection, output unmirroring, validity
decisions, body reprompt or the final IK/geometry merge. Those are separate
remaining full-refinement gates. No native hand-image public capability or
hand GGUF companion is advertised yet; hand decoder state is inline diagnostic
state, and the backbone/MHR are loaded from verified GGUFs.

## Original and native contracts

`capture_body_flow.py --hand-branch --image-pipeline` invokes unchanged original
`forward_pose_branch` with only the hand batch selected. That method executes
more than `forward_decoder_hand`: it adds the shared no-mask embedding **before**
camera conditioning and runs the shared box/classification heads **after** the
six decoder layers. The native `body_from_rgb` hand mode now includes both.
Independent dense hand positional encoding and sparse shared prompt encoding
remain distinct. The camera head retains the original hand scale factor 10.

The reference uses the same reviewed, pinned original source, safe trained
state, restricted offline container and streamed backbone as the Body capture.
All selected trained tensors are assigned with exact byte readback; every
original final field repeats exactly and is unchanged by observation. Original
CPU neural inference still uses the original CUDA ray-grid calculation, so
this capture is not an optimized CPU-only performance baseline.

The new diagnostic input format is `S3DHRG01`: existing flow dimensions/flags,
camera scale, 54 hand indices, 145 non-hand mask indices, RGB dimensions,
explicit box/intrinsics/pixels, prompts and sorted trained parameter arrays.
The native runner requires a backbone GGUF for this format. It checks all
529 boundary tensors and 117 independent final fields (646 comparisons),
including both shared heads. The normal tests ensure hand-image mode cannot
silently omit those heads or accept the non-hand pipeline contract.

## Current numerical evidence

Right-hand Vulkan passes **590/646** under the unchanged generic diagnostic
limits: 27 backbone, 23 decoder-operation and six other intermediate checks
fail. All **117 independent final fields pass**, including all six meshes and
projections. Last-layer vertex max error is `8.642673492431641e-7 m`; last-layer
vertex-projection max error is `0.00067138671875 px`. These passing finals do
not waive intermediate failures or establish full refinement parity.

Right-hand Debug ASan/UBSan CPU passes **519/646**: 30 backbone, 75 decoder,
20 other intermediate and two independent final checks fail. The two final
failures are layers 3 and 5 vertex projections, each `0.001220703125 px` versus
the unchanged `0.001 px` limit. CPU final acceptance is therefore also open.
Its report SHA-256 is
`dfb0417e1bb7e0a82b32601c901b3de194b4ed468b207c608319768893618ad6`.

The original mirrored left-hand CUDA capture is repeatable and observer-neutral.
Native Vulkan passes **506/646**: 27 backbone, 80 decoder, 27 other intermediate
and six final vertex-projection checks fail. All six layers' vertex projections
miss the pixel limit; the largest error is `0.00274658203125 px` at layer 4.
The original left-hand CPU capture has also completed; its cross-backend and
native CPU comparisons remain to be evaluated.

An optimized right-hand CPU run was interrupted by a global OOM killing its
invoking agent. Its saved raw output initially lacked verified clean-exit/report
evidence. A new serialized [bounded run](MEMORY_SAFETY.md) completed native
inference cleanly under ASan/UBSan; comparison still fails at **519/646**.
Its raw output is byte-identical to the interrupted artifact, and its complete
native tensors are byte-identical to the original Debug CPU run. Optimization
therefore changes neither those tensors nor the retained numerical failures.
The new report SHA-256 is
`1c8b1cebfbe641285de83130f2134816362d5ddde43421a7f86f43cd1b5a7362`.
The whole comparison took 172.43 seconds and peaked at 5,371,035,648 cgroup
bytes, with zero OOM events. This is a capped verification run, not a timing
baseline for performance parity.

Original CPU versus CUDA passes **538/646** on byte-identical inputs and state:
32 backbone, 63 decoder-operation and 13 other intermediate checks fail, but
all 117 independent final fields pass. This establishes reference variability;
it is not a replacement tolerance policy. The earlier full-body backbone
policy cannot be reused for these different image crops. No acceptance limits
have been changed for this experiment.

Right-hand evidence (SHA-256):

- Original CUDA manifest: `64e60c2a6ff190d647a25822867b25542b52fc1f6bfc74c40c28090b6a574610`.
- Original CPU manifest: `47b12c940addc5f95ebee0413e89e726fc4bca35addb3a47409fe09ec784e127`.
- Original cross-backend control: `5e2732b97e1f729060dbd0a68f0c2b548594a5fd9adecc7a5a3426cb43fbca34`.
- Verified Vulkan report: `8a50b200e826442783f73424037a88ad4c405533ea965011d963359a2f8c794e`.
- Verified Vulkan tensors: `32a4b45ef5aa0a50814d7a11ffe9a16d99f3f673510851a00ccdb90e824356a8`.

The finalized native runner hashes itself, its rotation comparison helper,
backend module, executable and supplied GGUF/parameter files before and after
execution. Its verified Vulkan repeat produces identical native tensor bytes
to the first diagnostic run. Use `body-hand-image-right-native-vulkan-verified`
as the reproducible report; the initial run overlapped a comparison-runner edit
and is retained only as a repeat-output diagnostic, not source-provenance proof.

## Reproduction

Inside the reviewed offline reference container, with the repository mounted
read-only at `/work` and an empty fixture directory writable at `/output`:

```sh
python /work/reference/capture_body_flow.py \
  --model /work/generated/models/mhr-public/assets/mhr_model.pt \
  --upstream /work/reference/upstream/sam-3d-body \
  --roma-wheel /work/generated/wheels/roma-1.6.1-py3-none-any.whl \
  --output /output --device cuda --full-width --decoder-operations \
  --trained-extraction /work/generated/extraction/body \
  --trained-config /work/generated/models/sam-3d-body-dinov3/model_config.yaml \
  --hand-branch --image-pipeline --dino-upstream /work/reference/upstream/dinov3 \
  --hand-crop-reference /work/generated/fixtures/body-hand-crop-upstream \
  --hand-side right
```

Use `--device cpu` for the original neural CPU control and `--hand-side left`
for the mirrored hand input, each in a separate empty output directory. Compare
native inference with `scripts/run_body_flow.py`, supplying the capture as
`--reference` and the checked Body DINOv3 file as `--backbone-gguf`. Use the
ASan/UBSan CPU build or strict-F32 UBSan Vulkan build, as in the Body evidence.
The comparison returns a failure exit status while any required check fails.
