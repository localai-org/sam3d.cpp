# SAM 3D Objects parity fixtures

`ss_latent_synthetic.bin` is the deterministic F32 decoder input generated
with NumPy seed `20260831` and scale `3.0` by the credited native port's
`gen_ss_latent.py` script.

`ss_occ_f32.samt` was independently captured from Meta's upstream
`SparseStructureDecoder` algebra at commit
`f91db411c50efee93d8db7aeb323885650f6f722`, using the locally verified
`ss_decoder.ckpt` whose SHA-256 is
`6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340`.
The reference ran in F32 on PyTorch CPU. The singleton output channel was
removed from the SAMT shape while preserving its contiguous payload.

The original fixture copied from `Asher-1/sam-3d-objects-ggml` was not used as
the parity oracle because it was produced by a different checkpoint. That
project remains credited for the synthetic input and initial GGML graph.
