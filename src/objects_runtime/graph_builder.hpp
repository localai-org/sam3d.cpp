// Shared graph building blocks for the SAM 3D ggml graphs.
//
// Conventions:
//  - token tensors are shape (C, N, B) in ggml ne[] order, i.e. ne[0]=C is
//    the fastest dim; a (B, N, C) torch tensor maps to (C, N, B).
//  - weights come from GGUF tensors: linear weight ne = {in, out} so that
//    ggml_mul_mat(w, x) yields (out, N).
#pragma once

#include <string>
#include <vector>

#include "ggml.h"

namespace sam3d {

// simple ggml context wrapper with growable buffer
class GraphContext {
  public:
    explicit GraphContext(size_t reserve = 32u * 1024 * 1024);
    ~GraphContext();

    ggml_context* ctx() const { return ctx_; }

    ggml_tensor* input_f32(const std::string& name, std::vector<int64_t> ne);
    ggml_tensor* input_i32(const std::string& name, std::vector<int64_t> ne);
    ggml_tensor* input_f16(const std::string& name, std::vector<int64_t> ne);

  private:
    ggml_context* ctx_ = nullptr;
    std::vector<uint8_t> buf_;
};

// x: (in, N) -> mul_mat(w, x) + b
ggml_tensor* gb_linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x);

// layer_norm over ne[0] with optional affine weights (eps from call site)
ggml_tensor* gb_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                           float eps);

// RMS normalize the last (head) dim of (D, N, H) tensors per head
ggml_tensor* gb_rms_norm_head(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma,
                              float head_dim);

// multi-head self attention with fused qkv weight (3C, C): x (C, N, B)
// qk_rms_norm applies per-head RMS normalization with gamma (H, D) weights.
// Returns (C, N, B) before out projection.
ggml_tensor* gb_self_attention_core(ggml_context* ctx, ggml_tensor* qkv, int n_heads,
                                    ggml_tensor* q_gamma, ggml_tensor* k_gamma);

// cross attention: x (C, N, B), context (Cctx, Nkv, B); to_q (C, C),
// to_kv (2C, Cctx). qk_rms handled by caller via gb_self_attention_core-like
// ops; this helper expects q and kv already projected and reshaped.
ggml_tensor* gb_attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                          float scale, bool use_flash);

// FFN: Linear -> GELU -> Linear (default tanh approximation; erf = exact)
ggml_tensor* gb_ffn_gelu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w0, ggml_tensor* b0,
                         ggml_tensor* w2, ggml_tensor* b2, bool erf = false);

// llama3 SwiGLU FFN: w2(silu(w1 x) * w3 x)
ggml_tensor* gb_ffn_swiglu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* w2,
                           ggml_tensor* w3);

// timestep sinusoidal embedding -> MLP; t is a (1, 1) F32 tensor.
// freq embedding table (half_dim, 2) is precomputed as cos||sin columns:
// out[i] = cos(t * freq[i]) then sin. Returns mlp output (C, 1).
ggml_tensor* gb_timestep_embedding(ggml_context* ctx, ggml_tensor* t, int out_dim,
                                   ggml_tensor* freq_table, ggml_tensor* w0, ggml_tensor* b0,
                                   ggml_tensor* w2, ggml_tensor* b2);

// AbsolutePositionEmbedder for coords (N, 3) F32: per-axis sin/cos with
// freqs[i] = 1/10000^(i/freq_dim), output (channels, N) with zero padding to
// `channels`. freq_table: (freq_dim) F32 precomputed on host.
ggml_tensor* gb_absolute_pos_embed(ggml_context* ctx, ggml_tensor* coords, int channels,
                                   ggml_tensor* freq_table);

// SiLU + Linear modulation used by adaLN: (6C, C) @ silu(t)
ggml_tensor* gb_adaln_modulation(ggml_context* ctx, ggml_tensor* t_emb, ggml_tensor* w,
                                 ggml_tensor* b);

// helper: reshape (3C, N, B) fused qkv -> view + transpose chain producing
// q/k/v as (D, N, B*H) with head dim merged into batch for flash attention
void gb_split_qkv(ggml_context* ctx, ggml_tensor* qkv, int n_heads,
                  ggml_tensor** q, ggml_tensor** k, ggml_tensor** v);

// Split a fused (2C, N, B) K/V projection into contiguous (C, N, B) tensors.
// The copies are required because attention head reshapes cannot consume a
// strided row view.  They still avoid running two GEMMs over the same input.
void gb_split_kv(ggml_context* ctx, ggml_tensor* kv,
                 ggml_tensor** k, ggml_tensor** v);

}  // namespace sam3d
