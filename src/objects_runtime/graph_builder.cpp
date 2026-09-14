#include "graph_builder.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sam3d {

GraphContext::GraphContext(size_t reserve) : buf_(reserve) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_.size(),
        /*.mem_buffer =*/ buf_.data(),
        /*.no_alloc   =*/ true,
    };
    ctx_ = ggml_init(params);
}

GraphContext::~GraphContext() {
    if (ctx_) ggml_free(ctx_);
}

ggml_tensor* GraphContext::input_f32(const std::string& name, std::vector<int64_t> ne) {
    ggml_tensor* t = ggml_new_tensor(ctx_, GGML_TYPE_F32, (int)ne.size(), ne.data());
    ggml_set_name(t, name.c_str());
    ggml_set_input(t);
    return t;
}

ggml_tensor* GraphContext::input_i32(const std::string& name, std::vector<int64_t> ne) {
    ggml_tensor* t = ggml_new_tensor(ctx_, GGML_TYPE_I32, (int)ne.size(), ne.data());
    ggml_set_name(t, name.c_str());
    ggml_set_input(t);
    return t;
}

ggml_tensor* GraphContext::input_f16(const std::string& name, std::vector<int64_t> ne) {
    ggml_tensor* t = ggml_new_tensor(ctx_, GGML_TYPE_F16, (int)ne.size(), ne.data());
    ggml_set_name(t, name.c_str());
    ggml_set_input(t);
    return t;
}

ggml_tensor* gb_linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
    ggml_tensor* out = ggml_mul_mat(ctx, w, x);
    // Keep the default backend-selected accumulation path. The opt-in F32
    // setting is a graph-local parity diagnostic for quantized SLat models;
    // unlike disabling FP16 on the device, it does not perturb unrelated ops.
    if (getenv("SAM3D_E2E_F32_MATMUL") != nullptr) {
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    }
    if (b) out = ggml_add(ctx, out, b);
    return out;
}

ggml_tensor* gb_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                           float eps) {
    ggml_tensor* out = ggml_norm(ctx, x, eps);
    if (w) out = ggml_mul(ctx, out, w);
    if (b) out = ggml_add(ctx, out, b);
    return out;
}

ggml_tensor* gb_rms_norm_head(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma,
                              float head_dim) {
    // x: (D, N, H, B). torch MultiHeadRMSNorm is
    //   F.normalize(x, dim=-1) * gamma * sqrt(D)
    // which equals plain RMS normalization: x*sqrt(D)/||x|| == x/rms(x),
    // so the head_dim factor cancels - just rms-norm and apply gamma.
    ggml_tensor* normed = ggml_rms_norm(ctx, x, 1e-6f);  // x / rms(x)
    if (gamma) {
        // gamma ggml ne = {D, H} (torch (H, D)); broadcast to (D, 1, H, 1)
        ggml_tensor* g = ggml_reshape_4d(ctx, gamma, gamma->ne[0], 1, gamma->ne[1], 1);
        normed = ggml_mul(ctx, normed, g);
    }
    return normed;
}

void gb_split_qkv(ggml_context* ctx, ggml_tensor* qkv, int n_heads, ggml_tensor** q,
                  ggml_tensor** k, ggml_tensor** v) {
    const int64_t C = qkv->ne[0] / 3;
    const int64_t D = C / n_heads;
    const int64_t N = qkv->ne[1];
    const int64_t B = qkv->ne[2];
    const size_t cs = sizeof(float);
    // per-slice view (C, N, B): token stride C*4, batch stride qkv->nb[2].
    // The views preserve a final batch dimension when present. The row slice
    // remains strided because a QKV row is 3C-wide, so each slice is made
    // contiguous before the head reshape.
    auto view = [&](int idx) {
        // row-slice of the (3C, N) fused qkv: token stride is the FULL row
        // (qkv->nb[1] = 3C*4 bytes), not C*4. The slice is strided, so
        // materialize it (the downstream reshape_4d needs contiguous input).
        return ggml_cont(ctx, ggml_view_3d(ctx, qkv, C, N, B, qkv->nb[1],
                                           qkv->nb[2], (size_t)idx * C * cs));
    };
    ggml_tensor* qv = view(0);
    ggml_tensor* kv = view(1);
    ggml_tensor* vv = view(2);
    // (C, N, B) -> (D, H, N, B) -> (D, N, H, B)
    auto to_heads = [&](ggml_tensor* t) {
        ggml_tensor* r = ggml_reshape_4d(ctx, t, D, n_heads, N, B);
        return ggml_permute(ctx, r, 0, 2, 1, 3);  // (D, N, H, B)
    };
    *q = to_heads(qv);
    *k = to_heads(kv);
    *v = to_heads(vv);
}

void gb_split_kv(ggml_context* ctx, ggml_tensor* kv,
                 ggml_tensor** k, ggml_tensor** v) {
    const int64_t C = kv->ne[0] / 2;
    const int64_t N = kv->ne[1];
    const int64_t B = kv->ne[2];
    GGML_ASSERT(kv->ne[0] == 2 * C && B == 1);
    const size_t offset = (size_t)C * sizeof(float);
    *k = ggml_cont(ctx, ggml_view_3d(ctx, kv, C, N, B, kv->nb[1], kv->nb[2], 0));
    *v = ggml_cont(ctx, ggml_view_3d(ctx, kv, C, N, B, kv->nb[1], kv->nb[2], offset));
}

ggml_tensor* gb_attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                          float scale, bool use_flash) {
    // The optimized path stores K/V in F16 for flash attention.  Keep a
    // graph-level, all-F32 formulation for numerical diagnosis: it is the
    // direct QK^T -> softmax -> V expression used by the official model and
    // requires no backend-specific ggml source change.  It intentionally is
    // opt-in because materializing N x N scores is not the production path.
    if (getenv("SAM3D_MANUAL_ATTN") != nullptr) {
        ggml_tensor* scores = ggml_mul_mat(ctx, k, q);       // [K, Q, heads, batch]
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_soft_max(ctx, scores);                 // normalize over keys
        ggml_tensor* values_t = ggml_cont(ctx, ggml_transpose(ctx, v));
        ggml_tensor* output = ggml_mul_mat(ctx, values_t, scores);
        ggml_mul_mat_set_prec(output, GGML_PREC_F32);
        // [D, Q, heads, batch] is the layout consumed by the projection.
        return ggml_cont(ctx, ggml_permute(ctx, output, 0, 2, 1, 3));
    }
    GGML_ASSERT(use_flash);
    const bool strict = getenv("SAM3D_STRICT_ATTN") != nullptr;
    // Both supported GPU flash-attention backends accept F32 K/V. Preserve
    // them for the numeric gate; the normal throughput path keeps the F16
    // cache representation used by the existing implementation.
    ggml_tensor* kk = strict ? k : ggml_cast(ctx, k, GGML_TYPE_F16);
    ggml_tensor* vv = strict ? v : ggml_cast(ctx, v, GGML_TYPE_F16);
    ggml_tensor* out = ggml_flash_attn_ext(ctx, q, kk, vv, nullptr, scale, 0.0f, 0.0f);
    // This changes only the precision contract of the established flash
    // graph. It is an opt-in parity diagnostic, not the default speed path.
    if (strict) {
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    }
    return out;
}

ggml_tensor* gb_self_attention_core(ggml_context* ctx, ggml_tensor* qkv, int n_heads,
                                    ggml_tensor* q_gamma, ggml_tensor* k_gamma) {
    const int64_t C = qkv->ne[0] / 3;
    const int64_t D = C / n_heads;
    ggml_tensor *q, *k, *v;
    gb_split_qkv(ctx, qkv, n_heads, &q, &k, &v);
    if (q_gamma) {
        q = gb_rms_norm_head(ctx, q, q_gamma, (float)D);
        k = gb_rms_norm_head(ctx, k, k_gamma, (float)D);
    }
    const float scale = 1.0f / sqrtf((float)D);
    return gb_attention(ctx, q, k, v, scale, true);
}

ggml_tensor* gb_ffn_gelu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w0, ggml_tensor* b0,
                         ggml_tensor* w2, ggml_tensor* b2, bool erf) {
    ggml_tensor* h = gb_linear(ctx, w0, b0, x);
    // ggml_gelu is the tanh approximation (tdfy_dit FeedForwardNet);
    // timm-style blocks (DINOv2) use the exact erf GELU
    h = erf ? ggml_gelu_erf(ctx, h) : ggml_gelu(ctx, h);
    return gb_linear(ctx, w2, b2, h);
}

ggml_tensor* gb_ffn_swiglu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* w2,
                           ggml_tensor* w3) {
    ggml_tensor* a = ggml_silu(ctx, gb_linear(ctx, w1, nullptr, x));
    ggml_tensor* b = gb_linear(ctx, w3, nullptr, x);
    return gb_linear(ctx, w2, nullptr, ggml_mul(ctx, a, b));
}

ggml_tensor* gb_adaln_modulation(ggml_context* ctx, ggml_tensor* t_emb, ggml_tensor* w,
                                 ggml_tensor* b) {
    return gb_linear(ctx, w, b, ggml_silu(ctx, t_emb));
}

ggml_tensor* gb_timestep_embedding(ggml_context* ctx, ggml_tensor* t, int out_dim,
                                   ggml_tensor* freq_table, ggml_tensor* w0, ggml_tensor* b0,
                                   ggml_tensor* w2, ggml_tensor* b2) {
    // t: (1, 1) F32; freq_table: (half, 1) F32 with freqs (from host)
    // matches torch: cat([cos(t*f), sin(t*f)], dim=-1)
    ggml_tensor* tf = ggml_mul(ctx, ggml_repeat(ctx, t, freq_table), freq_table);
    ggml_tensor* c = ggml_cos(ctx, tf);
    ggml_tensor* s = ggml_sin(ctx, tf);
    ggml_tensor* emb = ggml_concat(ctx, c, s, 0);
    ggml_tensor* h = gb_linear(ctx, w0, b0, emb);
    h = ggml_silu(ctx, h);
    return gb_linear(ctx, w2, b2, h);
}

}  // namespace sam3d
