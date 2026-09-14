// DINOv2 ViT-L/14 with register tokens (the condition embedder's RGB / mask
// backbones). Maps the checkpoint's Dino wrapper:
//   resize -> normalize -> patch_embed (14x14 conv, stride 14)
//   -> [cls | patches] + pos_embed -> insert 4 registers -> 24 blocks
//   -> final LayerNorm -> drop registers -> tokens (cls + patches)
// Tokens follow the repo convention: (C, N) in ggml ne[] order (ne[0] = C).
#include "dino_graph.hpp"

#include "graph_builder.hpp"
#include "common.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace sam3d {

// Host-side patch gather table: t[p * kdim + k] = flat index into canonical
// (H, W, C) image data for kernel element k of patch p. The p-major layout
// makes get_rows' (1, M) output reshape directly to (kdim, n_patch) with the
// contraction dim (k) on ne[0] for the patch GEMM.
// k = c * (P*P) + kh * P + kw matches the torch weight (OC, C, P, P) flatten.
static std::vector<int32_t> make_patch_table(int H, int W, int C, int P) {
    const int G = H / P;                      // patches per side
    const int n_patch = G * G;
    const int kdim = C * P * P;
    std::vector<int32_t> t((size_t)kdim * n_patch);
    for (int pi = 0; pi < G; pi++) {
        for (int pj = 0; pj < G; pj++) {
            const int p = pi * G + pj;
            for (int c = 0; c < C; c++) {
                for (int kh = 0; kh < P; kh++) {
                    for (int kw = 0; kw < P; kw++) {
                        const int k = c * P * P + kh * P + kw;
                        const int h = pi * P + kh, w = pj * P + kw;
                        t[(size_t)p * kdim + k] =
                            (h * W + w) * C + c;
                    }
                }
            }
        }
    }
    return t;
}

// view a GGUF param as F32 (small params may be stored F16 in f16 files)
static ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) t = ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// Quantized projection weights are consumed directly by ggml MUL_MAT. Casting
// them first inserts a dequantization copy in every graph evaluation and, for
// K-quants, bypasses the backend's optimized quantized GEMM implementation.
static ggml_tensor* matmul_weight(ggml_context* ctx, ggml_tensor* t) {
    return ggml_is_quantized(t->type) ? t : as_f32(ctx, t);
}


// debug: returns cls_token (C, 1) — lets the CLI verify the F16->F32 cast path
static ggml_tensor* pos_probe(ggml_context* ctx, const GGUFModel* m,
                              const std::string& prefix) {
    ggml_tensor* cls = m->get(prefix + ".backbone.cls_token");
    if (cls->type != GGML_TYPE_F32) cls = ggml_cast(ctx, cls, GGML_TYPE_F32);
    return ggml_reshape_2d(ctx, cls, ggml_nelements(cls), 1);
}

ggml_tensor* DinoGraph::build(ggml_tensor* img) {
    ggml_context* ctx = g->ctx();
    if (debug_stage == "img_echo") return img;
    const int64_t C = m->i32(prefix + ".embed_dim", 1024);
    const int64_t depth = m->i32(prefix + ".depth", 24);
    const int64_t n_heads = m->i32(prefix + ".num_heads", 16);
    const int64_t n_reg = m->i32(prefix + ".reg_tokens", 4);
    const int P = 14;
    // img is HWC in memory (channel fastest): ggml ne = [3, W, H]
    const int64_t H = img->ne[2], W = img->ne[1];
    GGML_ASSERT(img->ne[0] == 3);
    GGML_ASSERT(H % P == 0 && W % P == 0);
    const int64_t G = W / P, Gv = H / P;
    GGML_ASSERT(G == Gv && G == 37 && "expect 518x518 input (pos_embed must match)");
    const int64_t n_patch = G * Gv;
    const int64_t N = 1 + n_reg + n_patch;    // cls + registers + patches
    const int64_t kdim = 3 * P * P;
    const bool manual_attn = getenv("SAM3D_MANUAL_ATTN") != nullptr;

    // Dino wrapper normalization (normalize_images=true): per-channel
    // (x - mean) / std, broadcast over the channel dim of (W, H, 3). The
    // pipeline hands us raw [0,1] images; the official module normalizes
    // internally, so the constant must live in the graph.
    auto f32_buf = [](std::initializer_list<float> v) {
        auto p = std::make_shared<std::vector<int32_t>>(v.size());
        memcpy(p->data(), v.begin(), v.size() * sizeof(float));
        return p;
    };
    ggml_tensor* mean_t = g->input_f32("dino_mean", {3, 1, 1});
    ggml_tensor* istd_t = g->input_f32("dino_istd", {3, 1, 1});
    inputs.push_back(mean_t);
    table_data.push_back(f32_buf({0.485f, 0.456f, 0.406f}));
    inputs.push_back(istd_t);
    table_data.push_back(f32_buf({1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f}));
    img = ggml_mul(ctx, ggml_sub(ctx, img, mean_t), istd_t);
    if (debug_stage == "post_norm") return img;

    // patch gather table as a graph input (host data uploaded before compute)
    auto table = std::make_shared<std::vector<int32_t>>(
        make_patch_table((int)H, (int)W, 3, P));
    ggml_tensor* tids = g->input_i32("patch_table", {kdim * n_patch});
    inputs.push_back(tids);
    table_data.push_back(table);

    // patch embed: view the 4D stored conv weight (ne {P, P, 3, OC}) as a
    // (3*P*P, OC) matrix; kernel flat order k = c*P*P + kh*P + kw matches
    // the table rows, so patchify + one GEMM == the 14x14 stride-14 conv.
    ggml_tensor* w = as_f32(ctx, m->get(prefix + ".backbone.patch_embed.proj.weight"));
    ggml_tensor* wb = as_f32(ctx, m->get(prefix + ".backbone.patch_embed.proj.bias"));
    GGML_ASSERT(w);
    // row stride must follow the weight dtype (F16 in f16 GGUF files):
    // sizeof(float) here would make half the output channels read past the
    // tensor into neighbouring weights (NaN/garbage patches)
    ggml_tensor* w2d = ggml_view_2d(ctx, w, kdim, C, kdim * sizeof(float), 0);
    ggml_tensor* img1 = ggml_view_2d(ctx, img, 1, ggml_nelements(img),
                                     sizeof(float), 0);
    ggml_tensor* patches = ggml_get_rows(ctx, img1, tids);   // (1, kdim*n_patch)
    patches = ggml_reshape_2d(ctx, patches, kdim, n_patch);
    if (debug_stage == "patches_raw") return patches;
    ggml_tensor* x = gb_linear(ctx, w2d, wb, patches);       // (C, n_patch)
    if (debug_stage == "post_patch") return x;
    if (debug_stage == "pos_raw") return pos_probe(ctx, m, prefix);

    // DINOv2-reg ordering: [cls | patches] + pos_embed first, then the
    // register tokens are inserted after cls (they carry no positional info).
    ggml_tensor* cls = as_f32(ctx, m->get(prefix + ".backbone.cls_token"));
    ggml_tensor* reg = as_f32(ctx, m->get(prefix + ".backbone.register_tokens"));
    ggml_tensor* xfull = ggml_concat(ctx, ggml_reshape_2d(ctx, cls, C, 1),
                                     x, 1);                  // (C, 1+n_patch)
    const int64_t N0 = 1 + n_patch;
    ggml_tensor* pos = as_f32(ctx, m->get(prefix + ".backbone.pos_embed"));
    GGML_ASSERT(pos->ne[0] == C && pos->ne[1] == N0);
    xfull = ggml_add(ctx, xfull, pos);
    if (debug_stage == "post_pos") return xfull;
    ggml_tensor* cls_view =
        ggml_view_2d(ctx, xfull, C, 1, C * sizeof(float), 0);
    ggml_tensor* pat_view =
        ggml_view_2d(ctx, xfull, C, n_patch, C * sizeof(float), C * sizeof(float));
    xfull = ggml_concat(ctx, cls_view, ggml_reshape_2d(ctx, reg, C, n_reg), 1);
    xfull = ggml_concat(ctx, xfull, pat_view, 1);            // (C, N)
    if (debug_stage == "post_embed") return xfull;

    // transformer blocks: x += ls1 * attn(norm1(x)); x += ls2 * mlp(norm2(x))
    const float scale = 1.0f / sqrtf((float)(C / n_heads));
    const int64_t D = C / n_heads;
    for (int i = 0; i < (int)depth; i++) {
        const std::string b = prefix + ".backbone.blocks." + std::to_string(i);
        ggml_tensor* h = gb_layer_norm(ctx, xfull, m->get(b + ".norm1.weight"),
                                       m->get(b + ".norm1.bias"), 1e-6f);
        if (debug_stage == "b0_norm1" && i == 0) return h;
        // fused qkv weight (3C, C) -> three (C, C) row-slice GEMMs so q/k/v
        // come out as contiguous (C, N) tensors (slicing the fused GEMM
        // output would leave non-contiguous token strides)
        ggml_tensor* wqkv = matmul_weight(ctx, m->get(b + ".attn.qkv.weight"));
        ggml_tensor* bqkv = as_f32(ctx, m->get(b + ".attn.qkv.bias"));
        const int64_t Cw = wqkv->ne[0];
        ggml_tensor* bq = ggml_view_1d(ctx, bqkv, C, 0);
        ggml_tensor* bk = ggml_view_1d(ctx, bqkv, C, C * sizeof(float));
        ggml_tensor* bv = ggml_view_1d(ctx, bqkv, C, 2 * C * sizeof(float));
        ggml_tensor* q = gb_linear(ctx,
            ggml_view_2d(ctx, wqkv, Cw, C, wqkv->nb[1], 0), bq, h);
        ggml_tensor* k = gb_linear(ctx,
            ggml_view_2d(ctx, wqkv, Cw, C, wqkv->nb[1], C * wqkv->nb[1]), bk, h);
        ggml_tensor* v = gb_linear(ctx,
            ggml_view_2d(ctx, wqkv, Cw, C, wqkv->nb[1], 2 * C * wqkv->nb[1]), bv, h);
        if (debug_stage == "b0_q" && i == 0) return q;
        // to heads: (C, N) -> (D, H, N) -> (D, N, H)
        auto to_heads = [&](ggml_tensor* t) {
            ggml_tensor* r = ggml_reshape_4d(ctx, t, D, n_heads, N, 1);
            return ggml_permute(ctx, r, 0, 2, 1, 3);          // (D, N, H)
        };
        q = to_heads(q);
        k = to_heads(k);
        v = to_heads(v);

        ggml_tensor* att = nullptr;
        if (manual_attn) {
            // exact fp32 attention (debug path): softmax(Q^T K / sqrt(D)) V
            ggml_tensor* sc = ggml_mul_mat(ctx, k, q);        // (N, N, H)
            sc = ggml_scale(ctx, sc, scale);
            if (debug_stage == "b0_scores" && i == 0) return sc;
            sc = ggml_soft_max(ctx, sc);                      // over ne[0]=keys
            if (debug_stage == "b0_probs" && i == 0) return sc;
            // out[d,q,h] = sum_s v[d,s,h]*p[s,q,h]: contract over keys with a
            // materialized transposed v (mul_mat rejects TRANSPOSE as src0)
            ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, v));
            att = ggml_mul_mat(ctx, vt, sc);                  // (D, N, H)
        } else {
            // fused flash attention (F32 accumulators; k/v cast to F16 KV)
            att = gb_attention(ctx, q, k, v, scale, true);    // (D, N, H)
        }
        // merge heads: flash output is (D, H, N) with N fastest (token-major
        // head concat) -> straight reshape; manual output is (D, N, H) and
        // needs the head/token permute before the same reshape
        if (debug_stage == "b0_att" && i == 0) return att;
        if (manual_attn) {
            att = ggml_cont(ctx, ggml_permute(ctx, att, 0, 2, 1, 3));
        }
        att = ggml_reshape_2d(ctx, att, C, N);
        att = gb_linear(ctx, m->get(b + ".attn.proj.weight"),
                        m->get(b + ".attn.proj.bias"), att);
        att = ggml_mul(ctx, att, m->get(b + ".ls1.gamma"));   // LayerScale
        xfull = ggml_add(ctx, xfull, att);

        h = gb_layer_norm(ctx, xfull, m->get(b + ".norm2.weight"),
                          m->get(b + ".norm2.bias"), 1e-6f);
        h = gb_ffn_gelu(ctx, h, m->get(b + ".mlp.fc1.weight"),
                        m->get(b + ".mlp.fc1.bias"),
                        m->get(b + ".mlp.fc2.weight"),
                        m->get(b + ".mlp.fc2.bias"), /*erf=*/true);  // (C, N)
        h = ggml_mul(ctx, h, m->get(b + ".ls2.gamma"));
        xfull = ggml_add(ctx, xfull, h);
        if (debug_stage == "block" + std::to_string(i)) return xfull;
    }

    // final LayerNorm, drop registers: (C, 1+n_reg+n_patch) -> (C, 1+n_patch)
    if (prenorm) {
        // torch prenorm_features: tokens = F.layer_norm(x_prenorm) over ALL
        // tokens (cls + registers + patches), affine-free, eps 1e-5; the
        // checkpoint's backbone.norm buffer is never applied (Identity).
        return ggml_cont(ctx, ggml_norm(ctx, xfull, 1e-5f));
    }
    xfull = gb_layer_norm(ctx, xfull, m->get(prefix + ".backbone.norm.weight"),
                          m->get(prefix + ".backbone.norm.bias"), 1e-6f);
    ggml_tensor* cls_out =
        ggml_view_2d(ctx, xfull, C, 1, C * sizeof(float), 0);
    ggml_tensor* pat_out = ggml_view_2d(ctx, xfull, C, n_patch, C * sizeof(float),
                                        (1 + n_reg) * C * sizeof(float));
    return ggml_concat(ctx, cls_out, pat_out, 1);       // (C, 1 + n_patch)
}

}  // namespace sam3d
