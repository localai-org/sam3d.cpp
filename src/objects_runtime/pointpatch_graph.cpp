// PointPatchEmbed graph (cemb.emb2) - see pointpatch_graph.hpp for layout.
#include "pointpatch_graph.hpp"

#include "graph_builder.hpp"
#include "common.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace sam3d {

// torch F.interpolate(mode="nearest", align_corners=False) for downscale:
//   src = clamp(floor(scale * dst), 0, src_size - 1)   (no half-pixel offset)
static void nearest_index_table(std::vector<int32_t>& t, int src_size,
                                int dst_size) {
    const float scale = (float)src_size / (float)dst_size;
    t.resize(dst_size);
    for (int d = 0; d < dst_size; d++) {
        int i = (int)floorf(scale * (float)d);
        if (i < 0) i = 0;
        if (i > src_size - 1) i = src_size - 1;
        t[d] = i;
    }
}

static ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) t = ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

static ggml_tensor* matmul_weight(ggml_context* ctx, ggml_tensor* t) {
    return ggml_is_quantized(t->type) ? t : as_f32(ctx, t);
}

ggml_tensor* PointPatchGraph::build(ggml_tensor* pm) {
    ggml_context* ctx = g->ctx();
    const int64_t D = m->i32(prefix + ".embed_dim", 512);
    const int64_t in_size = m->i32(prefix + ".input_size", 256);
    const int64_t ps = m->i32(prefix + ".patch_size", 8);
    const int64_t n_heads = m->i32(prefix + ".num_heads", 16);
    GGML_ASSERT(pm->ne[2] == 3);
    const int64_t src_w = pm->ne[0], src_h = pm->ne[1];
    const int64_t n_px = in_size * in_size;        // 65536
    const int64_t n_w = in_size / ps;              // 32 windows per side
    const int64_t n_win = n_w * n_w;               // 1024
    const int64_t n_tok = n_win * (1 + ps * ps);   // 66560 (65 per window)
    const int64_t Hd = D / n_heads;

    // host table 1: per-dst source index (within one channel plane)
    auto resize = std::make_shared<std::vector<int32_t>>();
    {
        std::vector<int32_t> hi, wi;
        nearest_index_table(hi, (int)src_h, (int)in_size);
        nearest_index_table(wi, (int)src_w, (int)in_size);
        resize->resize(n_px);
        for (int h2 = 0; h2 < in_size; h2++)
            for (int w2 = 0; w2 < in_size; w2++)
                (*resize)[(size_t)h2 * in_size + w2] = hi[h2] * src_w + wi[w2];
    }
    ggml_tensor* t_resize = g->input_i32("pp_resize_idx", {n_px});
    inputs.push_back(t_resize);
    table_data.push_back(resize);

    // per-channel gather: (1, n_px) each, then stack -> (3, n_px) with the
    // 3 channels as the contraction dim of the point_proj GEMM
    ggml_tensor* x = nullptr;
    for (int c = 0; c < 3; c++) {
        // (1, n) 2D view: get_rows needs row-shaped src0 (1 element per row)
        ggml_tensor* plane = ggml_view_2d(ctx, pm, 1, src_w * src_h,
                                          sizeof(float),
                                          c * src_w * src_h * sizeof(float));
        ggml_tensor* g = ggml_get_rows(ctx, plane, t_resize);  // (1, n_px)
        x = x ? ggml_concat(ctx, x, g, 0) : g;
    }
    // point_proj: Linear(3 -> D)
    x = gb_linear(ctx, as_f32(ctx, m->get(prefix + ".point_proj.weight")),
                  as_f32(ctx, m->get(prefix + ".point_proj.bias")), x);
    if (debug_stage == "post_proj") return x;

    // invalid-pixel replacement: x = x - (x - invalid_token) * invalid.
    // ggml binary ops broadcast only the SECOND operand (can_repeat(b, a)),
    // so the (512, 1) token must stay on the b side; negation via scale.
    // Host code must pre-zero NaNs so x stays finite.
    ggml_tensor* mask_v = g->input_f32("pp_valid", {1, n_px});
    ggml_tensor* mask_i = g->input_f32("pp_invalid", {1, n_px});
    inputs.push_back(mask_v);
    table_data.push_back(nullptr);  // f32 masks uploaded via mask_inputs
    inputs.push_back(mask_i);
    table_data.push_back(nullptr);
    mask_inputs.push_back(mask_v);
    mask_inputs.push_back(mask_i);
    ggml_tensor* inv_tok = ggml_reshape_2d(
        ctx, as_f32(ctx, m->get(prefix + ".invalid_xyz_token")), D, 1);
    ggml_tensor* d = ggml_mul(
        ctx, ggml_sub(ctx, x, inv_tok), mask_i);
    x = ggml_add(ctx, x, ggml_scale(ctx, d, -1.0f));

    // host table 2: window reorder + CLS insertion. New token sequence is
    // [win0: cls, kt0..63, win1: cls, ...]; kt of window (wi, wj) is source
    // row (wi*8+kh)*in_size + (wj*8+kw). CLS slots read the sentinel row
    // n_px (the cls token appended after the pixels).
    auto win_idx = std::make_shared<std::vector<int32_t>>(n_tok);
    for (int wi = 0; wi < n_w; wi++) {
        for (int wj = 0; wj < n_w; wj++) {
            const int win = wi * n_w + wj;
            (*win_idx)[(size_t)win * (1 + ps * ps)] = (int32_t)n_px;  // cls
            for (int kh = 0; kh < ps; kh++)
                for (int kw = 0; kw < ps; kw++)
                    (*win_idx)[(size_t)win * (1 + ps * ps) + 1 + kh * ps + kw] =
                        (int32_t)((wi * ps + kh) * in_size + (wj * ps + kw));
        }
    }
    ggml_tensor* t_win = g->input_i32("pp_win_idx", {n_tok});
    inputs.push_back(t_win);
    table_data.push_back(win_idx);

    // append the cls sentinel row, then gather rows into window order
    ggml_tensor* cls = ggml_reshape_2d(ctx, as_f32(ctx, m->get(prefix + ".cls_token")), D, 1);
    ggml_tensor* xc = ggml_concat(ctx, x, cls, 1);                // (D, n_px+1)
    ggml_tensor* toks = ggml_get_rows(ctx, xc, t_win);            // (D, n_tok)
    toks = ggml_reshape_3d(ctx, toks, D, 1 + ps * ps, n_win);     // (D, 65, W)
    // per-window positional embedding (same for every window): (D, 65, 1)
    ggml_tensor* pw = ggml_reshape_3d(
        ctx, as_f32(ctx, m->get(prefix + ".pos_embed_window")), D, 1 + ps * ps, 1);
    toks = ggml_add(ctx, toks, pw);
    if (debug_stage == "post_posw") return toks;

    // single transformer block (timm layout, erf GELU)
    const std::string b = prefix + ".blocks.0";
    ggml_tensor* h = gb_layer_norm(ctx, toks, m->get(b + ".norm1.weight"),
                                   m->get(b + ".norm1.bias"), 1e-6f);
    if (debug_stage == "b0_norm1") return h;
    ggml_tensor* wqkv = matmul_weight(ctx, m->get(b + ".attn.qkv.weight"));
    ggml_tensor* bqkv = as_f32(ctx, m->get(b + ".attn.qkv.bias"));
    ggml_tensor* q = nullptr, *k = nullptr, *v = nullptr;
    // wqkv is stored (3D, D): rows are the 3*D output dim. Slice rows
    // [i*D, (i+1)*D) as (D, D) 2D views for mul_mat (batch rides on x).
    ggml_tensor* wq = ggml_view_2d(ctx, wqkv, D, D, wqkv->nb[1], 0);
    ggml_tensor* wk = ggml_view_2d(ctx, wqkv, D, D, wqkv->nb[1],
                                   (size_t)D * wqkv->nb[1]);
    ggml_tensor* wv = ggml_view_2d(ctx, wqkv, D, D, wqkv->nb[1],
                                   2 * (size_t)D * wqkv->nb[1]);
    ggml_tensor* bq = ggml_view_1d(ctx, bqkv, D, 0);
    ggml_tensor* bk = ggml_view_1d(ctx, bqkv, D, D * sizeof(float));
    ggml_tensor* bv = ggml_view_1d(ctx, bqkv, D, 2 * D * sizeof(float));
    // h (D, 65, n_win): mul_mat broadcasts ne[2] of src1 across src0
    q = gb_linear(ctx, wq, bq, h);
    k = gb_linear(ctx, wk, bk, h);
    v = gb_linear(ctx, wv, bv, h);
    auto to_heads4 = [&](ggml_tensor* t) {
        ggml_tensor* r = ggml_reshape_4d(ctx, t, Hd, n_heads, 1 + ps * ps, n_win);
        return ggml_permute(ctx, r, 0, 2, 1, 3);  // (Hd, N, heads, win)
    };
    q = to_heads4(q);
    k = to_heads4(k);
    v = to_heads4(v);
    if (debug_stage == "b0_q") return q;
    const float scale = 1.0f / sqrtf((float)Hd);
    // manual attention: the CUDA flash kernels reject this shape (batch 1024
    // windows x hd 32), and with only 65 tokens per window the two-GEMM form
    // is cheap and exact in F32.
    auto attention = [&](ggml_tensor* qh, ggml_tensor* kh, ggml_tensor* vh) {
        ggml_tensor* sc = ggml_mul_mat(ctx, kh, qh);       // (N, N, H, B)
        sc = ggml_scale(ctx, sc, scale);
        if (debug_stage == "b0_scores") return sc;
        sc = ggml_soft_max(ctx, sc);
        if (debug_stage == "b0_probs") return sc;
        ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vh));
        return ggml_mul_mat(ctx, vt, sc);                  // (Hd, N, H, B)
    };
    ggml_tensor* att = attention(q, k, v);
    // manual output is (Hd, N, H, B) with (B, H, N, D) memory; permute the
    // head dim next to Hd so the memory becomes token-major (B, N, H, D)
    att = ggml_cont(ctx, ggml_permute(ctx, att, 0, 2, 1, 3));
    att = ggml_reshape_3d(ctx, att, D, 1 + ps * ps, n_win);
    if (debug_stage == "b0_att") return att;
    att = gb_linear(ctx, matmul_weight(ctx, m->get(b + ".attn.proj.weight")),
                    as_f32(ctx, m->get(b + ".attn.proj.bias")), att);
    toks = ggml_add(ctx, toks, att);

    h = gb_layer_norm(ctx, toks, m->get(b + ".norm2.weight"),
                      m->get(b + ".norm2.bias"), 1e-6f);
    h = gb_ffn_gelu(ctx, h, m->get(b + ".mlp.fc1.weight"),
                    m->get(b + ".mlp.fc1.bias"),
                    m->get(b + ".mlp.fc2.weight"),
                    m->get(b + ".mlp.fc2.bias"), /*erf=*/true);
    toks = ggml_add(ctx, toks, h);
    if (debug_stage == "post_block_full") return toks;

    // extract per-window CLS: batched get_rows indexes the ne[1] (token)
    // dimension per batch slice, and the CLS sits at token 0 of each window,
    // so src1 is an all-zero (1, n_win) table.
    auto cls_idx = std::make_shared<std::vector<int32_t>>(n_win, 0);
    ggml_tensor* t_cls = g->input_i32("pp_cls_idx", {1, n_win});
    inputs.push_back(t_cls);
    table_data.push_back(cls_idx);
    ggml_tensor* cls_out = ggml_get_rows(ctx, toks, t_cls);  // (512, 1, n_win)
    cls_out = ggml_reshape_2d(ctx, cls_out, D, n_win);
    if (debug_stage == "post_block") return cls_out;

    // + pos_embed_patch: the converter pre-permutes it to token-major
    // (H*W, D) rows (token = h*H_n + w, D fastest), matching the window
    // raster order, so a plain reshape + add is exact.
    ggml_tensor* pos = as_f32(ctx, m->get(prefix + ".pos_embed"));
    // Q8_0 keeps the converter's original [D,H,W] logical dimensions in
    // GGUF; F16/F32 are already [D,H*W]. Both have the same element order.
    GGML_ASSERT(ggml_nelements(pos) == D * n_win);
    pos = ggml_reshape_2d(ctx, pos, D, n_win);
    return ggml_add(ctx, cls_out, pos);
}

}  // namespace sam3d
