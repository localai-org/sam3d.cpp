// SLat Gaussian decoder graph - see gs_decoder_graph.hpp.
#include "gs_decoder_graph.hpp"

#include "graph_builder.hpp"
#include "common.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace sam3d {

std::vector<ggml_tensor*> GsDecoderGraph::build() {
    ggml_context* ctx = g->ctx();
    const int64_t C = m->i32(prefix + ".model_channels", 768);
    const int64_t n_blocks = m->i32(prefix + ".num_blocks", 12);
    const int64_t n_heads = m->i32(prefix + ".num_heads", 12);
    const int64_t Hd = C / n_heads;
    const int64_t in_ch = m->i32(prefix + ".latent_channels", 8);
    const int64_t out_ch = m->i32(prefix + ".out_channels", 448);
    const int64_t N = x->ne[1];
    GGML_ASSERT(x->ne[0] == in_ch);

    auto as_f32 = [&](ggml_tensor* t) {
        return t->type != GGML_TYPE_F32 ? ggml_cast(ctx, t, GGML_TYPE_F32) : t;
    };
    auto add_i32 = [&](const std::vector<int32_t>& v, const char* name,
                       std::initializer_list<int64_t> ne) {
        ggml_tensor* t = g->input_i32(name, ne);
        inputs.push_back(t);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(v));
        return t;
    };

    // ---- input layer + APE ----------------------------------------------
    ggml_tensor* h = gb_linear(ctx, as_f32(m->get(prefix + ".input_layer.weight")),
                               as_f32(m->get(prefix + ".input_layer.bias")), x);
    if (debug_stage == "input_layer") return {h};
    {
        ggml_tensor* ape = g->input_f32("ape", {C, N});
        inputs.push_back(ape);
        // upload path: f32 data reinterpreted as the shared i32 vector storage
        // (same pattern as SlatFlowGraph)
        table_data.push_back(std::make_shared<std::vector<int32_t>>(
            (const int32_t*)tb->ape.data(),
            (const int32_t*)(tb->ape.data() + tb->ape.size())));
        h = ggml_add(ctx, h, ape);
    }
    if (debug_stage == "ape") return {h};
    // the torso runs fp16 (convert_to_fp16) in torch; keep activations f32
    // here - the weights are stored f16 and every op computes in f32 anyway.

    // ---- transformer blocks (swin windowed attention) -------------------
    for (int i = 0; i < (int)n_blocks; i++) {
        const std::string b = prefix + ".blocks." + std::to_string(i);
        const GsWindowTables& wt = tb->shifts[i % 2];
        ggml_tensor* h_in = h;  // block input (for the same-run debug dump)

        ggml_tensor* bkt_idx = add_i32(wt.bkt_idx, "gs_bkt_idx", {N});
        ggml_tensor* bwd_idx = add_i32(wt.bwd_idx, "gs_bwd_idx", {N});

        // norm1 (no affine) -> gather to bucket order
        ggml_tensor* h1 = ggml_norm(ctx, h, 1e-6f);
        ggml_tensor* hb = ggml_get_rows(ctx, h1, bkt_idx);  // (C, N)

        // fused qkv GEMM, three contiguous row-slice segments
        ggml_tensor* wqkv = as_f32(m->get(b + ".attn.to_qkv.weight"));
        ggml_tensor* bqkv = as_f32(m->get(b + ".attn.to_qkv.bias"));
        const size_t qw = ggml_type_size(wqkv->type);
        ggml_tensor* wq = ggml_view_2d(ctx, wqkv, C, C, wqkv->nb[1], 0);
        ggml_tensor* wkk = ggml_view_2d(ctx, wqkv, C, C, wqkv->nb[1], (size_t)C * C * qw);
        ggml_tensor* wvv = ggml_view_2d(ctx, wqkv, C, C, wqkv->nb[1], 2 * (size_t)C * C * qw);
        ggml_tensor* bqq = ggml_view_1d(ctx, bqkv, C, 0);
        ggml_tensor* bkk = ggml_view_1d(ctx, bqkv, C, C * sizeof(float));
        ggml_tensor* bvv = ggml_view_1d(ctx, bqkv, C, 2 * C * sizeof(float));
        ggml_tensor* q = ggml_cont(ctx, gb_linear(ctx, wq, bqq, hb));
        ggml_tensor* k = ggml_cont(ctx, gb_linear(ctx, wkk, bkk, hb));
        ggml_tensor* v = ggml_cont(ctx, gb_linear(ctx, wvv, bvv, hb));
        if (i == 0 && debug_stage == "b0_qkv")
            return {ggml_get_rows(ctx, q, bwd_idx),
                    ggml_get_rows(ctx, k, bwd_idx),
                    ggml_get_rows(ctx, v, bwd_idx)};

        // heads: (C, N) memory (n, (h,d)) -> ne (Hd, H, N)
        auto to_heads = [&](ggml_tensor* t) {
            return ggml_reshape_3d(ctx, t, Hd, n_heads, N);
        };
        ggml_tensor* qh = to_heads(q);
        ggml_tensor* kh = to_heads(k);
        ggml_tensor* vh = to_heads(v);
        const float scale = 1.0f / sqrtf((float)Hd);

        // per-bucket batched manual attention over (D, len, H, n_win)
        ggml_tensor* acc = nullptr;
        for (const auto& bk : wt.buckets) {
            const int64_t off = bk.off, len = bk.len, nw = bk.n_win;
            // bucket rows are contiguous in the (.., N, ..) token dim
            auto view_bkt = [&](ggml_tensor* t_heads3) {
                // t_heads3 ne = (Hd, H, N): memory (N, H, Hd) - token stride
                // nb[2], head stride nb[1]. New logical axes (Hd, len, H, nw):
                //   d stride 4 | token stride nb[2] | head stride nb[1] |
                //   window stride len*nb[2] | bucket offset off*nb[2]
                return ggml_view_4d(ctx, t_heads3, Hd, len, n_heads, nw,
                                    t_heads3->nb[2], t_heads3->nb[1],
                                    len * t_heads3->nb[2], off * t_heads3->nb[2]);
            };
            ggml_tensor* qs = ggml_cont(ctx, view_bkt(qh));
            ggml_tensor* ks = ggml_cont(ctx, view_bkt(kh));
            ggml_tensor* vs = ggml_cont(ctx, view_bkt(vh));

            ggml_tensor* sc = ggml_mul_mat(ctx, ks, qs);  // (len, len, H, nw)
            sc = ggml_scale(ctx, sc, scale);
            sc = ggml_soft_max(ctx, sc);
            ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vs));  // (len, Hd, H, nw)
            ggml_tensor* ao = ggml_mul_mat(ctx, vt, sc);                // (Hd, len, H, nw)

            // token-major (nw, len, h, d): permute (d,n,h,b) -> (d,h,n,b)
            ao = ggml_cont(ctx, ggml_permute(ctx, ao, 0, 2, 1, 3));
            acc = acc ? ggml_concat(ctx, acc, ggml_reshape_2d(ctx, ao, C, len * nw), 1)
                      : ggml_reshape_2d(ctx, ao, C, len * nw);
        }
        // back to original token order + out projection + residual
        ggml_tensor* ao = ggml_get_rows(ctx, acc, bwd_idx);  // (C, N)
        ao = gb_linear(ctx, as_f32(m->get(b + ".attn.to_out.weight")),
                       as_f32(m->get(b + ".attn.to_out.bias")), ao);
        if (i == 0 && debug_stage == "b0_attn") return {ao};  // torch attn hook = post-to_out
        h = ggml_add(ctx, h, ao);
        if (i == 0 && debug_stage == "b0_full")
            return {ao, h, acc};  // all block0 intermediates in ONE graph
        if (debug_stage == "b" + std::to_string(i) + "_attn") return {h};

        // norm2 -> MLP -> residual
        ggml_tensor* h2 = ggml_norm(ctx, h, 1e-6f);
        ggml_tensor* m1 = gb_linear(ctx, as_f32(m->get(b + ".mlp.mlp.0.weight")),
                                    as_f32(m->get(b + ".mlp.mlp.0.bias")), h2);
        ggml_tensor* g = ggml_gelu(ctx, m1);
        ggml_tensor* m2 = gb_linear(ctx, as_f32(m->get(b + ".mlp.mlp.2.weight")),
                                    as_f32(m->get(b + ".mlp.mlp.2.bias")), g);
        if (i == 0 && debug_stage == "b0_mlp")
            return {ao, h_in, h2, g, m2};
        h = ggml_add(ctx, h, m2);
        if (debug_stage == "b" + std::to_string(i)) return {h};
    }

    if (torso_only) return {h};

    // ---- final LayerNorm (F.layer_norm, eps 1e-5) + out layer -----------
    h = ggml_norm(ctx, h, 1e-5f);
    ggml_tensor* out = gb_linear(ctx, as_f32(m->get(prefix + ".out_layer.weight")),
                                 as_f32(m->get(prefix + ".out_layer.bias")), h);
    GGML_ASSERT(out->ne[0] == out_ch);
    return {out};
}

}  // namespace sam3d
