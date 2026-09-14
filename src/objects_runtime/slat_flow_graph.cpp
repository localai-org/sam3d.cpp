// SLat flow model single-step graph - see slat_flow_graph.hpp.
#include "slat_flow_graph.hpp"

#include "graph_builder.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace sam3d {

static ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) t = ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// CUDA and Vulkan both execute Q8_0 x F32 directly. Keeping quantized matrix
// weights in their GGUF representation avoids materializing an F32 copy of
// every transformer projection in the graph allocator. Non-quantized models
// retain the established F32 compute path.
static ggml_tensor* matmul_weight(ggml_context* ctx, ggml_tensor* t,
                                  bool attention_projection = false,
                                  bool conv_projection = false) {
    // Keep the compact GGUF as the source of truth, while allowing a
    // reproducible F32 projection path for quantization parity bisects.
    // This is strictly opt-in: it must not change the normal Q4/Q8 graph.
    const bool dequant_q4 =
        (attention_projection && getenv("SAM3D_E2E_Q4_DEQUANT_ATTN") != nullptr) ||
        (conv_projection && getenv("SAM3D_E2E_Q4_DEQUANT_CONV") != nullptr);
    const bool dequant_q8 =
        (attention_projection && getenv("SAM3D_E2E_Q8_DEQUANT_ATTN") != nullptr) ||
        (conv_projection && getenv("SAM3D_E2E_Q8_DEQUANT_CONV") != nullptr);
    const bool dequant_all_q8 = getenv("SAM3D_E2E_Q8_DEQUANT_ALL") != nullptr &&
        t->type == GGML_TYPE_Q8_0;
    const bool dequantize = (t->type == GGML_TYPE_Q4_0 && dequant_q4) ||
        (t->type == GGML_TYPE_Q8_0 && (dequant_q8 || dequant_all_q8));
    return (ggml_is_quantized(t->type) && !dequantize)
        ? t : as_f32(ctx, t);
}

// Slice output rows from a [input_channels, output_channels] matrix. The
// byte offset must use nb[1]: multiplying logical elements by ggml_type_size
// is wrong for block-quantized rows.
static ggml_tensor* matrix_rows(ggml_context* ctx, ggml_tensor* weight,
                                int64_t row_start, int64_t row_count) {
    GGML_ASSERT(row_start >= 0 && row_count >= 0);
    GGML_ASSERT(row_start + row_count <= weight->ne[1]);
    return ggml_view_2d(ctx, weight, weight->ne[0], row_count, weight->nb[1],
                        static_cast<size_t>(row_start) * weight->nb[1]);
}

// h*(1+s) + sh; mod (C, 1) stays the second operand.
static ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* h,
                             ggml_tensor* scale, ggml_tensor* shift) {
    ggml_tensor* t = ggml_add(ctx, ggml_mul(ctx, h, scale), h);
    return ggml_add(ctx, t, shift);
}

std::vector<ggml_tensor*> SlatFlowGraph::build() {
    ggml_context* ctx = g->ctx();
    const int64_t C = m->i32(prefix + ".model_channels", 1024);
    const int64_t n_blocks = m->i32(prefix + ".num_blocks", 24);
    const int64_t n_heads = m->i32(prefix + ".num_heads", 16);
    const int64_t Hd = C / n_heads;
    const int64_t Cin = m->i32(prefix + ".in_channels", 8);
    const int64_t io_ch = 128;  // io_block_channels[0]
    const int64_t nf = tb->nf;
    const int64_t nc = tb->nc;

    // ---------------- graph inputs ----------------
    t = g->input_f32("t", {1, 1});
    cond = g->input_f32("cond", {C, n_cond_tokens});
    x = g->input_f32("x", {Cin, nf});

    // host tables uploaded as int32 inputs; built once by the caller
    auto add_i32 = [&](const std::vector<int32_t>& v, const char* name,
                       std::initializer_list<int64_t> ne) {
        ggml_tensor* tidx = g->input_i32(name, ne);
        inputs.push_back(tidx);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(v));
        return tidx;
    };
    ggml_tensor* idx_f2c = add_i32(tb->fine_to_coarse, "f2c_idx", {nf});
    (void)idx_f2c;
    ggml_tensor* idx_conv_f = add_i32(tb->conv_fine, "conv_fine_idx",
                                      {27 * nf});
    ggml_tensor* idx_conv_c = add_i32(tb->conv_coarse, "conv_coarse_idx",
                                      {27 * nc});

    // per-channel-width zero sentinel rows for the gathers
    auto zero_row = [&](int64_t cw, const char* name) {
        ggml_tensor* z = g->input_f32(name, {cw, 1});
        inputs.push_back(z);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(cw, 0));
        return z;
    };
    ggml_tensor* z8 = zero_row(Cin, "zero8");
    ggml_tensor* z128 = zero_row(io_ch, "zero128");
    ggml_tensor* z256 = zero_row(2 * io_ch, "zero256");
    ggml_tensor* z1024 = zero_row(C, "zero1024");
    ggml_tensor* z2048 = zero_row(2 * C, "zero2048");

    // SubM conv as one GEMM: gather 27 neighbour rows (missing -> sentinel);
    // the gathered layout (n, o, c) with c fastest == (27*C, N) ne, matching
    // the converter's (Cout, 27*Cin) row-format weight.
    auto zero_for = [&](int64_t cw) {
        return cw == Cin ? z8 : cw == io_ch ? z128 : cw == 2 * io_ch
                   ? z256 : cw == C ? z1024 : z2048;
    };
    auto conv = [&](ggml_tensor* feat, const std::string& wname,
                    const std::string& bname, ggml_tensor* idx, int64_t n) {
        const int64_t cw = feat->ne[0];
        ggml_tensor* zero = zero_for(cw);
        ggml_tensor* xc = ggml_concat(ctx, feat, zero, 1);          // (C, n+1)
        ggml_tensor* weight = matmul_weight(ctx, m->get(wname.c_str()), false, true);
        ggml_tensor* bias = as_f32(ctx, m->get(bname.c_str()));

        // Quantized CUDA GEMMs quantize their full activation matrix into a
        // temporary Q8 buffer. The first decoder convolution otherwise needs
        // one for 2048 x (27 * 26281) floats. Each token is independent, so
        // tile only oversized gathers and concatenate the original token order.
        constexpr int64_t kMaxGatherElements = 256LL * 1024 * 1024;
        const int64_t tile_tokens = std::max<int64_t>(
            1, kMaxGatherElements / (cw * 27));
        if (n <= tile_tokens) {
            ggml_tensor* gat = ggml_get_rows(ctx, xc, idx);         // (C, 27n)
            gat = ggml_reshape_2d(ctx, gat, cw * 27, n);
            return gb_linear(ctx, weight, bias, gat);
        }

        const std::vector<int32_t>* host_idx = idx == idx_conv_f
            ? &tb->conv_fine : idx == idx_conv_c ? &tb->conv_coarse : nullptr;
        GGML_ASSERT(host_idx != nullptr);
        ggml_tensor* result = nullptr;
        for (int64_t first = 0; first < n; first += tile_tokens) {
            const int64_t count = std::min(tile_tokens, n - first);
            // CUDA get_rows accepts strided source rows but not a view of an
            // index input reliably across a replayed gallocr graph. Make the
            // small I32 slice an explicit immutable input instead.
            auto slice = std::make_shared<std::vector<int32_t>>(
                host_idx->begin() + 27 * first,
                host_idx->begin() + 27 * (first + count));
            ggml_tensor* idx_tile = g->input_i32(
                "conv_index_tile", {27 * count});
            inputs.push_back(idx_tile);
            table_data.push_back(std::move(slice));
            ggml_tensor* gat = ggml_get_rows(ctx, xc, idx_tile);
            gat = ggml_reshape_2d(ctx, gat, cw * 27, count);
            ggml_tensor* tile = gb_linear(ctx, weight, bias, gat);
            result = result ? ggml_concat(ctx, result, tile, 1) : tile;
        }
        return result;
    };

    // 2x mean-pool downsample (torch scatter mean over present children;
    // blocks are partial so the children table is (8, nc) with sentinels)
    auto downsample = [&](ggml_tensor* feat) {
        const int64_t cw = feat->ne[0];
        ggml_tensor* zero = zero_for(cw);
        ggml_tensor* xc = ggml_concat(ctx, feat, zero, 1);
        ggml_tensor* idx = add_i32(tb->coarse_children, "coarse_children",
                                   {8 * nc});
        ggml_tensor* gat = ggml_get_rows(ctx, xc, idx);            // (C, 8nc)
        gat = ggml_reshape_3d(ctx, gat, cw, 8, nc);
        ggml_tensor* sum = nullptr;
        for (int j = 0; j < 8; j++) {
            ggml_tensor* vj = ggml_view_3d(ctx, gat, cw, 1, nc,
                                           gat->nb[1], gat->nb[2],
                                           (size_t)j * gat->nb[1]);
            // the child slice strides 8*cw per token: materialize before the
            // reshape (ggml reshape requires contiguous inputs)
            vj = ggml_reshape_2d(ctx, ggml_cont(ctx, vj), cw, nc);
            sum = sum ? ggml_add(ctx, sum, vj) : vj;
        }
        ggml_tensor* inv = g->input_f32("coarse_inv_count", {1, nc});
        inputs.push_back(inv);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(
            (const int32_t*)tb->coarse_inv_count.data(),
            (const int32_t*)(tb->coarse_inv_count.data() + nc)));
        return ggml_mul(ctx, sum, inv);
    };

    // 2x nearest upsample (fine_i = coarse[fine_to_coarse[i]])
    auto upsample = [&](ggml_tensor* feat) {
        ggml_tensor* xc = ggml_concat(ctx, feat, zero_for(feat->ne[0]), 1);
        ggml_tensor* gat = ggml_get_rows(ctx, xc, idx_f2c);        // (C, nf)
        return ggml_reshape_2d(ctx, gat, feat->ne[0], nf);
    };

    // ---------------- timestep embedding ----------------
    const int64_t half = 128;
    auto freq_buf = std::make_shared<std::vector<int32_t>>(half);
    float* ff = (float*)freq_buf->data();
    for (int i = 0; i < (int)half; i++)
        ff[i] = expf(-logf(10000.0f) * (float)i / (float)half);
    ggml_tensor* freqs = g->input_f32("t_freqs", {half, 1});
    inputs.push_back(freqs);
    table_data.push_back(freq_buf);
    ggml_tensor* tf = ggml_mul(ctx, ggml_repeat(ctx, t, freqs), freqs);
    ggml_tensor* emb = ggml_concat(ctx, ggml_cos(ctx, tf), ggml_sin(ctx, tf), 0);
    ggml_tensor* t_emb = gb_linear(
        ctx, matmul_weight(ctx, m->get(prefix + ".t_embedder.mlp.0.weight")),
        as_f32(ctx, m->get(prefix + ".t_embedder.mlp.0.bias")), emb);
    t_emb = ggml_silu(ctx, t_emb);
    t_emb = gb_linear(ctx,
                      matmul_weight(ctx, m->get(prefix + ".t_embedder.mlp.2.weight")),
                      as_f32(ctx, m->get(prefix + ".t_embedder.mlp.2.bias")),
                      t_emb);  // (C, 1)
    const bool dump_block_outputs = getenv("SAM3D_DEBUG_DUMP_BLOCKS") != nullptr;
    const bool fuse_quant_qkv = getenv("SAM3D_E2E_FUSE_QUANT_QKV") != nullptr;

    // sparse res block: [down/up] -> norm1(affine) -> silu -> conv1 ->
    // norm2(no affine) -> *(1+scale)+shift -> silu -> conv2 -> + skip
    auto res_block = [&](ggml_tensor* xin, const std::string& b, int64_t cout,
                         bool down, bool up, bool* dbg_early) {
        const bool trace_input_block1 = dump_block_outputs &&
            b == prefix + ".input_blocks.1";
        if (down) {
            xin = downsample(xin);
            if (trace_input_block1) debug_block_outputs.push_back(xin);
            if (debug_stage == b + "_down") {
                if (dbg_early) *dbg_early = true;
                return xin;
            }
        }
        if (up) {
            xin = upsample(xin);
            if (debug_stage == b + "_up") {
                if (dbg_early) *dbg_early = true;
                return xin;
            }
        }
        const bool coarse = down || up;
        const int64_t n = coarse ? nc : nf;
        ggml_tensor* idx = coarse ? idx_conv_c : idx_conv_f;
        ggml_tensor* e = gb_linear(
            ctx, matmul_weight(ctx, m->get(b + ".emb_layers.1.weight")),
            as_f32(ctx, m->get(b + ".emb_layers.1.bias")),
            ggml_silu(ctx, t_emb));  // (2*cout, 1)
        ggml_tensor* scale = ggml_view_2d(ctx, e, cout, 1, e->nb[1], 0);
        ggml_tensor* shift = ggml_view_2d(ctx, e, cout, 1, e->nb[1],
                                          cout * sizeof(float));
        ggml_tensor* hh = gb_layer_norm(
            ctx, xin, as_f32(ctx, m->get(b + ".norm1.weight")),
            as_f32(ctx, m->get(b + ".norm1.bias")), 1e-6f);
        hh = ggml_silu(ctx, hh);
        if (trace_input_block1) debug_block_outputs.push_back(hh);
        hh = conv(hh, b + ".conv1.conv.weight", b + ".conv1.conv.bias", idx, n);
        if (trace_input_block1) debug_block_outputs.push_back(hh);
        if (debug_stage == b + "_conv1") {
            if (dbg_early) *dbg_early = true;
            return hh;
        }
        hh = gb_layer_norm(ctx, hh, nullptr, nullptr, 1e-6f);
        hh = modulate(ctx, hh, scale, shift);
        hh = ggml_silu(ctx, hh);
        if (trace_input_block1) debug_block_outputs.push_back(hh);
        hh = conv(hh, b + ".conv2.conv.weight", b + ".conv2.conv.bias", idx, n);
        if (trace_input_block1) debug_block_outputs.push_back(hh);
        if (debug_stage == b + "_conv2") {
            if (dbg_early) *dbg_early = true;
            return hh;
        }
        ggml_tensor* skip = xin;
        if (xin->ne[0] != cout)
            skip = gb_linear(
                ctx, matmul_weight(ctx, m->get(b + ".skip_connection.weight")),
                as_f32(ctx, m->get(b + ".skip_connection.bias")), xin);
        if (trace_input_block1) debug_block_outputs.push_back(skip);
        ggml_tensor* out = ggml_add(ctx, hh, skip);
        if (trace_input_block1) debug_block_outputs.push_back(out);
        return out;
    };

    // ---------------- input stage ----------------
    ggml_tensor* h = gb_linear(
        ctx, matmul_weight(ctx, m->get(prefix + ".input_layer.weight")),
        as_f32(ctx, m->get(prefix + ".input_layer.bias")), x);  // (128, nf)
    if (debug_stage == "input_layer") return {h};
    if (dump_block_outputs) debug_block_outputs.push_back(h);

    bool early = false;
    ggml_tensor* skip0 = res_block(h, prefix + ".input_blocks.0", io_ch,
                                   false, false, &early);       // (128, nf)
    if (early) return {skip0};
    if (dump_block_outputs) debug_block_outputs.push_back(skip0);
    ggml_tensor* skip1 = res_block(skip0, prefix + ".input_blocks.1", C,
                                   true, false, &early);        // (1024, nc)
    if (early) return {skip1};
    if (dump_block_outputs) debug_block_outputs.push_back(skip1);

    // ---------------- APE + transformer blocks ----------------
    ggml_tensor* ape = g->input_f32("ape", {C, nc});
    inputs.push_back(ape);
    table_data.push_back(std::make_shared<std::vector<int32_t>>(
        (const int32_t*)tb->ape.data(),
        (const int32_t*)(tb->ape.data() + tb->ape.size())));
    h = ggml_add(ctx, skip1, ape);
    if (debug_stage == "ape") return {h};
    if (debug_stage == "block_in") return {h};

    for (int i = 0; i < (int)n_blocks; i++) {
        const std::string b = prefix + ".blocks." + std::to_string(i);
        ggml_tensor* six = gb_linear(
            ctx, matmul_weight(ctx, m->get(b + ".adaLN_modulation.1.weight")),
            as_f32(ctx, m->get(b + ".adaLN_modulation.1.bias")),
            ggml_silu(ctx, t_emb));
        if (i == 0 && debug_stage == "b0_adaln") return {six};
        ggml_tensor* shift_msa = ggml_view_2d(ctx, six, C, 1, six->nb[1], 0);
        ggml_tensor* scale_msa = ggml_view_2d(ctx, six, C, 1, six->nb[1],
                                              C * sizeof(float));
        ggml_tensor* gate_msa = ggml_view_2d(ctx, six, C, 1, six->nb[1],
                                             2 * (size_t)C * sizeof(float));
        ggml_tensor* shift_mlp = ggml_view_2d(ctx, six, C, 1, six->nb[1],
                                              3 * (size_t)C * sizeof(float));
        ggml_tensor* scale_mlp = ggml_view_2d(ctx, six, C, 1, six->nb[1],
                                              4 * (size_t)C * sizeof(float));
        ggml_tensor* gate_mlp = ggml_view_2d(ctx, six, C, 1, six->nb[1],
                                             5 * (size_t)C * sizeof(float));

        ggml_tensor* hs = gb_layer_norm(ctx, h, nullptr, nullptr, 1e-6f);
        hs = modulate(ctx, hs, scale_msa, shift_msa);
        if (debug_stage == "b" + std::to_string(i) + "_attn_in") return {hs};
        // A 3C-wide quantized GEMM can select a different CUDA/Vulkan
        // reduction kernel than three C-wide projections. Keep the established
        // narrow quantized path until a complete E2E equivalence test approves
        // a fused quantized kernel.
        ggml_tensor* wqkv = matmul_weight(
            ctx, m->get(b + ".self_attn.to_qkv.weight"), true);
        ggml_tensor* bqkv = ggml_cont(ctx, as_f32(ctx, m->get(b + ".self_attn.to_qkv.bias")));
        ggml_tensor *q, *k, *v;
        if (ggml_is_quantized(wqkv->type)) {
            if (fuse_quant_qkv) {
                ggml_tensor* qkv = gb_linear(ctx, wqkv, bqkv, hs);
                if (debug_stage == "b" + std::to_string(i) + "_qpre") {
                    return {ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, nc, qkv->nb[1], 0))};
                }
                if (i == 0 && debug_stage == "b0_qkv") return {qkv};
                gb_split_qkv(ctx, qkv, n_heads, &q, &k, &v);
            } else {
                ggml_tensor* wq = matrix_rows(ctx, wqkv, 0, C);
                ggml_tensor* wk = matrix_rows(ctx, wqkv, C, C);
                ggml_tensor* wv = matrix_rows(ctx, wqkv, 2 * C, C);
                q = gb_linear(ctx, wq, ggml_view_1d(ctx, bqkv, C, 0), hs);
                k = gb_linear(ctx, wk, ggml_view_1d(ctx, bqkv, C, C * sizeof(float)), hs);
                v = gb_linear(ctx, wv, ggml_view_1d(ctx, bqkv, C, 2 * C * sizeof(float)), hs);
                if (debug_stage == "b" + std::to_string(i) + "_qpre") return {ggml_cont(ctx, q)};
                if (i == 0 && debug_stage == "b0_qkv") return {gb_linear(ctx, wqkv, bqkv, hs)};
                auto heads3 = [&](ggml_tensor* t) {
                    return ggml_permute(ctx, ggml_reshape_3d(ctx, t, Hd, n_heads, nc), 0, 2, 1, 3);
                };
                q = heads3(q);
                k = heads3(k);
                v = heads3(v);
            }
        } else {
            ggml_tensor* qkv = gb_linear(ctx, wqkv, bqkv, hs);
            if (debug_stage == "b" + std::to_string(i) + "_qpre") {
                return {ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, nc, qkv->nb[1], 0))};
            }
            if (i == 0 && debug_stage == "b0_qkv") return {qkv};
            gb_split_qkv(ctx, qkv, n_heads, &q, &k, &v);
        }
        q = gb_rms_norm_head(
            ctx, q, as_f32(ctx, m->get(b + ".self_attn.q_rms_norm.gamma")),
            (float)Hd);
        if (i == 0 && debug_stage == "b0_qrms") {
            return {ggml_cont(ctx, q)};
        }
        k = gb_rms_norm_head(
            ctx, k, as_f32(ctx, m->get(b + ".self_attn.k_rms_norm.gamma")),
            (float)Hd);
        const float attn_scale = 1.0f / sqrtf((float)Hd);
        ggml_tensor* out = gb_attention(ctx, q, k, v, attn_scale, true);
        out = ggml_reshape_2d(ctx, out, C, nc);
        out = gb_linear(ctx,
                        matmul_weight(ctx, m->get(b + ".self_attn.to_out.weight"), true),
                        as_f32(ctx, m->get(b + ".self_attn.to_out.bias")), out);
        if (i == 0 && debug_stage == "b0_attn_out") return {out};
        h = ggml_add(ctx, h, ggml_mul(ctx, out, gate_msa));
        if (i == 0 && debug_stage == "b0_res") return {h};

        // cross attention (no rms norm on cross)
        ggml_tensor* hc = gb_layer_norm(
            ctx, h, as_f32(ctx, m->get(b + ".norm2.weight")),
            as_f32(ctx, m->get(b + ".norm2.bias")), 1e-6f);
        ggml_tensor* cq = gb_linear(
            ctx, matmul_weight(ctx, m->get(b + ".cross_attn.to_q.weight"), true),
            as_f32(ctx, m->get(b + ".cross_attn.to_q.bias")), hc);
        ggml_tensor* wkv = matmul_weight(
            ctx, m->get(b + ".cross_attn.to_kv.weight"), true);
        ggml_tensor* bkv = as_f32(ctx, m->get(b + ".cross_attn.to_kv.bias"));
        ggml_tensor *ck, *cv;
        if (ggml_is_quantized(wkv->type)) {
            ck = gb_linear(ctx, matrix_rows(ctx, wkv, 0, C),
                           ggml_view_1d(ctx, bkv, C, 0), cond);
            cv = gb_linear(ctx, matrix_rows(ctx, wkv, C, C),
                           ggml_view_1d(ctx, bkv, C, C * sizeof(float)), cond);
        } else {
            gb_split_kv(ctx, gb_linear(ctx, wkv, bkv, cond), &ck, &cv);
        }
        auto heads = [&](ggml_tensor* t2, int64_t n) {
            ggml_tensor* r = ggml_reshape_3d(ctx, t2, Hd, n_heads, n);
            return ggml_permute(ctx, r, 0, 2, 1, 3);
        };
        const int64_t n_q = cq->ne[1];
        const int64_t n_kv = ck->ne[1];
        cq = heads(cq, n_q);
        ck = heads(ck, n_kv);
        cv = heads(cv, n_kv);
        ggml_tensor* co = gb_attention(ctx, cq, ck, cv, attn_scale, true);
        co = ggml_reshape_2d(ctx, co, C, n_q);
        co = gb_linear(ctx,
                       matmul_weight(ctx, m->get(b + ".cross_attn.to_out.weight"), true),
                       as_f32(ctx, m->get(b + ".cross_attn.to_out.bias")), co);
        if (i == 0 && debug_stage == "b0_cross_out") return {co};
        h = ggml_add(ctx, h, co);
        if (i == 0 && debug_stage == "b0_cross_res") return {h};

        // MLP branch
        ggml_tensor* hm = gb_layer_norm(ctx, h, nullptr, nullptr, 1e-6f);
        hm = modulate(ctx, hm, scale_mlp, shift_mlp);
        hm = gb_linear(ctx, matmul_weight(ctx, m->get(b + ".mlp.mlp.0.weight")),
                       as_f32(ctx, m->get(b + ".mlp.mlp.0.bias")), hm);
        hm = ggml_gelu(ctx, hm);
        hm = gb_linear(ctx, matmul_weight(ctx, m->get(b + ".mlp.mlp.2.weight")),
                       as_f32(ctx, m->get(b + ".mlp.mlp.2.bias")), hm);
        if (i == 0 && debug_stage == "b0_mlp_out") return {hm};
        h = ggml_add(ctx, h, ggml_mul(ctx, hm, gate_mlp));
        if (dump_block_outputs) debug_block_outputs.push_back(h);
        if (debug_stage == "block" + std::to_string(i)) return {h};
    }

    // ---------------- output stage ----------------
    // out_blocks[0]: in = concat([h(1024), skip1(1024)]) = 2048 coarse ->
    // upsample -> convs 2048->128, 128->128; skip = Linear(2048->128).
    // out_blocks[1]: in = concat([h(128), skip0(128)]) = 256 fine -> convs
    // 256->128, 128->128; skip = Linear(256->128).
    {
        const std::string b = prefix + ".out_blocks.0";
        ggml_tensor* xin = ggml_concat(ctx, h, skip1, 0);  // (2048, nc)
        xin = upsample(xin);                               // (2048, nf)
        if (dump_block_outputs) debug_block_outputs.push_back(xin);
        ggml_tensor* e = gb_linear(
            ctx, matmul_weight(ctx, m->get(b + ".emb_layers.1.weight")),
            as_f32(ctx, m->get(b + ".emb_layers.1.bias")),
            ggml_silu(ctx, t_emb));
        ggml_tensor* scale = ggml_view_2d(ctx, e, io_ch, 1, e->nb[1], 0);
        ggml_tensor* shift = ggml_view_2d(ctx, e, io_ch, 1, e->nb[1],
                                          io_ch * sizeof(float));
        ggml_tensor* hh = gb_layer_norm(
            ctx, xin, as_f32(ctx, m->get(b + ".norm1.weight")),
            as_f32(ctx, m->get(b + ".norm1.bias")), 1e-6f);
        hh = ggml_silu(ctx, hh);
        if (dump_block_outputs) debug_block_outputs.push_back(hh);
        hh = conv(hh, b + ".conv1.conv.weight", b + ".conv1.conv.bias",
                  idx_conv_f, nf);
        if (dump_block_outputs) debug_block_outputs.push_back(hh);
        if (debug_stage == "ob0_conv1") return {hh};
        hh = gb_layer_norm(ctx, hh, nullptr, nullptr, 1e-6f);
        hh = modulate(ctx, hh, scale, shift);
        hh = ggml_silu(ctx, hh);
        hh = conv(hh, b + ".conv2.conv.weight", b + ".conv2.conv.bias",
                  idx_conv_f, nf);
        if (dump_block_outputs) debug_block_outputs.push_back(hh);
        ggml_tensor* skip = gb_linear(
            ctx, matmul_weight(ctx, m->get(b + ".skip_connection.weight")),
            as_f32(ctx, m->get(b + ".skip_connection.bias")), xin);
        if (dump_block_outputs) debug_block_outputs.push_back(skip);
        h = ggml_add(ctx, hh, skip);  // (128, nf)
        if (dump_block_outputs) debug_block_outputs.push_back(h);  // out block 0
    }
    {
        const std::string b = prefix + ".out_blocks.1";
        ggml_tensor* xin = ggml_concat(ctx, h, skip0, 0);  // (256, nf)
        ggml_tensor* e = gb_linear(
            ctx, matmul_weight(ctx, m->get(b + ".emb_layers.1.weight")),
            as_f32(ctx, m->get(b + ".emb_layers.1.bias")),
            ggml_silu(ctx, t_emb));
        ggml_tensor* scale = ggml_view_2d(ctx, e, io_ch, 1, e->nb[1], 0);
        ggml_tensor* shift = ggml_view_2d(ctx, e, io_ch, 1, e->nb[1],
                                          io_ch * sizeof(float));
        ggml_tensor* hh = gb_layer_norm(
            ctx, xin, as_f32(ctx, m->get(b + ".norm1.weight")),
            as_f32(ctx, m->get(b + ".norm1.bias")), 1e-6f);
        hh = ggml_silu(ctx, hh);
        hh = conv(hh, b + ".conv1.conv.weight", b + ".conv1.conv.bias",
                  idx_conv_f, nf);
        if (debug_stage == "ob1_conv1") return {hh};
        hh = gb_layer_norm(ctx, hh, nullptr, nullptr, 1e-6f);
        hh = modulate(ctx, hh, scale, shift);
        hh = ggml_silu(ctx, hh);
        hh = conv(hh, b + ".conv2.conv.weight", b + ".conv2.conv.bias",
                  idx_conv_f, nf);
        if (debug_stage == "ob1_conv2") return {hh};
        ggml_tensor* skip = gb_linear(
            ctx, matmul_weight(ctx, m->get(b + ".skip_connection.weight")),
            as_f32(ctx, m->get(b + ".skip_connection.bias")), xin);
        h = ggml_add(ctx, hh, skip);
        if (dump_block_outputs) debug_block_outputs.push_back(h);  // out block 1
    }

    if (debug_stage == "pre_final") return {h};
    h = gb_layer_norm(ctx, h, nullptr, nullptr, 1e-6f);
    if (dump_block_outputs) debug_block_outputs.push_back(h);  // final LayerNorm
    h = gb_linear(ctx, matmul_weight(ctx, m->get(prefix + ".out_layer.weight")),
                  as_f32(ctx, m->get(prefix + ".out_layer.bias")), h);
    return {h};
}

}  // namespace sam3d
