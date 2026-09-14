// MOT sparse-structure DiT single-step graph - see ss_flow_graph.hpp.
#include "ss_flow_graph.hpp"

#include "graph_builder.hpp"
#include "common.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace sam3d {

static ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) t = ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// CUDA and Vulkan execute quantized matrix weights directly. Keep only
// elementwise parameters in F32; casting a projection here adds a full
// dequantization node to every diffusion evaluation and defeats Q4/Q8 GEMM.
static ggml_tensor* matmul_weight(ggml_context* ctx, ggml_tensor* t) {
    return ggml_is_quantized(t->type) ? t : as_f32(ctx, t);
}

// A quantized logical row occupies ne[0] / block_size physical blocks. Use
// nb[1], not logical elements times ggml_type_size(), for fused QKV slices.
static ggml_tensor* matrix_rows(ggml_context* ctx, ggml_tensor* weight,
                                int64_t row_start, int64_t row_count) {
    GGML_ASSERT(row_start >= 0 && row_count >= 0);
    GGML_ASSERT(row_start + row_count <= weight->ne[1]);
    return ggml_view_2d(ctx, weight, weight->ne[0], row_count, weight->nb[1],
                        static_cast<size_t>(row_start) * weight->nb[1]);
}

// h*(1+s) + sh without a constant-1 tensor: (h*s + h) + sh, all broadcasts
// keep mod (C, 1) as the second operand.
static ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* h,
                             ggml_tensor* scale, ggml_tensor* shift) {
    ggml_tensor* t = ggml_add(ctx, ggml_mul(ctx, h, scale), h);
    return ggml_add(ctx, t, shift);
}

std::vector<ggml_tensor*> SsFlowGraph::build() {
    ggml_context* ctx = g->ctx();
    const int64_t C = m->i32(prefix + ".model_channels", 1024);
    const int64_t n_blocks = m->i32(prefix + ".num_blocks", 24);
    const int64_t n_heads = m->i32(prefix + ".num_heads", 16);
    const int64_t Hd = C / n_heads;
    const int64_t n_shape = m->i32(prefix + ".shape_tokens", 4096);
    const int64_t n_pose = 4;  // merged [6drotation, translation, scale, ts]
    const bool fuse_quant_qkv = getenv("SAM3D_E2E_FUSE_QUANT_QKV") != nullptr;

    // Optional Q4 residual factors use ``.lora_a/.lora_b`` or, when the GGUF
    // tensor-name limit requires it, compact ``.ra/.rb`` suffixes. Their
    // absence is the common path and leaves existing GGUF graphs unchanged.
    auto residual_factor = [&](const std::string& weight_name, char factor) {
        const std::string long_suffix = factor == 'a' ? ".lora_a" : ".lora_b";
        if (ggml_tensor* tensor = m->get(weight_name + long_suffix)) return tensor;
        const std::string short_suffix = factor == 'a' ? ".ra" : ".rb";
        return m->get(weight_name + short_suffix);
    };
    auto linear = [&](const std::string& weight_name, ggml_tensor* bias, ggml_tensor* input) {
        ggml_tensor* weight = m->get(weight_name);
        ggml_tensor* output = gb_linear(ctx, matmul_weight(ctx, weight), bias, input);
        ggml_tensor* factor_a = residual_factor(weight_name, 'a');
        ggml_tensor* factor_b = residual_factor(weight_name, 'b');
        if (!factor_a && !factor_b) return output;
        GGML_ASSERT(factor_a && factor_b);
        ggml_tensor* residual = gb_linear(ctx, matmul_weight(ctx, factor_a), nullptr, input);
        residual = gb_linear(ctx, matmul_weight(ctx, factor_b), nullptr, residual);
        return ggml_add(ctx, output, residual);
    };
    auto linear_rows = [&](const std::string& weight_name, int64_t row_start, int64_t row_count,
                           ggml_tensor* bias, ggml_tensor* input) {
        ggml_tensor* weight = m->get(weight_name);
        ggml_tensor* output = gb_linear(ctx,
                                        matrix_rows(ctx, matmul_weight(ctx, weight), row_start, row_count),
                                        bias, input);
        ggml_tensor* factor_a = residual_factor(weight_name, 'a');
        ggml_tensor* factor_b = residual_factor(weight_name, 'b');
        if (!factor_a && !factor_b) return output;
        GGML_ASSERT(factor_a && factor_b);
        ggml_tensor* residual = gb_linear(ctx, matmul_weight(ctx, factor_a), nullptr, input);
        residual = gb_linear(ctx, matmul_weight(ctx,
                                                matrix_rows(ctx, factor_b, row_start, row_count)),
                             nullptr, residual);
        return ggml_add(ctx, output, residual);
    };

    // ---------------- graph inputs ----------------
    // scaled timestep / shortcut step (host multiplies by time_scale)
    t = g->input_f32("t", {1, 1});
    d = g->input_f32("d", {1, 1});
    const int64_t cond_ch = m->i32(prefix + ".cond_channels", 1024);
    cond = g->input_f32("cond", {cond_ch, n_cond_tokens});
    x_shape = g->input_f32("x_shape", {8, n_shape});
    x_6drot = g->input_f32("x_6drot", {6, 1});
    x_scale = g->input_f32("x_scale", {3, 1});
    x_trans = g->input_f32("x_trans", {3, 1});
    x_ts = g->input_f32("x_ts", {1, 1});

    // ---------------- timestep embedding ----------------
    // freqs = exp(-ln(10000) * i / 128), i = 0..127; emb = [cos(t f), sin(t f)]
    const int64_t half = 128;
    freq_data = std::make_shared<std::vector<int32_t>>(half);
    float* ff = (float*)freq_data->data();
    for (int i = 0; i < (int)half; i++)
        ff[i] = expf(-logf(10000.0f) * (float)i / (float)half);
    ggml_tensor* freqs = g->input_f32("t_freqs", {half, 1});
    inputs.push_back(freqs);
    table_data.push_back(freq_data);

    auto embedder = [&](ggml_tensor* scalar, const std::string& pfx) {
        ggml_tensor* tf = ggml_mul(ctx, ggml_repeat(ctx, scalar, freqs), freqs);
        ggml_tensor* c = ggml_cos(ctx, tf);
        ggml_tensor* s = ggml_sin(ctx, tf);
        ggml_tensor* emb = ggml_concat(ctx, c, s, 0);  // (256, 1)
        ggml_tensor* h = linear(pfx + ".mlp.0.weight", as_f32(ctx, m->get(pfx + ".mlp.0.bias")), emb);
        h = ggml_silu(ctx, h);
        return linear(pfx + ".mlp.2.weight", as_f32(ctx, m->get(pfx + ".mlp.2.bias")), h);  // (C, 1)
    };
    ggml_tensor* mod = ggml_add(ctx, embedder(t, prefix + ".t_embedder"),
                                embedder(d, prefix + ".d_embedder"));  // (C,1)
    if (debug_stage == "t_emb") return {mod};

    // ---------------- project input ----------------
    // per-modality: input_layer (C, ch) + pos_emb (C, tokens)
    auto proj_in = [&](const std::string& name, ggml_tensor* x) {
        ggml_tensor* h = linear(prefix + ".latent_mapping." + name + ".input_layer.weight",
                                as_f32(ctx, m->get(prefix + ".latent_mapping." + name + ".input_layer.bias")), x);
        ggml_tensor* pe = as_f32(ctx, m->get(prefix + ".latent_mapping." + name + ".pos_emb"));
        return ggml_add(ctx, h, pe);
    };
    ggml_tensor* hs = proj_in("shape", x_shape);          // (C, 4096)
    ggml_tensor* h6 = proj_in("6drotation_normalized", x_6drot);
    ggml_tensor* hsc = proj_in("scale", x_scale);
    ggml_tensor* htr = proj_in("translation", x_trans);
    ggml_tensor* hts = proj_in("translation_scale", x_ts);
    // merge order = latent_share_transformer list order
    ggml_tensor* hp = ggml_concat(ctx, h6, htr, 1);
    hp = ggml_concat(ctx, hp, hsc, 1);
    hp = ggml_concat(ctx, hp, hts, 1);                    // (C, 4)
    if (debug_stage == "proj_in") return {hs, hp};

    // ---------------- transformer blocks ----------------
    const std::string pose_name = "6drotation_normalized";
    for (int i = 0; i < (int)n_blocks; i++) {
        const std::string b = prefix + ".blocks." + std::to_string(i);
        // adaLN modulation: SiLU -> Linear(C -> 6C), chunked in 6
        ggml_tensor* six = linear(b + ".adaLN_modulation.1.weight",
                                  as_f32(ctx, m->get(b + ".adaLN_modulation.1.bias")),
                                  ggml_silu(ctx, mod));                              // (6C, 1)
        if (i == 0 && debug_stage == "b0_adaln") return {six};
        ggml_tensor* shift_msa = ggml_view_2d(ctx, six, C, 1, six->nb[1], 0 * C * sizeof(float));
        ggml_tensor* scale_msa = ggml_view_2d(ctx, six, C, 1, six->nb[1], 1 * C * sizeof(float));
        ggml_tensor* gate_msa  = ggml_view_2d(ctx, six, C, 1, six->nb[1], 2 * C * sizeof(float));
        ggml_tensor* shift_mlp = ggml_view_2d(ctx, six, C, 1, six->nb[1], 3 * C * sizeof(float));
        ggml_tensor* scale_mlp = ggml_view_2d(ctx, six, C, 1, six->nb[1], 4 * C * sizeof(float));
        ggml_tensor* gate_mlp  = ggml_view_2d(ctx, six, C, 1, six->nb[1], 5 * C * sizeof(float));

        // MOT self-attention over the two groups.
        // shape group (protected): attends only itself.
        ggml_tensor* h_s = gb_layer_norm(ctx, hs, nullptr, nullptr, 1e-6f);
        h_s = modulate(ctx, h_s, scale_msa, shift_msa);
        ggml_tensor* h_p = gb_layer_norm(ctx, hp, nullptr, nullptr, 1e-6f);
        h_p = modulate(ctx, h_p, scale_msa, shift_msa);
        if (i == 0 && debug_stage == "b0_attn_in_s") return {h_s};
        if (i == 0 && debug_stage == "b0_attn_in_p") return {h_p};

        auto qkv_heads = [&](ggml_tensor* h, const std::string& mn,
                             ggml_tensor** q, ggml_tensor** k, ggml_tensor** v) {
            const std::string qkv_name = b + ".self_attn.to_qkv." + mn + ".weight";
            ggml_tensor* wqkv = matmul_weight(ctx, m->get(qkv_name));
            ggml_tensor* bqkv = as_f32(ctx, m->get(b + ".self_attn.to_qkv." + mn + ".bias"));
            if (ggml_is_quantized(wqkv->type)) {
                if (fuse_quant_qkv) {
                    // One 3C-wide GEMM cuts two quantized launches per QKV
                    // projection. gb_split_qkv materializes the same three
                    // contiguous tensors required by the attention layout.
                    ggml_tensor* qkv = linear(qkv_name, bqkv, h);
                    gb_split_qkv(ctx, qkv, n_heads, q, k, v);
                } else {
                    *q = linear_rows(qkv_name, 0, C, ggml_view_1d(ctx, bqkv, C, 0), h);
                    *k = linear_rows(qkv_name, C, C, ggml_view_1d(ctx, bqkv, C, C * sizeof(float)), h);
                    *v = linear_rows(qkv_name, 2 * C, C,
                                     ggml_view_1d(ctx, bqkv, C, 2 * C * sizeof(float)), h);
                    auto heads = [&](ggml_tensor* t) {
                        return ggml_permute(ctx, ggml_reshape_3d(ctx, t, Hd, n_heads, h->ne[1]),
                                            0, 2, 1, 3);
                    };
                    *q = heads(*q);
                    *k = heads(*k);
                    *v = heads(*v);
                }
            } else {
                ggml_tensor* qkv = linear(qkv_name, bqkv, h);
                gb_split_qkv(ctx, qkv, n_heads, q, k, v);
            }
            *q = gb_rms_norm_head(ctx, *q,
                                  as_f32(ctx, m->get(b + ".self_attn.q_rms_norm." + mn + ".gamma")),
                                  (float)Hd);
            *k = gb_rms_norm_head(ctx, *k,
                                  as_f32(ctx, m->get(b + ".self_attn.k_rms_norm." + mn + ".gamma")),
                                  (float)Hd);
            if (i == 0 && debug_stage == "b0_q_" + mn) *q = ggml_cont(ctx, *q);
        };
        ggml_tensor *qs, *ks, *vs, *qp, *kp, *vp;
        qkv_heads(h_s, "shape", &qs, &ks, &vs);
        qkv_heads(h_p, pose_name, &qp, &kp, &vp);
        if (i == 0 && debug_stage == "b0_q_shape") return {qs};
        if (i == 0 && debug_stage == "b0_q_6drotation_normalized") return {qp};

        const float attn_scale = 1.0f / sqrtf((float)Hd);
        // shape: self-attention within 4096 tokens
        ggml_tensor* out_s = gb_attention(ctx, qs, ks, vs, attn_scale, true);
        // pose group: attends pose + shape keys (inference: no detach needed)
        ggml_tensor* kc = ggml_concat(ctx, kp, ks, 1);  // (Hd, 4+n_shape, heads)
        ggml_tensor* vc = ggml_concat(ctx, vp, vs, 1);
        ggml_tensor* out_p = gb_attention(ctx, qp, kc, vc, attn_scale, true);
        // flash output memory is token-major (N, H, Hd) -> straight reshape
        out_s = ggml_reshape_2d(ctx, out_s, C, n_shape);
        out_p = ggml_reshape_2d(ctx, out_p, C, n_pose);

        out_s = linear(b + ".self_attn.to_out.shape.weight",
                       as_f32(ctx, m->get(b + ".self_attn.to_out.shape.bias")), out_s);
        out_p = linear(b + ".self_attn.to_out." + pose_name + ".weight",
                       as_f32(ctx, m->get(b + ".self_attn.to_out." + pose_name + ".bias")), out_p);
        if (i == 0 && debug_stage == "b0_attn_out_s") return {out_s};
        if (i == 0 && debug_stage == "b0_attn_out_p") return {out_p};
        hs = ggml_add(ctx, hs, ggml_mul(ctx, out_s, gate_msa));
        hp = ggml_add(ctx, hp, ggml_mul(ctx, out_p, gate_msa));
        if (i == 0 && debug_stage == "b0_res_s") return {hs};
        if (i == 0 && debug_stage == "b0_res_p") return {hp};

        // cross attention to the fused condition tokens (per modality)
        // debug slots indexed by modality: 0=shape 1=pose
        ggml_tensor* d_n2[2] = {nullptr, nullptr};
        ggml_tensor* d_xo[2] = {nullptr, nullptr};
        ggml_tensor* d_xq[2] = {nullptr, nullptr};
        ggml_tensor* d_xk[2] = {nullptr, nullptr};
        ggml_tensor* d_xv[2] = {nullptr, nullptr};
        auto xidx = [](const std::string& mn2) { return mn2 == "shape" ? 0 : 1; };
        auto cross = [&](ggml_tensor* x_g, const std::string& mn) {
            const int di = xidx(mn);
            ggml_tensor* h = gb_layer_norm(
                ctx, x_g, as_f32(ctx, m->get(b + ".norm2." + mn + ".weight")),
                as_f32(ctx, m->get(b + ".norm2." + mn + ".bias")), 1e-6f);
            d_n2[di] = h;
            const std::string q_name = b + ".cross_attn." + mn + ".to_q.weight";
            ggml_tensor* q = linear(q_name, as_f32(ctx, m->get(b + ".cross_attn." + mn + ".to_q.bias")), h);
            d_xq[di] = q;
            // Keep quantized K/V projections narrow until complete E2E
            // equivalence approves a fused quantized reduction. F16/F32 may
            // use one GEMM.
            const std::string kv_name = b + ".cross_attn." + mn + ".to_kv.weight";
            ggml_tensor* wkv = matmul_weight(ctx, m->get(kv_name));
            ggml_tensor* bkv = as_f32(ctx, m->get(b + ".cross_attn." + mn + ".to_kv.bias"));
            ggml_tensor *k, *v;
            if (ggml_is_quantized(wkv->type)) {
                k = linear_rows(kv_name, 0, C, ggml_view_1d(ctx, bkv, C, 0), cond);
                v = linear_rows(kv_name, C, C, ggml_view_1d(ctx, bkv, C, C * sizeof(float)), cond);
            } else {
                gb_split_kv(ctx, linear(kv_name, bkv, cond), &k, &v);
            }
            d_xk[di] = k;
            d_xv[di] = v;
            const int64_t n_q = q->ne[1];
            const int64_t n_kv = k->ne[1];
            auto heads = [&](ggml_tensor* t2, int64_t n) {
                ggml_tensor* r = ggml_reshape_3d(ctx, t2, Hd, n_heads, n);
                return ggml_permute(ctx, r, 0, 2, 1, 3);
            };
            q = heads(q, n_q);
            k = heads(k, n_kv);
            v = heads(v, n_kv);
            ggml_tensor* o = gb_attention(ctx, q, k, v, attn_scale, true);
            o = ggml_reshape_2d(ctx, o, C, q->ne[1]);  // flash out ne=(Hd,H,N)
            o = linear(b + ".cross_attn." + mn + ".to_out.weight",
                       as_f32(ctx, m->get(b + ".cross_attn." + mn + ".to_out.bias")), o);
            d_xo[di] = o;
            return ggml_add(ctx, x_g, o);
        };
        hs = cross(hs, "shape");
        hp = cross(hp, pose_name);
        if (i == 0 && debug_stage.rfind("b0_xq_", 0) == 0) return {d_xq[xidx(debug_stage.substr(6))]};
        if (i == 0 && debug_stage.rfind("b0_xk_", 0) == 0) return {d_xk[xidx(debug_stage.substr(6))]};
        if (i == 0 && debug_stage.rfind("b0_xv_", 0) == 0) return {d_xv[xidx(debug_stage.substr(6))]};
        if (i == 0 && debug_stage == "b0_cross_in_s") return {d_n2[0]};
        if (i == 0 && debug_stage == "b0_cross_in_p") return {d_n2[1]};
        if (i == 0 && debug_stage == "b0_cross_out_s") return {d_xo[0]};
        if (i == 0 && debug_stage == "b0_cross_out_p") return {d_xo[1]};

        // MLP branch (per modality, shared adaLN gates)
        ggml_tensor* d_mlp[2] = {nullptr, nullptr};
        auto mlp_branch = [&](ggml_tensor* x_g, const std::string& mn) {
            const int di = xidx(mn);
            ggml_tensor* h = gb_layer_norm(ctx, x_g, nullptr, nullptr, 1e-6f);
            h = modulate(ctx, h, scale_mlp, shift_mlp);
            h = linear(b + ".mlp." + mn + ".mlp.0.weight",
                       as_f32(ctx, m->get(b + ".mlp." + mn + ".mlp.0.bias")), h);
            h = ggml_gelu(ctx, h);  // tdfy FeedForwardNet: GELU(tanh)
            h = linear(b + ".mlp." + mn + ".mlp.2.weight",
                       as_f32(ctx, m->get(b + ".mlp." + mn + ".mlp.2.bias")), h);
            d_mlp[di] = h;
            h = ggml_mul(ctx, h, gate_mlp);
            return ggml_add(ctx, x_g, h);
        };
        hs = mlp_branch(hs, "shape");
        hp = mlp_branch(hp, pose_name);
        if (i == 0 && debug_stage == "b0_mlp_out_s") return {d_mlp[0]};
        if (i == 0 && debug_stage == "b0_mlp_out_p") return {d_mlp[1]};
        if (debug_stage == "block" + std::to_string(i)) return {hs, hp};
    }

    // ---------------- project output ----------------
    // per modality: parameterless LayerNorm then out_layer (in_ch, C)
    auto proj_out = [&](ggml_tensor* h, const std::string& name) {
        h = gb_layer_norm(ctx, h, nullptr, nullptr, 1e-6f);
        return linear(prefix + ".latent_mapping." + name + ".out_layer.weight",
                      as_f32(ctx, m->get(prefix + ".latent_mapping." + name + ".out_layer.bias")), h);
    };
    // split the pose group back (order: 6drot, translation, scale, ts)
    auto pose_view = [&](int64_t token) {
        return ggml_view_2d(ctx, hp, C, 1, hp->nb[1], token * C * sizeof(float));
    };
    ggml_tensor* o6 = pose_view(0);
    ggml_tensor* otr = pose_view(1);
    ggml_tensor* osc = pose_view(2);
    ggml_tensor* ots = pose_view(3);
    return {
        proj_out(o6, "6drotation_normalized"),
        proj_out(osc, "scale"),
        proj_out(hs, "shape"),
        proj_out(otr, "translation"),
        proj_out(ots, "translation_scale"),
    };
}

}  // namespace sam3d
