#include "ss_decoder_graph.hpp"

#include "graph_builder.hpp"
#include "common.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

namespace sam3d {

namespace {

// Shared build state threaded through the helpers.
struct Ctx {
    ggml_context* ctx;
    const GGUFModel* m;
    std::vector<ggml_tensor*>* debug_tensors;
    ggml_tensor* im2col_shape = nullptr;
    int64_t max_conv_oc_ic = 0;
    // when set, the first conv3d returns this intermediate tensor directly
    // (as the graph output its buffer stays live, making dumps reliable)
    const char* debug_stage = nullptr;

    void dbg(ggml_tensor* t) { debug_tensors->push_back(t); }
};

// LayerNorm over the channel dim (ne[3]) via permute to ne[0].
// ggml_permute(a,b,c,d) moves old dimension i to new position axis_i.
ggml_tensor* channel_layer_norm(Ctx& c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    // contiguous materialization is required: norm/unary ops need
    // contiguous rows, and the permuted views are not row-contiguous
    ggml_tensor* h = ggml_cont(c.ctx, ggml_permute(c.ctx, x, 1, 2, 3, 0));  // (C, W, H, D)
    h = gb_layer_norm(c.ctx, h, w, b, 1e-5f);  // nn.LayerNorm default eps
    return ggml_cont(c.ctx, ggml_permute(c.ctx, h, 3, 0, 1, 2));  // (W, H, D, C)
}

// Conv3d(pad=1, stride=1) lowered through ggml's native IM2COL_3D.  The
// canonical activation is already [W, H, D, IC], exactly the source layout
// required by the backend-neutral operator.  A single F32 im2col plus GEMM
// replaces the former 27 gather/GEMM branches, avoiding both padding-buffer
// aliasing and per-offset submission overhead on CUDA and Vulkan.
ggml_tensor* conv3d(Ctx& c, const std::string& prefix, ggml_tensor* x, int pad) {
    ggml_tensor* w = c.m->get(prefix + ".weight");
    ggml_tensor* b = c.m->get(prefix + ".bias");
    GGML_ASSERT(w && "conv3d weight missing");
    // Quantized (q8_0) weights cannot be sliced per GEMM offset by byte
    // arithmetic - the per-offset rows do not land on block boundaries.
    // Dequantize once up front, then reuse the F32 path below: the weight
    // precision stays q8_0 (8-bit block quant), the compute stays fp32.
    if (ggml_is_quantized(w->type)) {
        w = ggml_cast(c.ctx, w, GGML_TYPE_F32);
    }
    const int64_t ic = x->ne[3];
    if (getenv("SAM3D_DEBUG_CONV_INPUT") && prefix == "dec.middle_block.0.conv1") {
        LOGI("%s input ne=(%lld,%lld,%lld,%lld) nb=(%zu,%zu,%zu,%zu)",
             prefix.c_str(), (long long)x->ne[0], (long long)x->ne[1],
             (long long)x->ne[2], (long long)x->ne[3], x->nb[0], x->nb[1],
             x->nb[2], x->nb[3]);
        c.dbg(x);
        ggml_set_output(x);
    }
    GGML_ASSERT(w->ne[0] == 27 * ic && "SS Conv3D weight layout mismatch");
    const int64_t oc = w->ne[1];  // weight ne = {27*IC, OC}
    const int64_t W = x->ne[0], H = x->ne[1], D = x->ne[2];
    GGML_ASSERT(oc * ic <= c.max_conv_oc_ic && "invalid SS Conv3D shape metadata");
    if (!c.im2col_shape) {
        // IM2COL_3D needs a contiguous kernel-shaped source for its shape,
        // but never reads its data. One shared proxy avoids duplicating every
        // Conv3D weight solely to satisfy that shape contract.
        c.im2col_shape = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 3, 3, 3,
                                             c.max_conv_oc_ic);
        ggml_set_name(c.im2col_shape, "ss_im2col_shape_proxy");
    }
    ggml_tensor* kernel_shape = ggml_view_4d(c.ctx, c.im2col_shape, 3, 3, 3, oc * ic,
                                              c.im2col_shape->nb[1], c.im2col_shape->nb[2],
                                              c.im2col_shape->nb[3], 0);
    ggml_tensor* cols = ggml_im2col_3d(c.ctx, kernel_shape, x, ic,
                                        1, 1, 1, pad, pad, pad, 1, 1, 1,
                                        GGML_TYPE_F32);
    const std::string im2col_stage = prefix + ".im2col";
    const std::string gemm_stage = prefix + ".gemm";
    const std::string convout_stage = prefix + ".convout";
    if (c.debug_stage && ((prefix == "dec.input_layer" && strcmp(c.debug_stage, "im2col") == 0) ||
                          im2col_stage == c.debug_stage)) {
        return cols;
    }
    ggml_tensor* cols2d = ggml_reshape_2d(c.ctx, cols, cols->ne[0],
                                           cols->ne[1] * cols->ne[2] * cols->ne[3]);
    ggml_tensor* acc = ggml_mul_mat(c.ctx, w, cols2d);  // (OC, W*H*D)
    if (c.debug_stage && ((prefix == "dec.input_layer" && strcmp(c.debug_stage, "gemm") == 0) ||
                          gemm_stage == c.debug_stage)) {
        return acc;
    }
    ggml_tensor* out = ggml_reshape_4d(c.ctx, acc, oc, W, H, D);
    out = ggml_cont(c.ctx, ggml_permute(c.ctx, out, 3, 0, 1, 2));
    if (b) {
        if (getenv("SAM3D_DEBUG_CONV")) {
            LOGI("%s out=(%lld,%lld,%lld,%lld) bias=%lld", prefix.c_str(),
                 (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2],
                 (long long)out->ne[3], (long long)b->ne[0]);
        }
        out = ggml_add(c.ctx, out, ggml_reshape_4d(c.ctx, b, 1, 1, 1, oc));
    }
    if (c.debug_stage && ((prefix == "dec.input_layer" && strcmp(c.debug_stage, "convout") == 0) ||
                          convout_stage == c.debug_stage)) {
        return out;
    }
    if (getenv("SAM3D_KEEP_SS_DEBUG") && prefix == "dec.input_layer") {
        ggml_set_output(out);
    }
    return out;
}


ggml_tensor* res_block(Ctx& c, const std::string& prefix, ggml_tensor* x) {
    ggml_tensor* h = channel_layer_norm(c, x, c.m->get(prefix + ".norm1.weight"),
                                        c.m->get(prefix + ".norm1.bias"));
    if (c.debug_stage && prefix + ".norm1" == c.debug_stage) return h;
    h = ggml_silu(c.ctx, h);
    if (c.debug_stage && prefix + ".silu1" == c.debug_stage) return h;
    h = conv3d(c, prefix + ".conv1", h, 1);
    if (c.debug_stage && (prefix + ".conv1" == c.debug_stage ||
                          prefix + ".conv1.im2col" == c.debug_stage ||
                          prefix + ".conv1.gemm" == c.debug_stage ||
                          prefix + ".conv1.convout" == c.debug_stage)) return h;
    h = channel_layer_norm(c, h, c.m->get(prefix + ".norm2.weight"),
                           c.m->get(prefix + ".norm2.bias"));
    if (c.debug_stage && prefix + ".norm2" == c.debug_stage) return h;
    h = ggml_silu(c.ctx, h);
    if (c.debug_stage && prefix + ".silu2" == c.debug_stage) return h;
    h = conv3d(c, prefix + ".conv2", h, 1);
    if (c.debug_stage && (prefix + ".conv2" == c.debug_stage ||
                          prefix + ".conv2.im2col" == c.debug_stage ||
                          prefix + ".conv2.gemm" == c.debug_stage ||
                          prefix + ".conv2.convout" == c.debug_stage)) return h;
    return ggml_add(c.ctx, h, x);
}

// UpsampleBlock3d: a plain conv3d to OC*8 channels on the input grid, then
// 3D pixel shuffle. The reshape/permute sequence is exactly
// out[w',h',d',c] = conv[w/2,h/2,d/2,c*8+d'%2*4+h'%2*2+w'%2].
ggml_tensor* upsample_block(Ctx& c, const std::string& prefix, ggml_tensor* x) {
    ggml_tensor* h = conv3d(c, prefix + ".conv", x, 1);  // (W, H, D, OC*8)
    ggml_tensor* w = c.m->get(prefix + ".conv.weight");
    GGML_ASSERT(w);
    const int64_t oc8 = w->ne[1];
    GGML_ASSERT(oc8 % 8 == 0);
    const int64_t W = x->ne[0], H = x->ne[1], D = x->ne[2];
    const int64_t OC = oc8 / 8;

    h = ggml_cont(c.ctx, ggml_permute(c.ctx, h, 1, 2, 3, 0));
    h = ggml_reshape_4d(c.ctx, h, 2, OC * 4, W, H * D);
    h = ggml_cont(c.ctx, ggml_permute(c.ctx, h, 0, 2, 1, 3));
    h = ggml_reshape_4d(c.ctx, h, W * 2, OC * 4, H, D);

    h = ggml_cont(c.ctx, ggml_permute(c.ctx, h, 1, 0, 2, 3));
    h = ggml_reshape_4d(c.ctx, h, 2, OC * 2, W * 2, H * D);
    h = ggml_cont(c.ctx, ggml_permute(c.ctx, h, 2, 1, 0, 3));
    h = ggml_reshape_4d(c.ctx, h, W * 2, OC * 2, H * 2, D);

    h = ggml_cont(c.ctx, ggml_permute(c.ctx, h, 1, 0, 2, 3));
    h = ggml_reshape_4d(c.ctx, h, 2, OC, W * 2 * H * 2, D);
    h = ggml_cont(c.ctx, ggml_permute(c.ctx, h, 1, 3, 0, 2));
    return ggml_reshape_4d(c.ctx, h, W * 2, H * 2, D * 2, OC);
}

}  // namespace

ggml_tensor* SsDecoderGraph::build(ggml_tensor* latent) {
    const std::string conv_layout = m->str("dec.conv_layout");
    GGML_ASSERT(conv_layout == "im2col3d-v1" &&
                "SS decoder GGUF must be regenerated with the current converter");
    Ctx c{ctx, m, &debug_tensors, nullptr, (int64_t)m->u32("dec.max_conv_oc_ic"),
          getenv("SAM3D_DEBUG_STAGE")};
    const auto channels = m->i32_array("dec.channels");
    GGML_ASSERT(!channels.empty());
    const int n_res = (int)m->u32("dec.num_res_blocks");
    const int n_mid = (int)m->u32("dec.num_res_blocks_middle");

    ggml_tensor* h = conv3d(c, "dec.input_layer", latent, 1);
    if (c.debug_stage && (!strcmp(c.debug_stage, "im2col") ||
                          !strcmp(c.debug_stage, "gemm") ||
                          !strcmp(c.debug_stage, "convout"))) return h;
    c.dbg(h);
    for (int i = 0; i < n_mid; i++) {
        h = res_block(c, "dec.middle_block." + std::to_string(i), h);
        if (c.debug_stage) return h;
        c.dbg(h);
    }
    // flat ModuleList: ResBlocks followed by an Upsample before each channel
    // transition (see SparseStructureDecoder).
    const int n_blocks = n_res * (int)channels.size() + ((int)channels.size() - 1);
    for (int i = 0; i < n_blocks; i++) {
        const std::string p = "dec.blocks." + std::to_string(i);
        if (m->has(p + ".conv.weight")) {
            h = upsample_block(c, p, h);
        } else {
            h = res_block(c, p, h);
        }
        c.dbg(h);
    }
    // out_layer: CLN -> SiLU -> conv3d(channels[-1] -> 1)
    h = channel_layer_norm(c, h, m->get("dec.out_layer.0.weight"),
                           m->get("dec.out_layer.0.bias"));
    h = ggml_silu(ctx, h);
    h = conv3d(c, "dec.out_layer.2", h, 1);
    c.dbg(h);
    const char* dbg_env = getenv("SAM3D_DEBUG_NODE");
    if (dbg_env) {
        const int idx = atoi(dbg_env);
        if (idx >= 0 && idx < (int)debug_tensors.size()) {
            LOGI("debug node %d of %zu", idx, debug_tensors.size());
            return debug_tensors[idx];
        }
    }
    return h;
}

}  // namespace sam3d
