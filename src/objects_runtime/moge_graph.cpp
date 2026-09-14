#include "moge_graph.hpp"

#include "common.hpp"

#include <cmath>
#include <cstring>

namespace sam3d {
namespace {

std::vector<int32_t> make_patch_table(int height, int width, int patch) {
    const int patches_h = height / patch;
    const int patches_w = width / patch;
    const int n_patches = patches_h * patches_w;
    const int patch_values = 3 * patch * patch;
    std::vector<int32_t> table(static_cast<size_t>(patch_values) * n_patches);
    for (int py = 0; py < patches_h; ++py) {
        for (int px = 0; px < patches_w; ++px) {
            const int patch_index = py * patches_w + px;
            for (int channel = 0; channel < 3; ++channel) {
                for (int ky = 0; ky < patch; ++ky) {
                    for (int kx = 0; kx < patch; ++kx) {
                        const int kernel_index = channel * patch * patch + ky * patch + kx;
                        const int source_y = py * patch + ky;
                        const int source_x = px * patch + kx;
                        table[static_cast<size_t>(patch_index) * patch_values + kernel_index] =
                            (source_y * width + source_x) * 3 + channel;
                    }
                }
            }
        }
    }
    return table;
}

std::vector<float> make_uv(int width, int height, float aspect_ratio) {
    std::vector<float> uv(static_cast<size_t>(width) * height * 2);
    const float diagonal = std::sqrt(1.0f + aspect_ratio * aspect_ratio);
    for (int y = 0; y < height; ++y) {
        const float v = ((static_cast<float>(y) + 0.5f) / height * 2.0f - 1.0f) / diagonal;
        for (int x = 0; x < width; ++x) {
            const float u = ((static_cast<float>(x) + 0.5f) / width * 2.0f - 1.0f) *
                            aspect_ratio / diagonal;
            // ggml conv layout is [W, H, C, N].
            uv[static_cast<size_t>(x) + width * y] = u;
            uv[static_cast<size_t>(x) + width * height + width * y] = v;
        }
    }
    return uv;
}

// PyTorch bicubic interpolation with align_corners=false evaluates output
// coordinate i at (i + 0.5) / scale - 0.5 and uses Keys cubic a=-0.75. MoGe's
// DINO backbone passes an explicit scale_factor rather than an output size,
// including its historical +0.1 offset. ggml's UPSCALE op cannot represent
// that offset, so build the small, separable resampling matrices directly.
std::vector<float> make_bicubic_resample_table(int source_size, int target_size,
                                                float scale_factor) {
    GGML_ASSERT(source_size > 0 && target_size > 0 && scale_factor > 0.0f);
    constexpr float cubic_a = -0.75f;
    const auto cubic = [](float distance) {
        constexpr float a = cubic_a;
        distance = std::fabs(distance);
        if (distance <= 1.0f) {
            return ((a + 2.0f) * distance - (a + 3.0f)) * distance * distance + 1.0f;
        }
        if (distance < 2.0f) {
            return ((a * distance - 5.0f * a) * distance + 8.0f * a) * distance - 4.0f * a;
        }
        return 0.0f;
    };

    std::vector<float> table(static_cast<size_t>(source_size) * target_size, 0.0f);
    for (int target = 0; target < target_size; ++target) {
        const float source = (static_cast<float>(target) + 0.5f) / scale_factor - 0.5f;
        const int first = static_cast<int>(std::floor(source)) - 1;
        for (int tap = 0; tap < 4; ++tap) {
            const int unbounded = first + tap;
            const int clamped = std::max(0, std::min(source_size - 1, unbounded));
            table[static_cast<size_t>(target) * source_size + clamped] +=
                cubic(source - static_cast<float>(unbounded));
        }
    }
    return table;
}

ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(ctx, tensor, GGML_TYPE_F32);
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* bias, ggml_tensor* x) {
    return gb_linear(ctx, as_f32(ctx, weight), as_f32(ctx, bias), x);
}

// PyTorch Conv2d(..., padding_mode="replicate") represented with standard
// ggml operations.  No custom backend operation is needed, so CUDA/Vulkan
// execute the same source graph and no ggml source patch is introduced.
ggml_tensor* replicate_pad_2d(ggml_context* ctx, ggml_tensor* x, int padding) {
    GGML_ASSERT(padding >= 0);
    if (padding == 0) return x;
    const int64_t width = x->ne[0];
    const int64_t height = x->ne[1];
    const int64_t channels = x->ne[2];
    const int64_t batch = x->ne[3];
    GGML_ASSERT(width > 0 && height > 0);

    ggml_tensor* left = ggml_view_4d(ctx, x, 1, height, channels, batch,
                                      x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor* right = ggml_view_4d(ctx, x, 1, height, channels, batch,
                                       x->nb[1], x->nb[2], x->nb[3],
                                       (width - 1) * x->nb[0]);
    ggml_tensor* padded_x = x;
    for (int i = 0; i < padding; ++i) padded_x = ggml_concat(ctx, left, padded_x, 0);
    for (int i = 0; i < padding; ++i) padded_x = ggml_concat(ctx, padded_x, right, 0);

    const int64_t padded_width = padded_x->ne[0];
    ggml_tensor* top = ggml_view_4d(ctx, padded_x, padded_width, 1, channels, batch,
                                     padded_x->nb[1], padded_x->nb[2], padded_x->nb[3], 0);
    ggml_tensor* bottom = ggml_view_4d(ctx, padded_x, padded_width, 1, channels, batch,
                                        padded_x->nb[1], padded_x->nb[2], padded_x->nb[3],
                                        (height - 1) * padded_x->nb[1]);
    for (int i = 0; i < padding; ++i) padded_x = ggml_concat(ctx, top, padded_x, 1);
    for (int i = 0; i < padding; ++i) padded_x = ggml_concat(ctx, padded_x, bottom, 1);
    return padded_x;
}

ggml_tensor* add_channel_bias(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias) {
    bias = as_f32(ctx, bias);
    const int64_t channels = x->ne[2];
    GGML_ASSERT(ggml_nelements(bias) == channels);
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, bias, 1, 1, channels, 1));
}

ggml_tensor* conv2d_replicate(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* bias,
                               ggml_tensor* x) {
    weight = as_f32(ctx, weight);
    GGML_ASSERT(weight->ne[0] == weight->ne[1]);
    const int padding = static_cast<int>(weight->ne[0] / 2);
    x = replicate_pad_2d(ctx, x, padding);
    x = ggml_conv_2d_direct(ctx, weight, x, 1, 1, 0, 0, 1, 1);
    return add_channel_bias(ctx, x, bias);
}

ggml_tensor* conv2d_1x1(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* bias,
                         ggml_tensor* x) {
    weight = as_f32(ctx, weight);
    GGML_ASSERT(weight->ne[0] == 1 && weight->ne[1] == 1);
    if (weight->ne[2] != x->ne[2]) {
        LOGE("MoGe 1x1 convolution channel mismatch: weight=[%lld,%lld,%lld,%lld] input=[%lld,%lld,%lld,%lld]",
             static_cast<long long>(weight->ne[0]), static_cast<long long>(weight->ne[1]),
             static_cast<long long>(weight->ne[2]), static_cast<long long>(weight->ne[3]),
             static_cast<long long>(x->ne[0]), static_cast<long long>(x->ne[1]),
             static_cast<long long>(x->ne[2]), static_cast<long long>(x->ne[3]));
    }
    x = ggml_conv_2d_direct(ctx, weight, x, 1, 1, 0, 0, 1, 1);
    return add_channel_bias(ctx, x, bias);
}

ggml_tensor* group_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight,
                        ggml_tensor* bias, int groups) {
    x = ggml_group_norm(ctx, x, groups, 1e-5f);
    const int64_t channels = x->ne[2];
    weight = as_f32(ctx, weight);
    bias = as_f32(ctx, bias);
    GGML_ASSERT(ggml_nelements(weight) == channels && ggml_nelements(bias) == channels);
    x = ggml_mul(ctx, x, ggml_reshape_4d(ctx, weight, 1, 1, channels, 1));
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, bias, 1, 1, channels, 1));
}

ggml_tensor* residual_conv_block(ggml_context* ctx, const GGUFModel* model,
                                 const std::string& prefix, ggml_tensor* x) {
    const int64_t channels = x->ne[2];
    ggml_tensor* h = group_norm(ctx, x, model->get(prefix + ".layers.0.weight"),
                                model->get(prefix + ".layers.0.bias"), 1);
    h = ggml_relu(ctx, h);
    h = conv2d_replicate(ctx, model->get(prefix + ".layers.2.weight"),
                         model->get(prefix + ".layers.2.bias"), h);
    const int groups = static_cast<int>(h->ne[2] / 32);
    GGML_ASSERT(groups > 0 && h->ne[2] % 32 == 0);
    h = group_norm(ctx, h, model->get(prefix + ".layers.3.weight"),
                   model->get(prefix + ".layers.3.bias"), groups);
    h = ggml_relu(ctx, h);
    h = conv2d_replicate(ctx, model->get(prefix + ".layers.5.weight"),
                         model->get(prefix + ".layers.5.bias"), h);
    GGML_ASSERT(x->ne[2] == channels && h->ne[2] == channels);
    return ggml_add(ctx, x, h);
}

ggml_tensor* tokens_to_image(ggml_context* ctx, ggml_tensor* tokens, int64_t width, int64_t height) {
    const int64_t channels = tokens->ne[0];
    GGML_ASSERT(tokens->ne[1] == width * height);
    ggml_tensor* chw = ggml_reshape_4d(ctx, tokens, channels, width, height, 1);
    // ggml_permute arguments map source axes to destination axes.  Move
    // source C->dst2, W->dst0, H->dst1 to obtain the Conv2d [W,H,C,N] form.
    return ggml_cont(ctx, ggml_permute(ctx, chw, 2, 0, 1, 3));
}

}  // namespace

MogeOutputs MogeGraph::build(ggml_tensor* img) {
    GGML_ASSERT(g != nullptr && m != nullptr && img != nullptr);
    ggml_context* ctx = g->ctx();
    const int64_t resized_height = img->ne[2];
    const int64_t resized_width = img->ne[1];
    const int64_t patch = m->u32("moge.image_patch_size", 14);
    const int64_t hidden = m->u32("moge.hidden_size", 1024);
    const int64_t heads = m->u32("moge.num_heads", 16);
    const int64_t blocks = m->u32("moge.num_blocks", 24);
    const int64_t intermediate = m->u32("moge.intermediate_layers", 4);
    const int64_t residual_blocks = m->u32("moge.residual_blocks", 2);
    GGML_ASSERT(img->ne[0] == 3 && resized_width >= patch && resized_height >= patch);
    const int64_t dino_width = resized_width / patch * patch;
    const int64_t dino_height = resized_height / patch * patch;
    const int64_t patches_w = dino_width / patch;
    const int64_t patches_h = dino_height / patch;
    const int64_t n_patches = patches_w * patches_h;
    const int64_t token_count = 1 + n_patches;
    const int64_t patch_values = 3 * patch * patch;
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(hidden / heads));

    auto add_f32_input = [&](const char* name, std::initializer_list<int64_t> shape,
                             std::shared_ptr<std::vector<float>> values) {
        ggml_tensor* tensor = g->input_f32(name, shape);
        inputs.push_back(tensor);
        f32_data.push_back(std::move(values));
        return tensor;
    };
    auto add_i32_input = [&](const char* name, std::initializer_list<int64_t> shape,
                             std::shared_ptr<std::vector<int32_t>> values) {
        ggml_tensor* tensor = g->input_i32(name, shape);
        inputs.push_back(tensor);
        i32_data.push_back(std::move(values));
        return tensor;
    };

    auto mean = std::make_shared<std::vector<float>>(std::initializer_list<float>{0.485f, 0.456f, 0.406f});
    auto inverse_std = std::make_shared<std::vector<float>>(
        std::initializer_list<float>{1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f});
    ggml_tensor* mean_tensor = add_f32_input("moge_mean", {3, 1, 1}, mean);
    ggml_tensor* inverse_std_tensor = add_f32_input("moge_inverse_std", {3, 1, 1}, inverse_std);
    img = ggml_mul(ctx, ggml_sub(ctx, img, mean_tensor), inverse_std_tensor);
    // The official DINO input is a bilinear, align_corners=false resize of
    // the normalized MoGe image to dimensions divisible by 14.  `img` has
    // HWC storage but ggml's spatial operators require [W, H, C, N], so make
    // both layout transitions explicit.  Treating [C,W,H] as a convolution
    // tensor directly only happens to work for already divisible square
    // images and corrupts a real arbitrary-size input.
    ggml_tensor* dino_img = ggml_cont(ctx, ggml_permute(ctx, img, 2, 0, 1, 3));
    dino_img = ggml_interpolate(ctx, dino_img, dino_width, dino_height, 3, 1,
                                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS);
    img = ggml_cont(ctx, ggml_permute(ctx, dino_img, 1, 2, 0, 3));

    auto patch_table = std::make_shared<std::vector<int32_t>>(
        make_patch_table(static_cast<int>(dino_height), static_cast<int>(dino_width), static_cast<int>(patch)));
    ggml_tensor* patch_indices = add_i32_input("moge_patch_table", {patch_values * n_patches}, patch_table);
    ggml_tensor* image_vector = ggml_view_2d(ctx, img, 1, ggml_nelements(img), sizeof(float), 0);
    ggml_tensor* patches = ggml_get_rows(ctx, image_vector, patch_indices);
    patches = ggml_reshape_2d(ctx, patches, patch_values, n_patches);
    ggml_tensor* patch_weight = as_f32(ctx, m->get(prefix + ".backbone.patch_embed.proj.weight"));
    ggml_tensor* patch_bias = as_f32(ctx, m->get(prefix + ".backbone.patch_embed.proj.bias"));
    ggml_tensor* x = linear(ctx,
        ggml_view_2d(ctx, patch_weight, patch_values, hidden, patch_values * sizeof(float), 0),
        patch_bias, patches);

    // This MoGe checkpoint uses DINOv2 bicubic position interpolation with
    // align_corners=false and its historical 0.1 scale-factor offset. Build
    // the equivalent separable resampler from normal ggml matrix operations,
    // which keeps the implementation portable to CPU, CUDA, and Vulkan.
    ggml_tensor* pos = as_f32(ctx, m->get(prefix + ".backbone.pos_embed"));
    GGML_ASSERT(pos->ne[0] == hidden && pos->ne[1] == 1 + 37 * 37);
    ggml_tensor* cls_pos = ggml_view_2d(ctx, pos, hidden, 1, hidden * sizeof(float), 0);
    ggml_tensor* patch_pos = ggml_view_4d(ctx, pos, hidden, 37, 37, 1,
                                           hidden * sizeof(float), 37 * hidden * sizeof(float),
                                           37 * 37 * hidden * sizeof(float), hidden * sizeof(float));
    auto pos_x_values = std::make_shared<std::vector<float>>(make_bicubic_resample_table(
        37, static_cast<int>(patches_w), (static_cast<float>(patches_w) + 0.1f) / 37.0f));
    auto pos_y_values = std::make_shared<std::vector<float>>(make_bicubic_resample_table(
        37, static_cast<int>(patches_h), (static_cast<float>(patches_h) + 0.1f) / 37.0f));
    ggml_tensor* pos_x = add_f32_input("moge_pos_resample_x", {37, patches_w}, pos_x_values);
    ggml_tensor* pos_y = add_f32_input("moge_pos_resample_y", {37, patches_h}, pos_y_values);
    // [C, W, H] -> [W, C, H], resample W, then [H, W, C], resample H.
    ggml_tensor* pos_w = ggml_cont(ctx, ggml_permute(ctx, patch_pos, 1, 0, 2, 3));
    pos_w = ggml_mul_mat(ctx, pos_x, ggml_reshape_2d(ctx, pos_w, 37, hidden * 37));
    pos_w = ggml_reshape_3d(ctx, pos_w, patches_w, hidden, 37);
    ggml_tensor* pos_h = ggml_cont(ctx, ggml_permute(ctx, pos_w, 1, 2, 0, 3));
    pos_h = ggml_mul_mat(ctx, pos_y, ggml_reshape_2d(ctx, pos_h, 37, patches_w * hidden));
    pos_h = ggml_reshape_3d(ctx, pos_h, patches_h, patches_w, hidden);
    patch_pos = ggml_cont(ctx, ggml_permute(ctx, pos_h, 2, 1, 0, 3));
    patch_pos = ggml_reshape_2d(ctx, patch_pos, hidden, n_patches);
    patch_pos = ggml_reshape_2d(ctx, patch_pos, hidden, n_patches);
    ggml_tensor* cls = as_f32(ctx, m->get(prefix + ".backbone.cls_token"));
    x = ggml_concat(ctx, ggml_reshape_2d(ctx, cls, hidden, 1), x, 1);
    x = ggml_add(ctx, x, ggml_concat(ctx, cls_pos, patch_pos, 1));
    ggml_tensor* backbone_input = x;

    std::vector<ggml_tensor*> features;
    features.reserve(static_cast<size_t>(intermediate));
    std::vector<ggml_tensor*> attention_outputs;
    attention_outputs.reserve(static_cast<size_t>(blocks));
    std::vector<ggml_tensor*> mlp_fc1_outputs;
    mlp_fc1_outputs.reserve(static_cast<size_t>(blocks));
    std::vector<ggml_tensor*> mlp_gelu_outputs;
    mlp_gelu_outputs.reserve(static_cast<size_t>(blocks));
    std::vector<ggml_tensor*> mlp_outputs;
    mlp_outputs.reserve(static_cast<size_t>(blocks));
    std::vector<ggml_tensor*> block_outputs;
    block_outputs.reserve(static_cast<size_t>(blocks));
    ggml_tensor* debug_q = nullptr;
    ggml_tensor* debug_k = nullptr;
    ggml_tensor* debug_v = nullptr;
    ggml_tensor* debug_attention_context = nullptr;
    for (int i = 0; i < blocks; ++i) {
        const std::string block = prefix + ".backbone.blocks." + std::to_string(i);
        ggml_tensor* h = gb_layer_norm(ctx, x, m->get(block + ".norm1.weight"),
                                       m->get(block + ".norm1.bias"), 1e-6f);
        ggml_tensor* qkv_weight = as_f32(ctx, m->get(block + ".attn.qkv.weight"));
        ggml_tensor* qkv_bias = as_f32(ctx, m->get(block + ".attn.qkv.bias"));
        // Keep the QKV projection as one contiguous GEMM. Apart from being
        // faster, this avoids backend-dependent handling of non-zero-offset
        // weight views for K/V. gb_split_qkv materializes the logical slices
        // before the head layout conversion.
        ggml_tensor* qkv = linear(ctx, qkv_weight, qkv_bias, h);
        ggml_tensor* q = nullptr;
        ggml_tensor* k = nullptr;
        ggml_tensor* v = nullptr;
        gb_split_qkv(ctx, qkv, static_cast<int>(heads), &q, &k, &v);
        ggml_tensor* attention = gb_attention(ctx, q, k, v, attention_scale, true);
        if (i == 0) {
            debug_q = ggml_cont(ctx, q);
            debug_k = ggml_cont(ctx, k);
            debug_v = ggml_cont(ctx, v);
            debug_attention_context = attention;
        }
        attention = ggml_reshape_2d(ctx, attention, hidden, token_count);
        attention = linear(ctx, m->get(block + ".attn.proj.weight"),
                           m->get(block + ".attn.proj.bias"), attention);
        attention = ggml_mul(ctx, attention, m->get(block + ".ls1.gamma"));
        attention_outputs.push_back(attention);
        x = ggml_add(ctx, x, attention);
        h = gb_layer_norm(ctx, x, m->get(block + ".norm2.weight"),
                          m->get(block + ".norm2.bias"), 1e-6f);
        // Keep each MLP stage addressable by the native/PyTorch parity
        // harness. These tensors are graph-local diagnostics; regular runs
        // request only the final point and mask tensors.
        ggml_tensor* mlp_fc1 = linear(ctx, m->get(block + ".mlp.fc1.weight"),
                                      m->get(block + ".mlp.fc1.bias"), h);
        ggml_tensor* mlp_gelu = ggml_gelu_erf(ctx, mlp_fc1);
        h = linear(ctx, m->get(block + ".mlp.fc2.weight"),
                   m->get(block + ".mlp.fc2.bias"), mlp_gelu);
        h = ggml_mul(ctx, h, m->get(block + ".ls2.gamma"));
        mlp_fc1_outputs.push_back(mlp_fc1);
        mlp_gelu_outputs.push_back(mlp_gelu);
        mlp_outputs.push_back(h);
        x = ggml_add(ctx, x, h);
        block_outputs.push_back(x);
        if (i >= blocks - static_cast<int>(intermediate)) {
            h = gb_layer_norm(ctx, x, m->get(prefix + ".backbone.norm.weight"),
                              m->get(prefix + ".backbone.norm.bias"), 1e-6f);
            features.push_back(ggml_view_2d(ctx, h, hidden, n_patches,
                                             hidden * sizeof(float), hidden * sizeof(float)));
        }
    }
    GGML_ASSERT(features.size() == static_cast<size_t>(intermediate));

    ggml_tensor* head = nullptr;
    for (int i = 0; i < static_cast<int>(features.size()); ++i) {
        ggml_tensor* feature = tokens_to_image(ctx, features[static_cast<size_t>(i)], patches_w, patches_h);
        feature = conv2d_1x1(ctx, m->get(prefix + ".head.projects." + std::to_string(i) + ".weight"),
                              m->get(prefix + ".head.projects." + std::to_string(i) + ".bias"), feature);
        head = head == nullptr ? feature : ggml_add(ctx, head, feature);
    }
    ggml_tensor* projected_features = head;
    std::vector<ggml_tensor*> upsample_outputs;
    std::vector<ggml_tensor*> third_upsample_stages;
    ggml_tensor* third_transpose_weight_f32 = nullptr;
    for (int i = 0; i < 3; ++i) {
        const int64_t width = head->ne[0];
        const int64_t height = head->ne[1];
        const std::string block = prefix + ".head.upsample_blocks." + std::to_string(i);
        auto uv = std::make_shared<std::vector<float>>(
            make_uv(static_cast<int>(width), static_cast<int>(height),
                    static_cast<float>(resized_width) / resized_height));
        ggml_tensor* uv_tensor = add_f32_input("moge_up_uv", {width, height, 2, 1}, uv);
        head = ggml_concat(ctx, head, uv_tensor, 2);
        if (i == 2) third_upsample_stages.push_back(head);
        ggml_tensor* transpose_weight = m->get(block + ".0.0.weight");
        if (i == 2) {
            transpose_weight = as_f32(ctx, transpose_weight);
            third_transpose_weight_f32 = transpose_weight;
        }
        head = ggml_conv_transpose_2d_p0(ctx, as_f32(ctx, transpose_weight), head, 2);
        head = add_channel_bias(ctx, head, m->get(block + ".0.0.bias"));
        if (i == 2) third_upsample_stages.push_back(head);
        head = conv2d_replicate(ctx, m->get(block + ".0.1.weight"),
                                m->get(block + ".0.1.bias"), head);
        if (i == 2) third_upsample_stages.push_back(head);
        for (int64_t residual = 0; residual < residual_blocks; ++residual) {
            head = residual_conv_block(ctx, m,
                block + "." + std::to_string(residual + 1), head);
            if (i == 2) third_upsample_stages.push_back(head);
        }
        upsample_outputs.push_back(head);
    }
    head = ggml_interpolate(ctx, head, resized_width, resized_height, head->ne[2], 1,
                            GGML_SCALE_MODE_BILINEAR);
    auto output_uv = std::make_shared<std::vector<float>>(
        make_uv(static_cast<int>(resized_width), static_cast<int>(resized_height),
                static_cast<float>(resized_width) / resized_height));
    ggml_tensor* output_uv_tensor = add_f32_input("moge_output_uv",
        {resized_width, resized_height, 2, 1}, output_uv);
    head = ggml_concat(ctx, head, output_uv_tensor, 2);
    ggml_tensor* output_block_input = head;

    const std::string output_prefix = prefix + ".head.output_block.";
    ggml_tensor* points = conv2d_replicate(ctx, m->get(output_prefix + "0.0.weight"),
                                            m->get(output_prefix + "0.0.bias"), head);
    ggml_tensor* points_hidden = points;
    points = ggml_relu(ctx, points);
    points = conv2d_1x1(ctx, m->get(output_prefix + "0.2.weight"),
                         m->get(output_prefix + "0.2.bias"), points);
    ggml_tensor* points_raw = points;
    ggml_tensor* mask = conv2d_replicate(ctx, m->get(output_prefix + "1.0.weight"),
                                          m->get(output_prefix + "1.0.bias"), head);
    ggml_tensor* mask_hidden = mask;
    mask = ggml_relu(ctx, mask);
    mask = conv2d_1x1(ctx, m->get(output_prefix + "1.2.weight"),
                      m->get(output_prefix + "1.2.bias"), mask);
    ggml_tensor* mask_raw = mask;

    const int64_t final_width = output_width > 0 ? output_width : resized_width;
    const int64_t final_height = output_height > 0 ? output_height : resized_height;
    GGML_ASSERT(final_width > 0 && final_height > 0);
    if (final_width != points->ne[0] || final_height != points->ne[1]) {
        points = ggml_interpolate(ctx, points, final_width, final_height, 3, 1,
                                  GGML_SCALE_MODE_BILINEAR);
        mask = ggml_interpolate(ctx, mask, final_width, final_height, 1, 1,
                                GGML_SCALE_MODE_BILINEAR);
    }

    GGML_ASSERT(points->ne[2] == 3 && mask->ne[2] == 1);
    ggml_tensor* xy = ggml_view_4d(ctx, points, points->ne[0], points->ne[1], 2, 1,
                                    points->nb[1], points->nb[2], points->nb[3], 0);
    ggml_tensor* z = ggml_view_4d(ctx, points, points->ne[0], points->ne[1], 1, 1,
                                   points->nb[1], points->nb[2], points->nb[3], 2 * points->nb[2]);
    z = ggml_exp(ctx, z);
    xy = ggml_mul(ctx, xy, ggml_repeat(ctx, z, xy));
    return {ggml_concat(ctx, xy, z, 2), mask, backbone_input, debug_q, debug_k, debug_v,
            debug_attention_context, attention_outputs, mlp_fc1_outputs, mlp_gelu_outputs,
            mlp_outputs, block_outputs, features, projected_features, upsample_outputs,
            third_upsample_stages, third_transpose_weight_f32, output_block_input,
            {points_hidden, mask_hidden}, {points_raw, mask_raw}};
}

}  // namespace sam3d
