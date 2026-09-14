#include "mesh_decoder_graph.hpp"

#include "graph_builder.hpp"

#include <cstring>

namespace sam3d {
namespace {

ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(ctx, tensor, GGML_TYPE_F32);
}

// SparseGroupNorm32 normalizes each channel group across every active token
// in a batch. The mesh parity/runtime path currently operates on batch one.
ggml_tensor* sparse_group_norm(ggml_context* ctx, ggml_tensor* x,
                               ggml_tensor* weight, ggml_tensor* bias,
                               int groups) {
    const int64_t channels = x->ne[0];
    const int64_t tokens = x->ne[1];
    ggml_tensor* channel_major = ggml_cont(ctx, ggml_transpose(ctx, x)); // [tokens, channels]
    channel_major = ggml_reshape_3d(ctx, channel_major, tokens, 1, channels);
    channel_major = ggml_group_norm(ctx, channel_major, groups, 1e-5f);
    channel_major = ggml_mul(ctx, channel_major,
        ggml_reshape_3d(ctx, as_f32(ctx, weight), 1, 1, channels));
    channel_major = ggml_add(ctx, channel_major,
        ggml_reshape_3d(ctx, as_f32(ctx, bias), 1, 1, channels));
    channel_major = ggml_reshape_2d(ctx, channel_major, tokens, channels);
    return ggml_cont(ctx, ggml_transpose(ctx, channel_major));
}

ggml_tensor* repeat_children(ggml_context* ctx, ggml_tensor* x) {
    const int64_t channels = x->ne[0];
    const int64_t parents = x->ne[1];
    ggml_tensor* source = ggml_reshape_3d(ctx, x, channels, 1, parents);
    ggml_tensor* shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, 8, parents);
    return ggml_reshape_2d(ctx, ggml_repeat(ctx, source, shape), channels, parents * 8);
}

}  // namespace

ggml_tensor* MeshUpsampleGraph::build() {
    GGML_ASSERT(g && m && tables && x);
    GGML_ASSERT(level == 0 || level == 1);
    ggml_context* ctx = g->ctx();
    const std::string prefix = "meshdec.upsample." + std::to_string(level);
    const int64_t parent_count = tables->parent_count;
    const int64_t child_count = tables->child_count;
    const int64_t input_channels = level == 0 ? 768 : 192;
    const int64_t output_channels = level == 0 ? 192 : 96;
    GGML_ASSERT(x->ne[0] == input_channels && x->ne[1] == parent_count);

    auto add_i32 = [&](std::vector<int32_t> values, const std::string& name) {
        ggml_tensor* tensor = g->input_i32(name, {child_count});
        inputs.push_back(tensor);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(std::move(values)));
        return tensor;
    };
    auto add_f32 = [&](std::vector<float> values, const std::string& name) {
        ggml_tensor* tensor = g->input_f32(name, {1, child_count});
        inputs.push_back(tensor);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(
            reinterpret_cast<const int32_t*>(values.data()),
            reinterpret_cast<const int32_t*>(values.data() + values.size())));
        return tensor;
    };

    std::vector<ggml_tensor*> neighbor_index(27), neighbor_mask(27);
    for (int offset = 0; offset < 27; ++offset) {
        std::vector<int32_t> index((size_t)child_count);
        std::vector<float> mask((size_t)child_count);
        for (int64_t row = 0; row < child_count; ++row) {
            const int32_t source = tables->neighbors[(size_t)row * 27 + offset];
            index[(size_t)row] = source < 0 ? 0 : source;
            mask[(size_t)row] = source < 0 ? 0.0f : 1.0f;
        }
        neighbor_index[offset] = add_i32(std::move(index), "mesh_neighbor_index");
        neighbor_mask[offset] = add_f32(std::move(mask), "mesh_neighbor_mask");
    }

    auto convolution = [&](ggml_tensor* features, const std::string& name,
                           int64_t in_channels, int64_t out_channels,
                           int kernel_volume) {
        ggml_tensor* packed = m->get(name + ".weight");
        ggml_tensor* bias = m->get(name + ".bias");
        GGML_ASSERT(packed && bias && packed->ne[0] == in_channels * out_channels &&
                    packed->ne[1] == kernel_volume);
        ggml_tensor* sum = nullptr;
        for (int offset = 0; offset < kernel_volume; ++offset) {
            const size_t byte_offset = (size_t)offset * packed->nb[1];
            ggml_tensor* weight = ggml_view_1d(ctx, packed,
                in_channels * out_channels, byte_offset);
            weight = ggml_reshape_2d(ctx, weight, in_channels, out_channels);
            weight = as_f32(ctx, weight);
            ggml_tensor* gathered = kernel_volume == 1 ? features
                : ggml_get_rows(ctx, features, neighbor_index[offset]);
            if (kernel_volume != 1)
                gathered = ggml_mul(ctx, gathered, neighbor_mask[offset]);
            ggml_tensor* value = ggml_mul_mat(ctx, weight, gathered);
            sum = sum ? ggml_add(ctx, sum, value) : value;
        }
        return ggml_add(ctx, sum, as_f32(ctx, bias));
    };

    ggml_tensor* norm1 = sparse_group_norm(ctx, x,
        m->get(prefix + ".act_layers.0.weight"),
        m->get(prefix + ".act_layers.0.bias"), 32);
    if (debug_stage == "norm1") return norm1;
    ggml_tensor* silu1 = ggml_silu(ctx, norm1);
    if (debug_stage == "silu1") return silu1;
    ggml_tensor* subdivided = repeat_children(ctx, silu1);
    if (debug_stage == "subdivided") return subdivided;

    ggml_tensor* conv1 = convolution(subdivided, prefix + ".out_layers.0.conv",
                                     input_channels, output_channels, 27);
    if (debug_stage == "conv1") return conv1;
    ggml_tensor* norm2 = sparse_group_norm(ctx, conv1,
        m->get(prefix + ".out_layers.1.weight"),
        m->get(prefix + ".out_layers.1.bias"), 32);
    if (debug_stage == "norm2") return norm2;
    ggml_tensor* silu2 = ggml_silu(ctx, norm2);
    if (debug_stage == "silu2") return silu2;
    ggml_tensor* conv2 = convolution(silu2, prefix + ".out_layers.3.conv",
                                     output_channels, output_channels, 27);
    if (debug_stage == "conv2") return conv2;
    ggml_tensor* skip_input = repeat_children(ctx, x);
    if (debug_stage == "skip_input") return skip_input;
    ggml_tensor* skip = convolution(skip_input, prefix + ".skip_connection.conv",
                                    input_channels, output_channels, 1);
    if (debug_stage == "skip") return skip;
    return ggml_add(ctx, conv2, skip);
}

ggml_tensor* build_mesh_output_layer(GraphContext& graph, const GGUFModel& model,
                                     ggml_tensor* features) {
    GGML_ASSERT(features->ne[0] == 96);
    ggml_context* ctx = graph.ctx();
    return gb_linear(ctx, as_f32(ctx, model.get("meshdec.out_layer.weight")),
                     as_f32(ctx, model.get("meshdec.out_layer.bias")), features);
}

}  // namespace sam3d
