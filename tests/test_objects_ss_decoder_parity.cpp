#include "backend.hpp"
#include "common.hpp"
#include "gguf_loader.hpp"
#include "graph_builder.hpp"
#include "ss_decoder_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: parity MODULE CPU|Vulkan DEVICE MODEL.gguf INPUT.samt REFERENCE.samt\n";
        return 2;
    }
    char *end = nullptr;
    const unsigned long device = std::strtoul(argv[3], &end, 10);
    if (end == argv[3] || *end || device > UINT32_MAX) return 2;
    auto backend = sam3d::Backend::create(argv[1], argv[2], static_cast<uint32_t>(device), 8);
    if (!backend) return 1;
    sam3d::GGUFModel model;
    if (!model.load(argv[4], backend->weights_buffer_type())) return 1;

    sam3d::RawTensor input, reference;
    if (!sam3d::load_raw_tensor(argv[5], input) ||
        !sam3d::load_raw_tensor(argv[6], reference) ||
        input.type != GGML_TYPE_F32 || reference.type != GGML_TYPE_F32 ||
        input.ne != std::vector<int64_t>({16, 16, 16, 8}) ||
        reference.ne != std::vector<int64_t>({64, 64, 64})) {
        std::cerr << "invalid decoder parity fixture contract\n";
        return 1;
    }
    sam3d::GraphContext context;
    ggml_tensor *latent = context.input_f32("ss_latent", {16, 16, 16, 8});
    sam3d::SsDecoderGraph decoder{context.ctx(), &model};
    ggml_tensor *output = decoder.build(latent);
    ggml_cgraph *graph = ggml_new_graph_custom(context.ctx(), 16384, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    if (!backend->alloc(graph) ||
        !backend->set_input_f32(latent, reinterpret_cast<const float *>(input.data.data()),
                                input.data.size() / sizeof(float)) ||
        !backend->run(graph)) return 1;
    std::vector<float> actual;
    if (!backend->get_tensor_f32(output, actual) || actual.size() * sizeof(float) != reference.data.size()) {
        std::cerr << "decoder output size mismatch\n";
        return 1;
    }
    const float *expected = reinterpret_cast<const float *>(reference.data.data());
    double error2 = 0.0, reference2 = 0.0;
    float max_abs = 0.0f;
    size_t sign_matches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            std::cerr << "non-finite decoder output\n";
            return 1;
        }
        const double delta = static_cast<double>(actual[i]) - expected[i];
        max_abs = std::max(max_abs, static_cast<float>(std::abs(delta)));
        error2 += delta * delta;
        reference2 += static_cast<double>(expected[i]) * expected[i];
        sign_matches += (actual[i] > 0.0f) == (expected[i] > 0.0f);
    }
    const double rel_l2 = std::sqrt(error2 / std::max(reference2, std::numeric_limits<double>::min()));
    const double sign_agreement = static_cast<double>(sign_matches) / actual.size();
    std::cout << "backend=" << backend->backend_name()
              << " max_abs=" << max_abs
              << " rel_l2=" << rel_l2
              << " sign_agreement=" << sign_agreement << '\n';
    // Frozen from the independently captured F32 source fixture before this
    // integration: preserve every occupancy decision and bound accumulated
    // decoder error separately. Vulkan receives its own measured policy once
    // the first successful graph is captured.
    return max_abs <= 0.04f && rel_l2 <= 2e-4 && sign_agreement == 1.0 ? 0 : 1;
}
