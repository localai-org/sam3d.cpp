// MoGe ViT-L graph used by the official SAM 3D Objects point-map pipeline.
//
// This owns the neural component only: caller supplies an RGB image at the
// official MoGe resized resolution.  The graph performs DINO normalization,
// 14-pixel crop, ViT intermediate feature extraction, point/mask head and
// the official exp point remapping.  Keeping the resize policy outside this
// graph makes it independently comparable with MoGeModel.forward().
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph_builder.hpp"
#include "gguf_loader.hpp"

namespace sam3d {

struct MogeOutputs {
    ggml_tensor* points = nullptr;  // [W, H, 3, 1], camera-space before shift
    ggml_tensor* mask_logits = nullptr; // [W, H, 1, 1]
    ggml_tensor* backbone_input = nullptr; // [C, 1 + patches], before block 0
    ggml_tensor* debug_q = nullptr; // block 0 [D, N, H], parity instrumentation
    ggml_tensor* debug_k = nullptr;
    ggml_tensor* debug_v = nullptr;
    ggml_tensor* debug_attention_context = nullptr;
    std::vector<ggml_tensor*> backbone_attention_outputs; // after attention projection and LayerScale
    std::vector<ggml_tensor*> backbone_mlp_fc1_outputs; // before exact GELU
    std::vector<ggml_tensor*> backbone_mlp_gelu_outputs; // after exact GELU
    std::vector<ggml_tensor*> backbone_mlp_outputs; // after MLP and LayerScale
    std::vector<ggml_tensor*> backbone_block_outputs; // token tensors after each DINO block
    // Last DINO feature maps after the official final norm and class-token
    // removal. Kept for the native parity harness; regular inference does
    // not materialize them on the host.
    std::vector<ggml_tensor*> backbone_features;
    ggml_tensor* projected_features = nullptr;
    std::vector<ggml_tensor*> upsample_outputs;
    // Third upsample block boundaries: UV concat, transpose convolution,
    // replicate convolution, then each residual block.
    std::vector<ggml_tensor*> third_upsample_stages;
    ggml_tensor* third_transpose_weight_f32 = nullptr;
    ggml_tensor* output_block_input = nullptr;
    std::vector<ggml_tensor*> output_branch_hidden;
    std::vector<ggml_tensor*> output_branch_raw;
};

struct MogeGraph {
    GraphContext* g = nullptr;
    const GGUFModel* m = nullptr;
    std::string prefix = "moge";
    std::vector<ggml_tensor*> inputs;
    std::vector<std::shared_ptr<std::vector<float>>> f32_data;
    std::vector<std::shared_ptr<std::vector<int32_t>>> i32_data;
    // MoGe first resizes the image to its token budget, then resizes the raw
    // head outputs back to the source image before applying the exp remap.
    // Zero retains the model-input extent for stage-level diagnostics.
    int64_t output_width = 0;
    int64_t output_height = 0;

    // img has HWC storage / ggml shape [3, W, H].  It must have the exact
    // integer resized dimensions selected by MoGeModel.forward().
    MogeOutputs build(ggml_tensor* img);
};

}  // namespace sam3d
