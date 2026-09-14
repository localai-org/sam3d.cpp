// DINOv2 ViT-L/14 with register tokens (condition embedder RGB / mask backbones).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph_builder.hpp"
#include "gguf_loader.hpp"

namespace sam3d {

// Build a DINOv2 forward graph. `img` is an HWC F32 input (channel fastest:
// ggml ne = [3, W, H]) already resized to 518x518; the graph applies the
// Dino wrapper's per-channel normalization internally. The output tokens are
// (C, 1 + n_patch) ggml ne order: [cls token, patch tokens] after the final
// LayerNorm, with register tokens dropped (matches Dino.forward).
struct DinoGraph {
    GraphContext* g = nullptr;         // owns graph inputs (tables) + ctx
    const GGUFModel* m = nullptr;
    ggml_context* ctx() const { return g->ctx(); }
    std::string prefix = "cemb.emb0";  // per-embedder weight/KV prefix
    std::string debug_stage;           // post_patch | post_embed | block<N>
    // prenorm output mode (SLat embedders, prenorm_features: true): return
    // ALL 1374 tokens (cls + registers + patches) normalized by an affine-free
    // LayerNorm(eps 1e-5) of the pre-final-norm features, instead of the
    // register-dropped final-norm 1370.
    bool prenorm = false;
    std::vector<ggml_tensor*> inputs;  // graph inputs (patch gather table)
    std::vector<std::shared_ptr<std::vector<int32_t>>> table_data;

    ggml_tensor* build(ggml_tensor* img);
};

}  // namespace sam3d
