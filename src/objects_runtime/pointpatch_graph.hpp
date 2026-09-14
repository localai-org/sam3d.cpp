// PointPatchEmbed (the condition embedder's pointmap backbone, cemb.emb2).
// Maps the checkpoint's PointPatchEmbed:
//   nearest-resize pointmap to input_size -> point_proj(3->D) per pixel
//   (invalid pixels -> invalid_xyz_token) -> split 8x8 windows -> [cls | 64]
//   + pos_embed_window -> 1 transformer block (erf GELU, 16 heads)
//   -> per-window CLS tokens + pos_embed_patch
// Tokens follow the repo convention: (D, n_windows) ggml ne order.
//
// Input pointmap layout: ggml ne = [W, H, 3] (torch CHW C-order memory:
// W fastest) - note this is the transpose of the image input of DinoGraph.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph_builder.hpp"
#include "gguf_loader.hpp"

namespace sam3d {

struct PointPatchGraph {
    GraphContext* g = nullptr;
    const GGUFModel* m = nullptr;
    ggml_context* ctx() const { return g->ctx(); }
    std::string prefix = "cemb.emb2";
    std::string debug_stage;           // post_proj | post_block
    std::vector<ggml_tensor*> inputs;  // gather tables + valid mask
    std::vector<std::shared_ptr<std::vector<int32_t>>> table_data;
    // host-side masks uploaded after alloc (same order as mask_inputs)
    std::vector<ggml_tensor*> mask_inputs;

    ggml_tensor* build(ggml_tensor* pm);
};

}  // namespace sam3d
