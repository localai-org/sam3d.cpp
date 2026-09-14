// MOT sparse-structure DiT (ss_generator backbone, dit.*) - single forward
// step. The ShortCut sampling loop (noise, Euler integration, CFG blending)
// lives host-side in the session; this graph computes one velocity evaluation
// for the 5-modality latent dict:
//   shape (4096, 8) | 6drotation (1, 6) | scale (1, 3) | translation (1, 3) |
//   translation_scale (1, 1)
// with the pose modalities merged into a single 4-token group that attends
// the shape tokens (which are "protected": they only attend themselves).
//
// Inputs (all graph inputs, so one graph build serves all 25 Euler steps and
// both CFG branches):
//   x_shape (8, 4096), x_pose_* (ch, 1), t (1, 1), d (1, 1), cond (7528, 1024)
// CFG is expressed by feeding zeroed condition tokens (force_zeros_cond).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph_builder.hpp"
#include "gguf_loader.hpp"

namespace sam3d {

struct SsFlowGraph {
    GraphContext* g = nullptr;
    const GGUFModel* m = nullptr;
    ggml_context* ctx() const { return g->ctx(); }
    std::string debug_stage;  // proj_in | block<N> | t_emb
    std::string prefix = "dit";
    int64_t n_cond_tokens = 7528;  // host sets from the cond SAMT shape
    std::vector<ggml_tensor*> inputs;
    std::vector<std::shared_ptr<std::vector<int32_t>>> table_data;
    std::vector<ggml_tensor*> mask_inputs;

    // host tables (freq tables for timestep embedding)
    std::shared_ptr<std::vector<int32_t>> freq_data;
    ggml_tensor* t = nullptr;      // (1, 1) F32 scalar timestep (scaled)
    ggml_tensor* d = nullptr;      // (1, 1) F32 shortcut step (scaled)
    ggml_tensor* cond = nullptr;   // (cond_ch, n_cond) F32 tokens
    ggml_tensor* x_shape = nullptr;
    ggml_tensor* x_6drot = nullptr;
    ggml_tensor* x_scale = nullptr;
    ggml_tensor* x_trans = nullptr;
    ggml_tensor* x_ts = nullptr;

    // build the graph; returns the 5 velocity outputs (order: 6drotation,
    // scale, shape, translation, translation_scale - the ckpt dict order)
    std::vector<ggml_tensor*> build();
};

}  // namespace sam3d
