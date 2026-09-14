// SLat flow model (slat_generator backbone, dit.*) - single forward step.
// Structure (SLatFlowModel): SparseLinear(8->128) -> 2 sparse res blocks
// (SubM conv 3^3, 2x mean-pool downsample into the 1024-channel grid) ->
// AbsolutePositionEmbedder -> 24 dense-attention DiT blocks (qk rms norm,
// cross attention to the fused condition) -> 2 sparse res blocks with skip
// concatenation and 2x nearest upsample -> LayerNorm -> SparseLinear(128->8).
// The sparse structure (coords) is fixed for the whole sampling loop, so all
// gather tables live in SlatTables (host, built once).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph_builder.hpp"
#include "gguf_loader.hpp"
#include "sparse_ops.hpp"

namespace sam3d {

struct SlatFlowGraph {
    GraphContext* g = nullptr;
    const GGUFModel* m = nullptr;
    const SlatTables* tb = nullptr;   // host tables (built by the caller)
    std::string debug_stage;          // input_layer | ib0_conv2 | ib1_conv1 |
                                      // ib1_conv2 | ape | block0 | block23 |
                                      // ob0_conv1 | ob1_conv1 | ob1_conv2
    std::string prefix = "dit";
    int64_t n_cond_tokens = 5496;     // host sets from the condition SAMT

    ggml_tensor* x = nullptr;         // (8, nf) input latents
    ggml_tensor* cond = nullptr;      // (1024, n_cond) condition tokens
    ggml_tensor* t = nullptr;         // (1, 1) scaled timestep
    std::vector<ggml_tensor*> inputs;
    std::vector<std::shared_ptr<std::vector<int32_t>>> table_data;
    std::vector<ggml_tensor*> mask_inputs;  // unused; kept for CLI parity
    // Optional post-run diagnostics. These are deliberately not graph outputs,
    // so recording them does not change graph allocation or execution.
    std::vector<ggml_tensor*> debug_block_outputs;

    std::vector<ggml_tensor*> build();
};

}  // namespace sam3d
