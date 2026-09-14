// SparseStructureDecoder: dense 3D-conv U-Net mapping the 8x16^3 sparse
// structure latent to a 1x64^3 occupancy field.
#pragma once

#include "gguf_loader.hpp"

#include <vector>

namespace sam3d {

struct SsDecoderGraph {
    ggml_context* ctx = nullptr;
    const GGUFModel* m = nullptr;
    std::vector<ggml_tensor*> debug_tensors;  // valid after compute

    // returns the raw logits field (1, 64, 64, 64)
    ggml_tensor* build(ggml_tensor* latent);
};

}  // namespace sam3d
