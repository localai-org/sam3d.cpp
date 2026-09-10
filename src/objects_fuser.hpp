#pragma once
#include "dino_block.hpp"
namespace sam3d {
struct fuser_input {uint32_t embedder,tokens;int32_t position=-1;bool forced_drop=false;};
struct fuser_shape {
    uint32_t batch=1;
    std::vector<uint32_t> embed_dims;
    std::vector<fuser_input> inputs; // Original embedder/kwargs iteration order.
    bool pre_norm=true,random_position=false;
    double projection_multiplier=4,compression_multiplier=0;
};
void validate_fuser_shape(const fuser_shape &);
std::vector<std::pair<std::string,uint64_t>> fuser_parameter_sizes(const fuser_shape &);
// Original EmbedderFuser POST-ENCODER eval path. Inputs are native/supplied
// [B,N,D] embeddings, not raw pixels. Shared embedder projections remain shared.
// Position IDs follow first-encounter group order; -1 means no position add.
// Forced modality drops occur AFTER projection/position, before concatenation.
// No training dropout or implicit encoder/config selection. Output: 90.output.
named_floats objects_fuse(neural_session &,const fuser_shape &,
    const std::vector<std::vector<float>> &embeddings,const named_floats &parameters,
    bool capture_all=true);
}
