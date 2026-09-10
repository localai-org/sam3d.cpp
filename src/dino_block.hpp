#pragma once
#include "neural.hpp"
#include <map>
#include <utility>

namespace sam3d {
struct dino_shape { uint32_t batch, height, width, dim, heads, prefix, hidden; };
using named_floats = std::map<std::string, std::vector<float>>;
using named_weights = std::map<std::string, validated_weights>;
// Parameter subsets copy ownership handles, never the underlying float arrays.
// Ordinary diagnostic maps enter through an owned, checked snapshot exactly once.
class weight_map : public named_weights {
public:
    weight_map()=default;
    weight_map(named_floats values){
        for(auto &[name,value]:values)emplace(name,validated_weights::checked(std::move(value)));
    }
};
void validate_dino_shape(dino_shape);
std::vector<std::pair<std::string,uint64_t>> dino_parameter_sizes(dino_shape);
// Shared graph equations for the block oracle and resident backbone. The caller
// owns all tensor storage; angles and parameters are immutable model inputs.
using named_tensors = std::map<std::string,ggml_tensor *>;
std::vector<float> dino_angles(dino_shape,std::span<const float> periods);
named_tensors dino_block_graph(neural_session &,ggml_context *,dino_shape,
    ggml_tensor *input,const named_tensors &parameters,ggml_tensor *angles,bool capture_all,bool bf16=false);
// Body ViT-H+ block contract: F32, LayerNorm eps=1e-5, key bias mask,
// eval-only axial RoPE, LayerScale and SwiGLU. Inputs are [B,prefix+HW,D].
// periods is persistent model state, not injected reference activations.
named_floats dino_block(neural_session &, dino_shape, std::span<const float> input,
                       const named_floats &parameters, bool capture_all = true);
named_floats dino_block(neural_session &, dino_shape, std::span<const float> input,
                       const named_weights &parameters, bool capture_all = true);
named_floats dino_block_bf16(neural_session &,dino_shape,std::span<const float>,const named_weights &,bool capture_all=true);
}
