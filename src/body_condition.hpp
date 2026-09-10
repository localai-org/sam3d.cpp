#pragma once
#include "body_prompt.hpp"

namespace sam3d {
struct condition_shape {
    uint32_t batch,height,width,patch,context_dim,token_dim,points,joints,pose_dim,keypoints;
    bool keypoints3d,hand_tokens;
};
void validate_condition_shape(condition_shape);
uint32_t condition_token_count(condition_shape);
std::vector<std::pair<std::string,uint64_t>> condition_parameter_sizes(condition_shape);
// Original DINO/CLIFF Body forward_decoder input construction. Input features
// are NCHW, rays B,2,H,W, condition B,3, prompts B,P,3. An empty previous
// estimate selects the learned initial pose+camera. No external initial estimate
// and no alternate ViT positional cropping in this first Body path.
// Outputs include NCHW image/image PE, B,N,D tokens/token PE. The decoder mask
// is deliberately absent, even for invalid prompts, matching original code.
// images_channels_last returns image/image PE as B,N,D / 1,N,D to avoid
// converting GPU token layouts to NCHW only for the decoder to undo it.
named_floats body_condition(neural_session &,condition_shape,
    std::span<const float> features,std::span<const float> rays,
    std::span<const float> cliff,std::span<const float> keypoints,
    std::span<const float> previous,const weight_map &parameters,
    scalar_division = scalar_division::direct,std::span<const float> dense_gaussian = {},bool images_channels_last = false);
}
