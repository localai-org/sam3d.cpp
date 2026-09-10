#pragma once
#include "dino_block.hpp"

namespace sam3d {
struct prompt_shape { uint32_t batch, points, dim, joints, height, width; };
// PyTorch CPU scalar division is direct; CUDA's scalar fast path multiplies
// by an F32 reciprocal. Explicit so composition can preserve its chosen oracle.
enum class scalar_division { direct, reciprocal_multiply };
void validate_prompt_shape(prompt_shape);
std::vector<std::pair<std::string,uint64_t>> prompt_parameter_sizes(prompt_shape);
// Body PromptEncoder without mask-convolution variants. Normalized keypoints
// are B,N,3 (x,y,label); labels are integral [-2,joints). At least one point;
// baseline Body uses the invalid [0,0,-2] prompt, not an absent prompt tensor.
// Dense grid is 1,D,H,W; sparse embedding B,N,D; mask is F32 B,N as upstream.
// Optional dense_gaussian selects independent image PE for the hand branch;
// it never replaces the shared sparse prompt encoder's own Gaussian.
// dense_channels_last selects 22.dense_tokens [1,H*W,D], without a transpose,
// instead of 22.dense_nchw. Sparse outputs and numerical operations are unchanged.
named_floats body_prompt_encode(neural_session &,prompt_shape,
    std::span<const float> keypoints,const weight_map &parameters,
    scalar_division = scalar_division::direct,std::span<const float> dense_gaussian = {},bool capture_all = true,bool dense_channels_last = false);
// Original PositionEmbeddingRandom.forward_with_coords contract. Pixel xy
// points need not lie inside the image. Matrix is stored original [2,D/2].
named_floats body_position_pixels(neural_session &,uint32_t batch,uint32_t points,
    uint32_t dim,uint32_t image_height,uint32_t image_width,
    std::span<const float> xy,std::span<const float> gaussian,
    scalar_division = scalar_division::direct);
}
