#pragma once
#include "dino_block.hpp"
namespace sam3d {
named_floats body_full_to_crop(neural_session &,uint32_t batch,uint32_t points,
    std::span<const float> pixels,std::span<const float> affine,std::span<const float> crop_size);
struct feedback_shape {
    uint32_t batch,tokens,dim,context_dim,height,width,points,keypoints,keypoints3d;
    uint32_t start2d,start3d,hip_left,hip_right,layer,depth;
};
void validate_feedback_shape(feedback_shape);
std::vector<std::pair<std::string,uint64_t>> feedback_parameter_sizes(feedback_shape);
// DINO Body callbacks after pose/camera projection. Image NCHW, tokens/augment
// B,T,D, full-image points B,J,2, depth B,J, world points B,J,3, affine B,2,3,
// crop size B,2. Index lists select source keypoints after pelvis normalization.
// No alternate ViT horizontal-grid rescaling. Final layer is a strict no-op
// for tokens/augment, as upstream (crop coordinates can still be observed).
// Internal channels-last option accepts B,H,W,C image storage. This changes
// only addressing, not sample coordinates, corner accumulation or masking.
named_floats body_feedback(neural_session &,feedback_shape,
    std::span<const float> image,std::span<const float> tokens,std::span<const float> augment,
    std::span<const float> pixels,std::span<const float> depth,std::span<const float> world,
    std::span<const float> affine,std::span<const float> crop_size,
    std::span<const int32_t> indices2d,std::span<const int32_t> indices3d,
    const weight_map &parameters,bool capture_all = true,bool image_channels_last = false);
}
