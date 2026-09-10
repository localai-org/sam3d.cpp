#pragma once
#include "dino_block.hpp"
namespace sam3d {
struct camera_head_shape {
    uint32_t batch,dim,hidden,depth,points;
    float scale_factor;
    bool intrinsics_center;
};
void validate_camera_head_shape(camera_head_shape);
std::vector<std::pair<std::string,uint64_t>> camera_head_parameter_sizes(camera_head_shape);
// PerspectiveHead FFN + optional initial-estimate residual + full-perspective
// projection, all F32 GGML. Token B,D; initial empty or B,3; points B,N,3;
// center/image_size B,2; box_size B; intrinsics B,3,3. Points must already be
// in MHRHead's output coordinate convention. Does not generate MHR geometry.
named_floats body_camera_head(neural_session &,camera_head_shape,
    std::span<const float> token,std::span<const float> initial,
    std::span<const float> points,std::span<const float> box_center,
    std::span<const float> box_size,std::span<const float> image_size,
    std::span<const float> intrinsics,const weight_map &parameters);
// Project another point set using the already predicted B,3 camera. No FFN
// evaluation or residual addition: needed for the per-layer vertex projection.
named_floats body_camera_project(neural_session &,camera_head_shape,
    std::span<const float> camera,std::span<const float> points,
    std::span<const float> box_center,std::span<const float> box_size,
    std::span<const float> image_size,std::span<const float> intrinsics);
}
