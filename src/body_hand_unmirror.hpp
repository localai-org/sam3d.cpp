#pragma once
#include "body_pose.hpp"
namespace sam3d {
// Original run_inference left-hand output conversion, not final refinement.
// Inputs remain untouched; all four returned arrays own their storage.
named_floats body_unmirror_left(uint32_t batch,uint32_t image_width,
    std::span<const float> scale,std::span<const float> joint_global_rotations,
    std::span<const float> hand,std::span<const float> box_center,
    std::span<const float> body_scale_mean,std::span<const float> body_scale_components);
}
