#pragma once
#include "body_pose.hpp"
namespace sam3d {
named_floats body_hand_frame(uint32_t batch,std::span<const float> rotation_xyz,
    std::span<const float> translation,std::span<const float> local_to_world,
    std::span<const float> wrist,std::span<const float> root);
std::vector<float> body_hand_mask_parameters(uint32_t batch,std::span<const float> model_parameters,std::span<const int32_t> nonhand_indices);
std::vector<float> body_hand_mask_keypoints(uint32_t batch,std::span<const float> keypoints308);
}
