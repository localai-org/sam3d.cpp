#pragma once
#include "body_pose.hpp"
#include "body_prompt.hpp"
#include "mhr_geometry.hpp"
namespace sam3d {
// MHRHead body-mode output convention: vertices/keypoints/joint coordinates in
// meters with Y/Z flipped; joint rotation matrices retain original MHR axes.
// Vertex count is variable for isolated regression; full Body uses 18439.
named_floats body_map_geometry(neural_session &,uint32_t batch,uint32_t vertices,
    std::span<const float> vertices_cm,std::span<const float> skeleton,
    const validated_weights &mapping,scalar_division=scalar_division::direct,bool hand=false);
// Composed pose head -> actual MHR -> output mapping, own intermediates. No
// trained defaults: all head/PCA/index/keypoint state must be supplied explicitly.
named_floats body_pose_geometry(neural_session &,tensor_archive &,pose_shape,
    std::span<const float> token,std::span<const float> initial,
    std::span<const int32_t> hand_indices,const weight_map &parameters,
    const validated_weights &mapping,scalar_division=scalar_division::direct,const hand_pose_config *hand=nullptr);
}
