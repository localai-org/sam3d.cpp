#pragma once
#include "dino_block.hpp"
namespace sam3d {
struct pose_shape { uint32_t batch,dim,hidden,depth; };
// Internal borrowed views of the separate hand head's checkpoint buffers.
// This is not a public C ABI struct. Null config selects the body head.
struct hand_pose_config {
    std::span<const float> local_to_world,wrist,root;
    std::span<const int32_t> nonhand_indices;
};
void validate_pose_shape(pose_shape);
// Isolated diagnostic for the original global 6D -> quaternion -> Euler path.
// Accepts B,6 and captures intermediate arithmetic, without any neural weights.
named_floats body_global_rotation(uint32_t batch,std::span<const float> rotation6d);
// Original RoMa matrix -> normalized quaternion -> extrinsic xyz Euler.
named_floats body_matrix_rotation_xyz(uint32_t batch,std::span<const float> matrices);
std::vector<std::pair<std::string,uint64_t>> pose_parameter_sizes(pose_shape);
// Body-mode MHRHead prefix, through the inputs to MHR itself. Not mesh decoding.
// Rotations/trigonometry and scatter are checked CPU geometry, even for Vulkan.
// Hand indices address the 136-value [translation, global Euler, body] layout.
named_floats body_pose(neural_session &,pose_shape,std::span<const float> token,
    std::span<const float> initial,std::span<const int32_t> hand_indices,
    const weight_map &parameters,const hand_pose_config *hand=nullptr);
}
