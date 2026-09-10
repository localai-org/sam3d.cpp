#pragma once
#include "body_condition.hpp"
#include "body_camera_head.hpp"
#include "body_output.hpp"
namespace sam3d {
struct body_flow_shape {
    condition_shape condition;
    uint32_t depth,heads,head_dim,hidden,pose_hidden,pose_depth,camera_hidden,camera_depth;
    float camera_scale;
    bool repeat_pe,twoway,intrinsics_center;
    bool capture_decoder_operations=false;
    bool hand_branch=false;
    bool capture_boundaries=true;
};
void validate_body_flow_shape(body_flow_shape);
std::vector<std::pair<std::string,uint64_t>> body_flow_parameter_sizes(body_flow_shape);
// Explicit original checkpoint namespace: only independently parameterized
// modules receive _hand; prompt encoder/projection and hand-box tokens are shared.
std::string body_flow_parameter_name(const std::string &,bool hand_branch);
std::vector<float> body_decoder_norm(neural_session &,uint32_t batch,uint32_t tokens,uint32_t dim,
    std::span<const float> values,std::span<const float> weight,std::span<const float> bias);
// Complete DINO Body forward_decoder composition with intermediate MHR pose,
// camera, vertex projection and 2D/3D feedback. Trained state is explicit; this
// starts at backbone features, NOT images. Default 70-keypoint indexing only;
// no external initial estimate, full hand refinement or alternate ViT.
// hand_branch selects original *_hand modules, independent dense image PE and
// the wrist-centric MHR head; requires the explicit 145-entry nonhand mask.
// Returns diagnostic per-layer tensors, final normalized token output (or the
// upstream hand-detection slice), and every layer's complete Body tensor output.
named_floats body_forward_decoder(neural_session &,tensor_archive &,body_flow_shape,
    std::span<const float> features,std::span<const float> rays,std::span<const float> cliff,
    std::span<const float> prompts,std::span<const float> previous,
    std::span<const float> box_center,std::span<const float> box_size,
    std::span<const float> image_size,std::span<const float> intrinsics,
    std::span<const float> affine,std::span<const float> crop_size,
    std::span<const int32_t> hand_indices,const weight_map &parameters,
    scalar_division=scalar_division::direct,std::span<const int32_t> nonhand_indices={});
}
