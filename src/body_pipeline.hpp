#pragma once
#include "body_flow.hpp"
#include "dino_backbone.hpp"
namespace sam3d {
// Internal composition, not a public trained-model session API. Models/parameters
// remain explicit. One person, RGB pixels already decoded by the caller, square
// DINO crop, explicit bbox and pinhole intrinsics; no detector/full refinement.
// Hand mode takes an already mirrored image/box for a left hand. It is a hand
// image branch, not body-derived crop selection, unmirroring or final merging.
struct body_pipeline_shape {backbone_shape backbone;body_flow_shape decoder;bool trained_branch=false;};
void validate_body_pipeline_shape(body_pipeline_shape);
std::vector<std::pair<std::string,uint64_t>> body_pipeline_parameter_sizes(body_pipeline_shape);
named_floats body_hand_detection(neural_session &,uint32_t dim,std::span<const float> tokens,const weight_map &);
named_floats body_prepare_rgb(std::span<const uint8_t> rgb,uint32_t width,uint32_t height,
    uint64_t row_stride,std::span<const float> box,std::span<const float> intrinsics,
    uint32_t crop_size,bool intrinsics_center,float padding=1.25f);
named_floats body_from_rgb(neural_session &,tensor_archive &mhr,body_pipeline_shape,
    std::span<const uint8_t> rgb,uint32_t width,uint32_t height,uint64_t row_stride,
    std::span<const float> box,std::span<const float> intrinsics,
    std::span<const float> prompts,std::span<const float> previous,
    const parameter_reader &backbone_weights,std::span<const int32_t> hand_indices,
    const weight_map &decoder_weights,scalar_division=scalar_division::direct,
    std::span<const int32_t> nonhand_indices={},dino_resident_stack *resident=nullptr,bool bf16=false);
}
