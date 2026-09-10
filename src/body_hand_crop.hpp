#pragma once
#include "body_pipeline.hpp"
namespace sam3d {
// _get_hand_box plus full-mode left mirroring and padding=0.9 preparation.
// Input boxes are normalized crop-relative center_x,center_y,width,height,
// left then right. This is preprocessing only, not hand inference or merging.
named_floats body_hand_boxes(std::span<const float> normalized_boxes,
    std::span<const float> body_affine,uint32_t image_width,uint32_t crop_size=512);
named_floats body_prepare_hands(std::span<const uint8_t> rgb,uint32_t width,uint32_t height,
    uint64_t stride,std::span<const float> normalized_boxes,std::span<const float> body_affine,
    std::span<const float> intrinsics,uint32_t crop_size=512);
}
