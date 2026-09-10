#pragma once
#include "objects_image.hpp"
#include <array>
namespace sam3d {
enum class objects_ssi_mode:uint32_t {
    basic,object_scene,object_quantile,object_norm_median,
    apparent_scene,apparent_object,disparity_scene,disparity_object
};
struct objects_ssi_shape {uint32_t height,width,mask_height,mask_width;};
struct objects_ssi_options {
    objects_ssi_mode mode=objects_ssi_mode::basic;
    double quantile_drop=.1,clip=0,scale_factor=1,log_disparity_shift=0;
    bool allow_override=false,raise_on_no_valid_points=false;
};
struct objects_ssi_result {
    std::vector<float> pointmap; // [3,H,W], NaN triples mark invalid/clipped points.
    std::array<float,3> scale,shift;
    objects_image_taps taps;
};
void validate_objects_ssi(objects_ssi_shape,objects_ssi_options);
// Mask is [1,MH,MW], finite soft values in [0,1]. Overrides empty or 3 floats.
// The original normalizer's per-mode override rules are preserved, not unified.
objects_ssi_result objects_normalize_pointmap(objects_ssi_shape,objects_ssi_options,
    std::span<const float> xyz_chw,std::span<const float> mask,
    std::span<const float> scale_override={},std::span<const float> shift_override={},bool observe=false);
// Consumes a native-produced normalized map and its own scale/shift. Optional
// taps expose the original homogeneous-transform boundary before inverse remap.
std::vector<float> objects_denormalize_pointmap(uint32_t height,uint32_t width,
    objects_ssi_mode,std::span<const float> xyz_chw,std::span<const float> scale,
    std::span<const float> shift,objects_image_taps *taps=nullptr);
}
