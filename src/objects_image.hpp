#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>
namespace sam3d {
// Internal original-default Objects image/mask preprocessing, not a model API.
// CPU preprocessing is also used before Vulkan inference, as in upstream before
// its device transfer. The default-RGBA entry has no pointmap or normalization;
// the explicit joint entry below does not infer a checkpoint configuration.
struct objects_image_options {
    uint32_t output_side=518;
    double box_size_factor=1.0;
    double padding_factor=0.1;
};
using objects_image_taps=std::map<std::string,std::vector<float>>;
void validate_objects_image_options(objects_image_options);
// RGBA U8 with alpha > 0 selecting the object, row stride in bytes. Final names
// 08.image/08.rgb_image are [1,3,S,S]; masks [1,1,S,S], CHW, no ImageNet norm.
// Diagnostic intermediates remain owned by the returned map.
objects_image_taps objects_prepare_rgba(std::span<const uint8_t>,uint32_t width,
    uint32_t height,uint64_t row_stride,objects_image_options={});
struct objects_joint_result {
    uint32_t height,width;
    std::vector<float> rgb,mask,pointmap;
    objects_image_taps taps;
};
// Explicit original resize_all_to_same_size -> crop -> rembg triple chain.
// Soft RGBA alpha is preserved. XYZ is supplied CHW, not estimated or SSI-normalized
// here. No implicit claim about which transforms a published config selects.
objects_joint_result objects_prepare_pointmap_joint(std::span<const uint8_t> rgba,
    uint32_t width,uint32_t height,uint64_t row_stride,std::span<const float> xyz,
    uint32_t point_height,uint32_t point_width,double box_factor,double padding,bool observe=false);
using objects_tensor_observer=std::function<void(const std::string &,std::span<const float>)>;
// Explicit pad-to-square and bicubic-AA (RGB) or nearest (mask/XYZ) transform.
std::vector<float> objects_square_resize(std::span<const float>,uint32_t channels,
    uint32_t height,uint32_t width,uint32_t side,bool bicubic,bool nan_padding,
    const objects_tensor_observer &observer={});
// Direct CHW resize used at the MoGe model boundary. This follows PyTorch's
// antialiased bicubic or non-antialiased bilinear align_corners=false rules
// without first padding the image to a square.
std::vector<float> objects_resize_chw(std::span<const float>,uint32_t channels,
    uint32_t height,uint32_t width,uint32_t output_height,uint32_t output_width,
    bool bicubic_antialias);
}
