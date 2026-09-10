#pragma once
#include "objects_ssi.hpp"
namespace sam3d {
struct objects_preprocess_options {
    uint32_t image_side=518,point_side=256;
    double box_factor=1,padding=.1;
    bool normalize=true,point_nan_padding=true;
    objects_ssi_options object_normalizer,full_normalizer;
};
void validate_objects_preprocess_options(objects_preprocess_options);
// Complete original preprocessing METHOD for this explicit transform configuration.
// Supplied CHW XYZ is not MoGe output produced by this function. Preprocessing is
// CPU; the caller can transfer resulting F32 fields to the selected GGML backend.
// 11 named output fields follow InferencePipelinePointMap.preprocess_image.
// F32 buffers flatten the original batch-one CHW tensors; moments have 3 values.
// Observer spans are borrowed only for the synchronous callback's duration.
objects_image_taps objects_preprocess_pointmap(std::span<const uint8_t> rgba,
    uint32_t width,uint32_t height,uint64_t stride,std::span<const float> xyz,
    uint32_t point_height,uint32_t point_width,objects_preprocess_options,
    const objects_tensor_observer &observer={});
}
