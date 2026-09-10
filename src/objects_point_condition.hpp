#pragma once
#include "objects_preprocess.hpp"
#include "objects_pointpatch.hpp"
#include "objects_fuser.hpp"
namespace sam3d {
enum class point_condition_field:uint32_t {object,full,unnormalized_full};
struct point_condition_shape {
    objects_preprocess_options preprocessing;
    pointpatch_shape encoder{1,256,256};
    fuser_shape fusion;
    std::vector<point_condition_field> fields;
};
struct point_condition_result {
    objects_image_taps prepared;
    std::vector<std::vector<float>> embeddings;
    std::vector<float> conditioning;
};
void validate_point_condition_shape(const point_condition_shape &);
// Raw RGBA + supplied XYZ -> native preprocessing -> shared PointPatch encoder
// -> native fusion. Explicit point-only conditioning configuration, NOT the
// complete published image/point conditioner or MoGe inference. No reference
// intermediates accepted. Batch one; observer spans are synchronous/borrowed.
point_condition_result objects_point_condition(neural_session &,const point_condition_shape &,
    std::span<const uint8_t> rgba,uint32_t width,uint32_t height,uint64_t stride,
    std::span<const float> xyz,uint32_t point_height,uint32_t point_width,
    const named_floats &encoder_parameters,const named_floats &fusion_parameters,
    const pointpatch_observer &observer={},uint32_t chunk_windows=32);
}
