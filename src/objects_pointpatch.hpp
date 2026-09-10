#pragma once
#include "dino_block.hpp"
#include <functional>
namespace sam3d {
// PointPatchEmbed eval contract; explicit synthetic/trained state, no pointmap
// inference or SSI normalization. Nonfinite input triples select invalid tokens.
enum class point_remapping:uint32_t {linear,sinh,exp,sinh_exp,exp_disparity};
struct pointpatch_shape {
    uint32_t batch,height,width,side=256,patch=8,dim=768;
    point_remapping remap=point_remapping::exp;
    bool dropout_enabled=false,force_dropout=false;
};
void validate_pointpatch_shape(pointpatch_shape);
std::vector<std::pair<std::string,uint64_t>> pointpatch_parameter_sizes(pointpatch_shape);
// Each tap is window-major. Chunks cover [offset,offset+values.size()) of a
// tensor with total F32 elements. Allows complete captures without retaining
// every full-resolution intermediate in RAM. Callbacks must copy/consume data.
using pointpatch_observer=std::function<void(const std::string &,std::span<const float>,uint64_t offset,uint64_t total)>;
std::vector<float> objects_pointpatch(neural_session &,pointpatch_shape,
    std::span<const float> xyz_chw,std::span<const uint8_t> valid_mask,
    const named_floats &,const pointpatch_observer &observer={},uint32_t chunk_windows=32);
}
