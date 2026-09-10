#pragma once
#include "dino_block.hpp"
#include "tensor_archive.hpp"

namespace sam3d {
// Released LOD1 MHR contract. Internal API, cm / XYZW / scale. No mesh yet.
struct mhr_skeleton_data {
    std::vector<float> offsets, prerotations;
    std::vector<int32_t> prefix, parents;
};
struct mhr_skeleton_taps {
    named_floats f32;
    std::map<std::string,std::vector<double>> f64;
};
mhr_skeleton_data load_mhr_skeleton(tensor_archive &);
// Checked CPU geometry, including original F64 prefix arithmetic even when the
// preceding projection runs on Vulkan. Not a GPU-resident geometry claim.
mhr_skeleton_taps mhr_local_skeleton(uint32_t batch,std::span<const float> joint_parameters,
                                    const mhr_skeleton_data &);
// Own parameter projection and all subsequent local/FK intermediates.
mhr_skeleton_taps mhr_skeleton(neural_session &,tensor_archive &,uint32_t batch,
                              std::span<const float> parameters);
}
