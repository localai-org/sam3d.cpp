#pragma once
#include "mhr_skeleton.hpp"
namespace sam3d {
// Complete released LOD1 MHR forward, not SAM image estimation or Body's later
// keypoint/axis mapping. Outputs in cm. Real GGUF parameters, own intermediates.
// Neural projections use the selected GGML backend; trig and skinning are
// currently checked CPU geometry. Diagnostic taps remain materialized.
named_floats mhr_geometry(neural_session &,tensor_archive &,uint32_t batch,
                         std::span<const float> identity,std::span<const float> parameters,
                         std::span<const float> face,bool correctives,bool skin_operation_taps=true);
// The original LBS operation also applies to a selected vertex subset. This
// boundary supports small normal regression tests without loading GGUF weights.
named_floats mhr_skinning(uint32_t batch,uint32_t vertex_count,
                         std::span<const float> skeleton,std::span<const float> unposed,
                         std::span<const float> inverse_bind,std::span<const int32_t> skin_joints,
                         std::span<const float> weights,std::span<const int32_t> skin_vertices,bool operation_taps=true);
}
