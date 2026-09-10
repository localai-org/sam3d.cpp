#pragma once
#include "dino_block.hpp"

namespace sam3d {
struct camera_shape { uint32_t batch, height, width, patch, dim; };
void validate_camera_shape(camera_shape);
// Rays: Bx2xHxW, features: BxDx(H/P)x(W/P), F32. Parameters are original
// conv.weight [D,D+99,1,1], norm.weight [D], norm.bias [D]. No hidden ray source.
// channels_last returns 07.output as B,N,D instead of B,D,N; all arithmetic
// and other diagnostic taps are identical. Internal layout option only.
named_floats camera_encode(neural_session &,camera_shape,std::span<const float> features,
    std::span<const float> rays,const weight_map &parameters,bool capture_all = true,bool channels_last = false);
}
