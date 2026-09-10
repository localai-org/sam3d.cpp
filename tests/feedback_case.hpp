#pragma once
#include "body_feedback.hpp"
#include <array>
inline sam3d::feedback_shape feedback_dimensions(const std::array<uint32_t,15> &v){
    sam3d::feedback_shape s{v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14]};
    sam3d::validate_feedback_shape(s);return s;
}
inline std::vector<std::pair<std::string,uint64_t>> feedback_inputs(sam3d::feedback_shape s){const uint64_t b=s.batch,j=s.points;
    return {{"image",b*s.context_dim*s.height*s.width},{"tokens",b*s.tokens*s.dim},{"augment",b*s.tokens*s.dim},
        {"pixels",b*j*2},{"depths",b*j},{"world",b*j*3},{"affine",b*6},{"crop_size",b*2}};
}
inline sam3d::named_floats run_feedback(sam3d::neural_session &session,sam3d::feedback_shape s,const sam3d::named_floats &input,
    const sam3d::named_floats &parameters,std::span<const int32_t> idx,std::span<const int32_t> idx3,bool capture_all=true,bool channels_last=false){
    return sam3d::body_feedback(session,s,input.at("image"),input.at("tokens"),input.at("augment"),input.at("pixels"),input.at("depths"),input.at("world"),
        input.at("affine"),input.at("crop_size"),idx,idx3,parameters,capture_all,channels_last);
}
