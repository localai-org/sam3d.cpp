// Copyright (c) Meta Platforms, Inc. and affiliates. Adapted from the left
// output-unflip block of SAM3DBody.run_inference. SAM License; see notices.
#include "body_hand_unmirror.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace sam3d {
named_floats body_unmirror_left(uint32_t b,uint32_t width,
    std::span<const float> scale,std::span<const float> rotations,
    std::span<const float> hand,std::span<const float> center,
    std::span<const float> mean,std::span<const float> components){
    auto require=[](bool valid,const char *message){if(!valid)throw std::invalid_argument(message);};
    require(b>0 && b<=64 && width>0 && width<=32766,"invalid left-hand unmirror dimensions");
    require(scale.size()==uint64_t(b)*28 && rotations.size()==uint64_t(b)*127*9 &&
        hand.size()==uint64_t(b)*108 && center.size()==uint64_t(b)*2 &&
        mean.size()==68 && components.size()==28*68,"left-hand unmirror shape mismatch");
    auto finite=[&](std::span<const float> values){require(std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}),"nonfinite left-hand unmirror data");};
    for(auto values:{scale,rotations,hand,center,mean,components})finite(values);
    const float right_std=components[8*68+8],left_std=components[9*68+9];
    require(left_std!=0,"zero left-hand scale component");
    named_floats result{{"scale",{scale.begin(),scale.end()}},{"joint_global_rots",{rotations.begin(),rotations.end()}},
                       {"hand",{hand.begin(),hand.end()}},{"bbox_center",{center.begin(),center.end()}}};
    for(uint32_t i=0;i<b;++i){
        // Keep the original separate F32 operations; no multiply-add fusion.
        result.at("scale")[i*28+9]=((mean[8]+right_std*scale[i*28+8])-mean[9])/left_std;
        auto &r=result.at("joint_global_rots");
        for(uint32_t row=0;row<3;++row)for(uint32_t col=0;col<3;++col){
            float value=rotations[(i*127+42)*9+row*3+col];
            r[(i*127+78)*9+row*3+col]=row? -value:value;
        }
        std::copy_n(hand.begin()+i*108+54,54,result.at("hand").begin()+i*108);
        result.at("bbox_center")[i*2]=float(width)-center[i*2]-1.f;
    }
    for(auto &[_,values]:result)finite(values);
    return result;
}
}
