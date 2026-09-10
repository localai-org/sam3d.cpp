#include "body_hand_crop.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *input,size_t size){
    if(size<80)return 0;
    std::array<float,8> boxes;std::array<float,6> affine;std::array<float,4> camera;
    std::memcpy(boxes.data(),input,32);std::memcpy(affine.data(),input+32,24);std::memcpy(camera.data(),input+56,16);
    std::array<uint8_t,192> rgb;for(size_t i=0;i<rgb.size();++i)rgb[i]=input[i%size];
    uint32_t w=8,h=8,crop=input[76]%16+1;uint64_t stride=24;
    if(input[72]&1){
        for(size_t i=0;i<boxes.size();++i)boxes[i]=float(input[i])/255.f;
        affine={.5f+input[32]/64.f,0,float(input[33])-128,0,.5f+input[32]/64.f,float(input[34])-128};
        camera={50,60,4,4};
    }
    if(input[72]&2){std::memcpy(&w,input+4,4);std::memcpy(&h,input+8,4);std::memcpy(&stride,input+12,8);}
    try{
        auto result=sam3d::body_prepare_hands(rgb,w,h,stride,boxes,affine,camera,crop);
        for(auto &[_,values]:result)for(float value:values)if(!std::isfinite(value))std::abort();
        if(result.at("left.normalized_rgb").size()!=uint64_t(crop)*crop*3 || result.at("right.normalized_rgb").size()!=uint64_t(crop)*crop*3)std::abort();
    }catch(const std::invalid_argument &){}catch(const std::bad_alloc &){}
    return 0;
}
