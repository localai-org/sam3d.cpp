#include "objects_image.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *input,size_t size){
    if(size<32)return 0;
    uint32_t w=input[0]%16+1,h=input[1]%16+1,ph=input[2]%16+1,pw=input[3]%16+1;uint64_t stride=w*4+(input[4]%8);
    std::vector<uint8_t> rgba(h*stride);for(size_t i=0;i<rgba.size();++i)rgba[i]=input[(i+5)%size];
    std::vector<float> xyz(size_t(3)*ph*pw);for(size_t i=0;i<xyz.size();++i){std::array<uint8_t,4> bytes;for(size_t k=0;k<4;++k)bytes[k]=input[(i+k+7)%size];std::memcpy(&xyz[i],bytes.data(),4);if(input[5]&1)xyz[i]=int8_t(input[(i+6)%size])/32.f;}
    double factor,pad;std::memcpy(&factor,input+8,8);std::memcpy(&pad,input+16,8);if(input[6]&1){factor=.25+input[7]/68.;pad=input[8]/255.;}
    try{
        auto out=sam3d::objects_prepare_pointmap_joint(rgba,w,h,stride,xyz,ph,pw,factor,pad,input[9]&1);auto n=size_t(out.height)*out.width;
        if(out.rgb.size()!=3*n || out.mask.size()!=n || out.pointmap.size()!=3*n)std::abort();
        for(size_t i=0;i<n;++i){float m=out.mask[i];if(!std::isfinite(m) || m<0 || m>1)std::abort();
            for(size_t c=0;c<3;++c){float v=out.rgb[c*n+i];if(!std::isfinite(v) || v<0 || v>1 || (m==0 && !std::isnan(out.pointmap[c*n+i])))std::abort();}}
    }catch(const std::invalid_argument &){}
    return 0;
}
