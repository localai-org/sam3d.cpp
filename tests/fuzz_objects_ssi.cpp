#include "objects_ssi.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *input,size_t size){
    if(size<32)return 0;
    auto raw=[&](size_t offset){std::array<uint8_t,4> bytes;for(size_t j=0;j<4;++j)bytes[j]=input[(offset+j)%size];float value;std::memcpy(&value,bytes.data(),4);return value;};
    sam3d::objects_ssi_shape s{uint32_t(input[1]%16+1),uint32_t(input[2]%16+1),uint32_t(input[3]%16+1),uint32_t(input[4]%16+1)};
    sam3d::objects_ssi_options o;o.mode=sam3d::objects_ssi_mode(input[0]%9);o.quantile_drop=input[5]/510.;o.clip=input[6]&1?input[6]/16.:0;
    o.scale_factor=input[7]&1?1.3:raw(8);o.log_disparity_shift=int8_t(input[12])/8.;o.allow_override=input[13]&1;o.raise_on_no_valid_points=input[14]&1;
    std::vector<float> xyz(uint64_t(3)*s.height*s.width),mask(uint64_t(s.mask_height)*s.mask_width);
    for(size_t i=0;i<xyz.size();++i)xyz[i]=input[15]&1?float(int8_t(input[(i+16)%size]))/32.f+(i>=2*xyz.size()/3?4.f:0.f):raw(i+16);
    for(size_t i=0;i<mask.size();++i)mask[i]=input[16]&1?float(input[(i+17)%size])/255.f:raw(i+17);
    std::array<float,3> scale{raw(20),raw(24),raw(28)},shift{raw(19),raw(23),raw(27)};
    try{
        auto out=sam3d::objects_normalize_pointmap(s,o,xyz,mask,input[17]&1?std::span<const float>(scale):std::span<const float>(),input[18]&1?std::span<const float>(shift):std::span<const float>(),input[19]&1);
        if(out.pointmap.size()!=xyz.size())std::abort();
        for(auto x:out.scale)if(!std::isfinite(x) || x<=0)std::abort();
        for(auto x:out.shift)if(!std::isfinite(x))std::abort();
        auto back=sam3d::objects_denormalize_pointmap(s.height,s.width,o.mode,out.pointmap,out.scale,out.shift);
        if(back.size()!=xyz.size())std::abort();
    }catch(const std::invalid_argument &){}
    return 0;
}
