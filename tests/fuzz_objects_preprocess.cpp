#include "objects_preprocess.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *input,size_t size){
    if(size<32)return 0;
    auto raw=[&](size_t at){std::array<uint8_t,4>b;for(size_t j=0;j<4;++j)b[j]=input[(at+j)%size];float v;std::memcpy(&v,b.data(),4);return v;};
    uint32_t h=input[0]%12+1,w=input[1]%12+1,ph=input[2]%12+1,pw=input[3]%12+1,stride=w*4+input[4]%4;
    sam3d::objects_preprocess_options o;o.image_side=input[5]%24+1;o.point_side=input[6]%16+1;o.normalize=input[7]&1;o.point_nan_padding=input[8]&1;
    o.box_factor=input[9]&1?1.:raw(9);o.padding=input[10]&1?.1:raw(10);
    for(auto [opt,at]:{std::pair{&o.object_normalizer,11u},std::pair{&o.full_normalizer,20u}}){
        opt->mode=sam3d::objects_ssi_mode(input[at]%9);opt->quantile_drop=input[at+1]/510.;opt->clip=input[at+2]&1?0:input[at+2]/16.;opt->scale_factor=input[at+3]&1?1.3:raw(at+3);opt->log_disparity_shift=int8_t(input[at+4])/8.;opt->allow_override=input[at+5]&1;opt->raise_on_no_valid_points=input[at+6]&1;
    }
    std::vector<uint8_t> rgba(h*stride);for(size_t i=0;i<rgba.size();++i)rgba[i]=input[(i+29)%size];
    std::vector<float> xyz(3*ph*pw);for(size_t i=0;i<xyz.size();++i)xyz[i]=input[28]&1?float(int8_t(input[(i+31)%size]))/32.f+(i>=xyz.size()*2/3?5.f:0.f):raw(i+31);
    if(input[29]&1)xyz.pop_back();if(input[30]&1)rgba.pop_back();
    try{
        sam3d::objects_image_taps taps;
        bool observe=input[31]&1;
        auto result=sam3d::objects_preprocess_pointmap(rgba,w,h,stride,xyz,ph,pw,o,observe?sam3d::objects_tensor_observer([&](const std::string &key,std::span<const float> v){if(key.starts_with("06.return."))taps.emplace(key.substr(10),std::vector<float>(v.begin(),v.end()));}):sam3d::objects_tensor_observer());
        if(result.size()!=11)std::abort();
        for(auto &[key,v]:result){
            if(observe){auto &other=taps.at(key);if(other.size()!=v.size() || !std::equal(v.begin(),v.end(),other.begin(),[](float a,float b){return a==b || (std::isnan(a) && std::isnan(b));}))std::abort();}
            bool moments=key.ends_with("scale") || key.ends_with("shift");
            size_t n=moments?3:size_t(key.find("pointmap")!=std::string::npos?o.point_side:o.image_side)*(key.find("pointmap")!=std::string::npos?o.point_side:o.image_side)*(key.find("mask")!=std::string::npos?1:3);
            if(v.size()!=n)std::abort();
            if(moments)for(float x:v)if(!std::isfinite(x) || (key.ends_with("scale") && x<=0))std::abort();
        }
    }catch(const std::invalid_argument &){}
    return 0;
}
