// Copyright (c) Meta Platforms, Inc. and affiliates. Adapted from SAM3DBody
// _get_hand_box / run_inference; SAM license, THIRD_PARTY_NOTICES.md.
#include "body_hand_crop.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace sam3d {
namespace {
void require(bool b,const char *s){if(!b)throw std::invalid_argument(s);}
void finite(std::span<const float> a){require(std::all_of(a.begin(),a.end(),[](float v){return std::isfinite(v);}),"nonfinite hand crop geometry");}
}
named_floats body_hand_boxes(std::span<const float> boxes,std::span<const float> a,uint32_t width,uint32_t crop){
    require(boxes.size()==8 && a.size()==6 && width>0 && width<=32766 && crop>0 && crop<=512,"invalid hand crop dimensions");finite(boxes);finite(a);
    const float magnitude=std::max(std::abs(a[0]),std::abs(a[4]));
    // Original OpenCV LU may leave roundoff-sized off-diagonal coefficients.
    require(a[0]>0 && a[4]>0 && std::abs(a[1])<=magnitude*1e-6f && std::abs(a[3])<=magnitude*1e-6f && std::abs(a[0]-a[4])<=magnitude*1e-6f,"hand boxes require an axis-aligned square body crop");
    require(std::all_of(boxes.begin(),boxes.end(),[](float v){return v>=0 && v<=1;}),"hand boxes outside normalized detector range");
    named_floats out;
    for(size_t h=0;h<2;++h){
        const auto p=boxes.subspan(h*4,4);const std::string prefix=h?"right.":"left.";
        require(p[2]>0 && p[3]>0,"empty hand box");
        // Keep NumPy's F32 operation order, including the pre-inverse square.
        const float side=std::max(p[2]*float(crop),p[3]*float(crop))/a[0];
        const float x=(p[0]*float(crop)-a[2])/a[0],y=(p[1]*float(crop)-a[5])/a[0];
        out[prefix+"center"]={x,y};out[prefix+"scale"]={side,side};
        auto &box=out[prefix+"full_xyxy"];box={x-side*1.f/2.f,y-side*1.f/2.f,x+side*1.f/2.f,y+side*1.f/2.f};
        out[prefix+"input_xyxy"]=box;
        if(!h){out[prefix+"input_xyxy"][0]=float(width)-box[2]-1.f;out[prefix+"input_xyxy"][2]=float(width)-box[0]-1.f;}
    }
    for(auto &[_,v]:out)finite(v);return out;
}
named_floats body_prepare_hands(std::span<const uint8_t> rgb,uint32_t width,uint32_t height,uint64_t stride,
    std::span<const float> boxes,std::span<const float> affine,std::span<const float> intrinsics,uint32_t crop){
    require(width && height && width<=32766 && height<=32766 && uint64_t(width)*height<=16000000,"invalid hand image dimensions");
    const uint64_t row=uint64_t(width)*3;
    require(stride>=row && rgb.size()>=row && (height==1 || stride<=(rgb.size()-row)/(height-1)),"hand RGB buffer too small");
    auto result=body_hand_boxes(boxes,affine,width,crop);
    std::vector<uint8_t> mirrored(row*height);
    for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x)for(uint32_t c=0;c<3;++c)mirrored[uint64_t(y)*row+x*3+c]=rgb[uint64_t(y)*stride+(width-1-x)*3+c];
    for(const auto prefix:{std::string("left."),std::string("right.")}){
        auto prepared=body_prepare_rgb(prefix=="left."?std::span<const uint8_t>(mirrored):rgb,width,height,prefix=="left."?row:stride,result.at(prefix+"input_xyxy"),intrinsics,crop,true,.9f);
        for(auto &[name,v]:prepared)result[prefix+name]=std::move(v);
    }
    // Upstream uses the unflipped centre later for wrist validity/merging, but
    // the actual left image crop, rays and intrinsics stay in mirrored space.
    result["left.unflipped_center"]=result.at("left.box_center");
    result["left.unflipped_center"][0]=float(width)-result["left.box_center"][0]-1.f;
    return result;
}
}
