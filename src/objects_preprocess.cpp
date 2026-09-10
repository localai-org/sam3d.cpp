// Copyright (c) Meta Platforms, Inc. and affiliates.
// Modified C++ composition of original PointMap preprocessing: SAM License.
#include "objects_preprocess.hpp"
#include <stdexcept>
namespace sam3d {
void validate_objects_preprocess_options(objects_preprocess_options o){
    validate_objects_image_options({o.image_side,o.box_factor,o.padding});
    if(!o.point_side || o.point_side>1024)throw std::invalid_argument("invalid pointmap output side");
    validate_objects_ssi({1,1,1,1},o.object_normalizer);validate_objects_ssi({1,1,1,1},o.full_normalizer);
}
objects_image_taps objects_preprocess_pointmap(std::span<const uint8_t> rgba,uint32_t w,uint32_t h,uint64_t stride,
    std::span<const float> xyz,uint32_t ph,uint32_t pw,objects_preprocess_options o,const objects_tensor_observer &observe){
    validate_objects_preprocess_options(o);
    if(!w || !h || w>4096 || h>4096 || stride<uint64_t(w)*4 || stride>uint64_t(w)*4+4096 ||
       rgba.size()<uint64_t(h-1)*stride+w*4 || !ph || !pw || ph>2048 || pw>2048 || xyz.size()!=uint64_t(3)*ph*pw)
        throw std::invalid_argument("invalid pointmap preprocessing input");
    auto emit=[&](const std::string &key,std::span<const float> v){if(observe)observe(key,v);};
    std::vector<float> rgb(uint64_t(3)*h*w),mask(uint64_t(h)*w);
    for(uint32_t y=0;y<h;++y)for(uint32_t x=0;x<w;++x){
        for(uint32_t c=0;c<3;++c)rgb[(uint64_t(c)*h+y)*w+x]=float(double(rgba[y*stride+x*4+c])/255.);
        mask[uint64_t(y)*w+x]=float(double(rgba[y*stride+x*4+3])/255.);
    }
    emit("00.rgb",rgb);emit("00.mask",mask);
    auto object=objects_normalize_pointmap({ph,pw,h,w},o.object_normalizer,xyz,mask);
    if(!o.normalize)object.pointmap.assign(xyz.begin(),xyz.end());
    emit("01.object.pointmap",object.pointmap);emit("01.object.scale",object.scale);emit("01.object.shift",object.shift);
    emit("02.full_before.rgb",rgb);emit("02.full_before.mask",mask);
    auto joint=objects_prepare_pointmap_joint(rgba,w,h,stride,object.pointmap,ph,pw,o.box_factor,o.padding,bool(observe));
    for(auto &[key,v]:joint.taps)if(key!="00.rgba" && key!="01.alpha")emit("03."+key,v);
    joint.taps.clear();
    objects_image_taps out;uint32_t transform=0;
    auto apply=[&](std::span<const float> v,uint32_t c,uint32_t height,uint32_t width,bool image,bool point){
        auto prefix="04.apply."+std::to_string(transform++)+".";
        return objects_square_resize(v,c,height,width,point?o.point_side:o.image_side,image,point && o.point_nan_padding,
            observe?objects_tensor_observer([&](const std::string &key,std::span<const float> data){emit(prefix+key,data);}):objects_tensor_observer());
    };
    out["image"]=apply(joint.rgb,3,joint.height,joint.width,true,false);
    out["mask"]=apply(joint.mask,1,joint.height,joint.width,false,false);
    out["pointmap"]=apply(joint.pointmap,3,joint.height,joint.width,false,true);
    out["rgb_image"]=apply(rgb,3,h,w,true,false);out["rgb_image_mask"]=apply(mask,1,h,w,false,false);
    // Upstream supplies the TRANSFORMED full-image mask to the second normalizer.
    // With normalize=false it recomputes moments without forwarding overrides.
    auto full=objects_normalize_pointmap({ph,pw,o.image_side,o.image_side},o.full_normalizer,xyz,out.at("rgb_image_mask"),
        o.normalize?std::span<const float>(object.scale):std::span<const float>(),o.normalize?std::span<const float>(object.shift):std::span<const float>());
    if(!o.normalize)full.pointmap.assign(xyz.begin(),xyz.end());
    emit("05.full.pointmap",full.pointmap);emit("05.full.scale",full.scale);emit("05.full.shift",full.shift);
    out["rgb_pointmap"]=apply(full.pointmap,3,ph,pw,false,true);
    out["rgb_pointmap_unnorm"]=apply(xyz,3,ph,pw,false,true);
    out["pointmap_scale"]={object.scale.begin(),object.scale.end()};out["pointmap_shift"]={object.shift.begin(),object.shift.end()};
    out["rgb_pointmap_scale"]={full.scale.begin(),full.scale.end()};out["rgb_pointmap_shift"]={full.shift.begin(),full.shift.end()};
    for(auto &[key,v]:out)emit("06.return."+key,v);
    return out;
}
}
