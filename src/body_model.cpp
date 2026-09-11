#include "body_model.hpp"
#include <array>
#include <stdexcept>
namespace sam3d {
void validate_body_inference_options(body_inference_options o){
    if((o.crop_size!=512 && o.crop_size!=448 && o.crop_size!=384) || (o.intermediate_mask & ~31u))
        throw std::invalid_argument("Body crop must be 384, 448 or 512; intermediate mask uses bits 0..4 (final layer always runs)");
}
body_pipeline_shape trained_body_pipeline_shape(bool operations,body_inference_options o){
    validate_body_inference_options(o);
    body_pipeline_shape shape={{1,o.crop_size,o.crop_size,16,1280,20,5120,32,4},
            {{1,512,512,16,1280,1024,1,70,519,70,true,true},6,8,64,1024,1024,2,1024,2,1.f,true,false,true,operations,false,operations},true};
    shape.decoder.condition.height=shape.decoder.condition.width=o.crop_size;
    shape.decoder.intermediate_mask=o.intermediate_mask;
    shape.decoder.correctives=o.correctives;
    shape.decoder.slim_intermediates=o.slim_intermediates;
    return shape;
}
body_model::body_model(const std::filesystem::path &backbone,const std::filesystem::path &branch,
    const std::filesystem::path &mhr,const std::string &module,const std::string &backend,
    uint32_t device,uint32_t threads,const std::string &description,bool bf16,body_inference_options inference):
    backbone_(backbone,"sam3d.body.dinov3.vith16plus",body_dino_shapes()),
    branch_(branch,"sam3d.body.pose_branch",body_branch_shapes()),
    mhr_(mhr,"sam3d.mhr.lod1",mhr_lod1_shapes()),
    session_(module,backend,device,threads,description,true),
    arithmetic_(backend=="Vulkan"?scalar_division::reciprocal_multiply:scalar_division::direct),bf16_(bf16),inference_(inference){
    validate_body_inference_options(inference_);
    if(backbone_.metadata("sam3d.source.checkpoint_sha256")!=branch_.metadata("sam3d.source.checkpoint_sha256") ||
       backbone_.metadata("sam3d.source.config_sha256")!=branch_.metadata("sam3d.source.config_sha256") ||
       branch_.metadata("sam3d.required_backbone")!=backbone_.metadata("general.architecture") ||
       branch_.metadata("sam3d.required_mhr_sha256")!=mhr_.metadata("sam3d.source.asset_sha256"))
        throw std::invalid_argument("incompatible Body GGUF companions");
    for(auto &[name,n]:body_pipeline_parameter_sizes(trained_body_pipeline_shape(false,inference_)))weights_[name]=branch_.load(name,n*4);
    hand_indices_=branch_.read_i32("head_pose.hand_joint_idxs_left",27*4);
    auto right=branch_.read_i32("head_pose.hand_joint_idxs_right",27*4);hand_indices_.insert(hand_indices_.end(),right.begin(),right.end());
    faces_=branch_.read_i32("head_pose.faces",36874*3*4);
    if(faces_!=mhr_.read_i32("mesh.faces",36874*3*4))throw std::invalid_argument("Body head and MHR mesh topology differ");
    if(backend=="Vulkan")resident_=std::make_unique<dino_resident_stack>(session_,trained_body_pipeline_shape(false,inference_).backbone,
        [&](const std::string &name,uint64_t count){return backbone_.load(name,count*4);},bf16_);
}
named_floats body_model::infer_rgb(std::span<const uint8_t> rgb,uint32_t width,uint32_t height,uint64_t stride,
    std::span<const float> box,std::span<const float> intrinsics,bool operations){
    std::lock_guard lock(mutex_);const std::array<float,3> dummy{0,0,-2};
    return body_from_rgb(session_,mhr_,trained_body_pipeline_shape(operations,inference_),rgb,width,height,stride,box,intrinsics,dummy,{},
        [&](const std::string &name,uint64_t count){return backbone_.load(name,count*4);},hand_indices_,weights_,arithmetic_,{},resident_.get(),bf16_);
}
}
