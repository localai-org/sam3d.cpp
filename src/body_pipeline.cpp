// Copyright (c) Meta Platforms, Inc. and affiliates.
// Composition follows SAM3DBody.forward_pose_branch. SAM license and detailed
// attribution: NOTICE. No reference intermediates are accepted.
#include "body_pipeline.hpp"
#include "sam3d.h"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
namespace sam3d {
void validate_body_pipeline_shape(body_pipeline_shape s){
    validate_backbone_shape(s.backbone);validate_body_flow_shape(s.decoder);
    auto b=s.backbone;auto c=s.decoder.condition;
    if(b.batch!=1 || c.batch!=1 || b.height!=b.width || b.height>512 ||
       b.height!=c.height || b.width!=c.width || b.patch!=c.patch || b.dim!=c.context_dim)
        throw std::invalid_argument("incompatible image/backbone/decoder composition");
    if(s.decoder.hand_branch && !s.trained_branch)throw std::invalid_argument("hand image composition requires the explicit trained branch contract");
    if(s.trained_branch && !c.hand_tokens)throw std::invalid_argument("trained branch requires hand detection tokens");
}
std::vector<std::pair<std::string,uint64_t>> body_pipeline_parameter_sizes(body_pipeline_shape s){
    validate_body_pipeline_shape(s);auto out=body_flow_parameter_sizes(s.decoder);const uint64_t d=s.decoder.condition.token_dim;
    if(s.trained_branch){
        out.emplace_back("prompt_encoder.no_mask_embed.weight",s.backbone.dim);
        for(uint32_t i=0;i<3;++i){const uint64_t n=i==2?4:d;const auto p="bbox_embed.layers."+std::to_string(i);
            out.emplace_back(p+".weight",d*n);out.emplace_back(p+".bias",n);}
        out.emplace_back("hand_cls_embed.weight",d*2);out.emplace_back("hand_cls_embed.bias",2);
    }return out;
}
named_floats body_hand_detection(neural_session &session,uint32_t d,std::span<const float> tokens,const weight_map &p){
    if(!d || d>1280 || tokens.size()!=uint64_t(2)*d || p.size()!=8)throw std::invalid_argument("invalid hand detection inputs");
    auto finite=[](auto values){if(!std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}))throw std::invalid_argument("nonfinite hand detection input");};
    finite(tokens); // weight_map owns checked immutable parameter snapshots.
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({ggml_tensor_overhead()*80+ggml_graph_overhead_custom(80,false),nullptr,true}),ggml_free);
    if(!ctx)throw std::bad_alloc();auto c=ctx.get();auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,d,2);
    std::vector<std::pair<ggml_tensor *,std::span<const float>>> uploads{{x,tokens}};
    std::map<std::string,ggml_tensor *> taps;
    auto tap=[&](const std::string &name,ggml_tensor *t){ggml_set_output(t);taps[name]=t;return t;};
    auto linear=[&](ggml_tensor *v,const std::string &name,uint32_t out){
        if(!p.contains(name+".weight") || !p.contains(name+".bias") || p.at(name+".weight").size()!=uint64_t(d)*out || p.at(name+".bias").size()!=out)throw std::invalid_argument("hand detection parameter mismatch");
        auto w=session.parameter(c,p.at(name+".weight"),d,out),b=session.parameter(c,p.at(name+".bias"),out,1);
        auto y=ggml_mul_mat(c,w,v);ggml_mul_mat_set_prec(y,GGML_PREC_F32);return ggml_add(c,y,b);
    };
    auto boxes=x;for(uint32_t i=0;i<3;++i){boxes=tap("box."+std::to_string(i)+".linear",linear(boxes,"bbox_embed.layers."+std::to_string(i),i==2?4:d));if(i<2)boxes=ggml_relu(c,boxes);}
    tap("hand_box",ggml_sigmoid(c,boxes));tap("hand_logits",linear(x,"hand_cls_embed",2));
    auto graph=ggml_new_graph_custom(c,80,false);for(auto &[name,t]:taps)ggml_build_forward_expand(graph,t);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)if(!ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)))throw std::invalid_argument("unsupported hand detection operation");
    auto buffer=session.allocate(c);
    if(!buffer)throw std::bad_alloc();auto out=session.evaluate_f32(graph,uploads,taps);
    for(auto &[_,v]:out)finite(v);return out;
}
named_floats body_prepare_rgb(std::span<const uint8_t> rgb,uint32_t width,uint32_t height,
    uint64_t row_stride,std::span<const float> box,std::span<const float> intrinsics,
    uint32_t crop_size,bool intrinsics_center,float padding){
    if(box.size()!=4 || intrinsics.size()!=4 || crop_size<1 || crop_size>512)
        throw std::invalid_argument("invalid image pipeline box/intrinsics/crop size");
    std::array<char,256> error{};
    auto check=[&](s3d_status status){if(status==S3D_OK)return;if(status==S3D_OUT_OF_MEMORY)throw std::bad_alloc();if(status==S3D_INVALID_ARGUMENT)throw std::invalid_argument(error.data());throw std::runtime_error(error.data());};
    s3d_body_crop_request *cr=nullptr;check(s3d_body_crop_request_create(&cr,error.data(),error.size()));
    std::unique_ptr<s3d_body_crop_request,decltype(&s3d_body_crop_request_free)> crop_request(cr,s3d_body_crop_request_free);
    check(s3d_body_crop_request_set_padding(cr,padding,error.data(),error.size()));
    check(s3d_body_crop_request_set_box(cr,box.data(),box.size(),error.data(),error.size()));
    check(s3d_body_crop_request_set_output_size(cr,crop_size,crop_size,error.data(),error.size()));
    s3d_body_crop_result *cp=nullptr;check(s3d_body_crop_compute(cr,&cp,error.data(),error.size()));
    std::unique_ptr<s3d_body_crop_result,decltype(&s3d_body_crop_result_free)> crop(cp,s3d_body_crop_result_free);
    s3d_body_camera_request *ca=nullptr;check(s3d_body_camera_request_create(&ca,error.data(),error.size()));
    std::unique_ptr<s3d_body_camera_request,decltype(&s3d_body_camera_request_free)> camera_request(ca,s3d_body_camera_request_free);
    check(s3d_body_camera_request_set_intrinsics(ca,intrinsics.data(),intrinsics.size(),error.data(),error.size()));
    check(s3d_body_camera_request_set_image_size(ca,width,height,error.data(),error.size()));
    check(s3d_body_camera_request_set_use_intrinsics_center(ca,intrinsics_center,error.data(),error.size()));
    s3d_body_camera_result *cm=nullptr;check(s3d_body_camera_compute(ca,cp,&cm,error.data(),error.size()));
    std::unique_ptr<s3d_body_camera_result,decltype(&s3d_body_camera_result_free)> camera(cm,s3d_body_camera_result_free);
    s3d_body_image_result *im=nullptr;check(s3d_body_image_prepare(cp,rgb.data(),rgb.size(),width,height,row_stride,&im,error.data(),error.size()));
    std::unique_ptr<s3d_body_image_result,decltype(&s3d_body_image_result_free)> image(im,s3d_body_image_result_free);
    named_floats out;const float *data=nullptr;uint64_t count=0;
    check(s3d_body_image_get_normalized(im,&data,&count,error.data(),error.size()));out["normalized_rgb"]={data,data+count};
    check(s3d_body_camera_get_rays(cm,&data,&count,error.data(),error.size()));out["rays"]={data,data+count};
    check(s3d_body_camera_get_condition(cm,&data,&count,error.data(),error.size()));out["cliff"]={data,data+count};
    check(s3d_body_crop_get_center(cp,&data,&count,error.data(),error.size()));out["box_center"]={data,data+count};
    check(s3d_body_crop_get_scale(cp,&data,&count,error.data(),error.size()));out["box_size"]={data[0]};
    const double *affine=nullptr;check(s3d_body_crop_get_affine(cp,&affine,&count,error.data(),error.size()));
    // Original prepare_batch rounds affine metadata to F32, but image sampling
    // itself uses the crop's F64 affine. Do not use this rounded copy for pixels.
    auto &a=out["affine"];for(uint64_t i=0;i<count;++i)a.push_back(float(affine[i]));
    out["image_size"]={float(width),float(height)};out["crop_size"]={float(crop_size),float(crop_size)};
    out["intrinsics"]={intrinsics[0],0,intrinsics[2],0,intrinsics[1],intrinsics[3],0,0,1};
    return out;
}
named_floats body_from_rgb(neural_session &session,tensor_archive &mhr,body_pipeline_shape shape,
    std::span<const uint8_t> rgb,uint32_t width,uint32_t height,uint64_t row_stride,
    std::span<const float> box,std::span<const float> intrinsics,
    std::span<const float> prompts,std::span<const float> previous,
    const parameter_reader &backbone_weights,std::span<const int32_t> hand_indices,
    const weight_map &decoder_weights,scalar_division arithmetic,std::span<const int32_t> nonhand_indices,dino_resident_stack *resident,bool bf16){
    validate_body_pipeline_shape(shape);if(!backbone_weights)throw std::invalid_argument("missing backbone parameter reader");
    if(nonhand_indices.size()!=(shape.decoder.hand_branch?145u:0u))throw std::invalid_argument("invalid image branch nonhand mask size");
    const auto sizes=body_pipeline_parameter_sizes(shape);
    if(decoder_weights.size()!=sizes.size())throw std::invalid_argument("pipeline parameter set mismatch");
    for(auto &[name,n]:sizes)if(!decoder_weights.contains(name) || decoder_weights.at(name).size()!=n)throw std::invalid_argument("pipeline parameter shape mismatch");
    auto p=body_prepare_rgb(rgb,width,height,row_stride,box,intrinsics,shape.backbone.height,shape.decoder.intrinsics_center,shape.decoder.hand_branch?.9f:1.25f);
    named_floats backbone_taps;
    backbone_observer observe;
    if(shape.decoder.capture_boundaries)observe=[&](const std::string &name,std::span<const float> values){backbone_taps["backbone."+name]={values.begin(),values.end()};};
    auto features=dino_backbone(session,shape.backbone,p.at("normalized_rgb"),backbone_weights,
        observe,{},resident,bf16);
    if(shape.trained_branch){
        // Original no-mask branch selects the learned channel embedding (not
        // zero) after the backbone. This mode accepts no segmentation mask.
        const auto &bias=decoder_weights.at("prompt_encoder.no_mask_embed.weight");const uint64_t grid=(shape.backbone.height/shape.backbone.patch)*(shape.backbone.width/shape.backbone.patch);
        for(uint64_t channel=0;channel<bias.size();++channel)for(uint64_t i=0;i<grid;++i)features[channel*grid+i]+=bias[channel];
    }
    weight_map flow_weights;for(auto &[name,n]:body_flow_parameter_sizes(shape.decoder))flow_weights[name]=decoder_weights.at(name);
    auto result=body_forward_decoder(session,mhr,shape.decoder,features,p.at("rays"),p.at("cliff"),prompts,previous,
        p.at("box_center"),p.at("box_size"),p.at("image_size"),p.at("intrinsics"),p.at("affine"),p.at("crop_size"),hand_indices,flow_weights,arithmetic,nonhand_indices);
    if(shape.trained_branch){
        weight_map head;for(auto &[name,v]:decoder_weights)if(name.starts_with("bbox_embed.") || name.starts_with("hand_cls_embed."))head[name]=v;
        auto detected=body_hand_detection(session,shape.decoder.condition.token_dim,result.at("90.output_tokens"),head);
        for(auto &[name,v]:detected)result["branch."+name]=std::move(v);
    }
    if(shape.decoder.capture_boundaries){for(auto &[name,v]:p)result["prepare."+name]=std::move(v);result["backbone.features"]=std::move(features);}
    for(auto &[name,v]:backbone_taps)result[name]=std::move(v);
    return result;
}
}
