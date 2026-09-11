// Copyright (c) Meta Platforms, Inc. and affiliates.
// SAM3DBody.forward_decoder input adaptation; SAM license, NOTICE.
#include "body_condition.hpp"
#include "finite.hpp"
#include "camera_encoder.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sam3d {
namespace {
void require(bool b,const char *m) {if(!b)throw std::invalid_argument(m);}
void finite(std::span<const float> v) {
    require(all_finite_f32(v),"non-finite conditioning tensor");
}
weight_map subset(const weight_map &parameters,const std::string &prefix) {
    weight_map result;
    for(auto &[name,v]:parameters) if(name.starts_with(prefix)) result[name.substr(prefix.size())]=v;
    return result;
}
}
void validate_condition_shape(condition_shape s) {
    validate_camera_shape({s.batch,s.height,s.width,s.patch,s.context_dim});
    validate_prompt_shape({s.batch,s.points,s.context_dim,s.joints,s.height/s.patch,s.width/s.patch});
    require(s.token_dim>=1 && s.token_dim<=1280 && s.pose_dim>=1 && s.pose_dim<=1024 &&
        s.keypoints>=1 && s.keypoints<=70 && 2+s.points+s.keypoints*(1+uint32_t(s.keypoints3d))+2*uint32_t(s.hand_tokens)<=256,
        "invalid conditioning token dimensions");
}
uint32_t condition_token_count(condition_shape s) {
    validate_condition_shape(s);return 2+s.points+s.keypoints*(1+uint32_t(s.keypoints3d))+2*uint32_t(s.hand_tokens);
}
std::vector<std::pair<std::string,uint64_t>> condition_parameter_sizes(condition_shape s) {
    validate_condition_shape(s);const uint64_t d=s.token_dim,c=s.context_dim,p=s.pose_dim+3;
    std::vector<std::pair<std::string,uint64_t>> result{{"init_pose.weight",s.pose_dim},{"init_camera.weight",3},
        {"init_to_token_mhr.weight",d*(p+3)},{"init_to_token_mhr.bias",d},
        {"prev_to_token_mhr.weight",d*p},{"prev_to_token_mhr.bias",d},
        {"prompt_to_token.weight",d*c},{"prompt_to_token.bias",d},{"keypoint_embedding.weight",s.keypoints*d},
        {"ray_cond_emb.conv.weight",c*(c+99)},{"ray_cond_emb.norm.weight",c},{"ray_cond_emb.norm.bias",c}};
    if(s.keypoints3d)result.emplace_back("keypoint3d_embedding.weight",s.keypoints*d);
    if(s.hand_tokens)result.emplace_back("hand_box_embedding.weight",2*d);
    for(auto &[name,size]:prompt_parameter_sizes({s.batch,s.points,s.context_dim,s.joints,s.height/s.patch,s.width/s.patch}))
        result.emplace_back("prompt_encoder."+name,size);
    return result;
}
named_floats body_condition(neural_session &session,condition_shape s,std::span<const float> features,
    std::span<const float> rays,std::span<const float> cliff,std::span<const float> keypoints,
    std::span<const float> previous,const weight_map &parameters,scalar_division division,std::span<const float> dense_gaussian,bool images_channels_last) {
    const auto sizes=condition_parameter_sizes(s);
    const uint64_t b=s.batch,d=s.token_dim,cx=s.context_dim,p=s.pose_dim+3,grid=(s.height/s.patch)*(s.width/s.patch);
    require(features.size()==b*cx*grid && rays.size()==b*2*s.height*s.width && cliff.size()==b*3 &&
        keypoints.size()==b*s.points*3 && (previous.empty() || previous.size()==b*p),"conditioning input size mismatch");
    for(auto v:{features,rays,cliff,keypoints,previous})finite(v);
    require(parameters.size()==sizes.size(),"conditioning parameter set mismatch");
    for(auto &[name,size]:sizes) {require(parameters.contains(name) && parameters.at(name).size()==size,"conditioning parameter size/name mismatch");}
    auto prompt=body_prompt_encode(session,{s.batch,s.points,s.context_dim,s.joints,s.height/s.patch,s.width/s.patch},
        keypoints,subset(parameters,"prompt_encoder."),division,dense_gaussian,false,images_channels_last);
    auto camera=camera_encode(session,{s.batch,s.height,s.width,s.patch,s.context_dim},features,rays,subset(parameters,"ray_cond_emb."),false,images_channels_last);
    named_floats result;
    result["10.prompt_input"]=std::move(prompt.at("20.embeddings"));result["11.prompt_mask"]=std::move(prompt.at("21.mask"));
    result["20.image"]=std::move(camera.at("07.output"));result["21.image_pe"]=std::move(prompt.at(images_channels_last?"22.dense_tokens":"22.dense_nchw"));
    auto &initial=result["00.init_input"];initial.resize(b*(p+3));
    auto &prev=result["02.prev_input"];prev.resize(b*p);
    for(uint64_t batch=0;batch<b;++batch) {
        auto dst=initial.begin()+batch*(p+3);
        std::copy_n(cliff.begin()+batch*3,3,dst);dst+=3;
        std::copy(parameters.at("init_pose.weight").begin(),parameters.at("init_pose.weight").end(),dst);dst+=s.pose_dim;
        std::copy(parameters.at("init_camera.weight").begin(),parameters.at("init_camera.weight").end(),dst);
        if(previous.empty())std::copy_n(initial.begin()+batch*(p+3)+3,p,prev.begin()+batch*p);
        else std::copy_n(previous.begin()+batch*p,p,prev.begin()+batch*p);
    }
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({
        ggml_tensor_overhead()*128+ggml_graph_overhead_custom(128,false),nullptr,true}),ggml_free);
    if(!ctx)throw std::bad_alloc();auto c=ctx.get();
    std::map<std::string,ggml_tensor *> taps;
    std::vector<std::pair<ggml_tensor *,std::span<const float>>> uploads;
    auto linear=[&](const std::string &name,const std::string &output,std::span<const float> input,uint64_t in,uint64_t n) {
        auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,in,n),weight=session.parameter(c,parameters.at(name+".weight"),in,d),bias=session.parameter(c,parameters.at(name+".bias"),d,1);
        uploads.emplace_back(x,input);
        auto projected=ggml_mul_mat(c,weight,x);ggml_mul_mat_set_prec(projected,GGML_PREC_F32);
        auto out=ggml_add(c,projected,bias);ggml_set_output(out);ggml_set_name(out,output.c_str());taps[output]=out;
    };
    linear("init_to_token_mhr","01.pose_token",initial,p+3,b);
    linear("prev_to_token_mhr","03.prev_token",prev,p,b);
    linear("prompt_to_token","12.prompt_token",result.at("10.prompt_input"),cx,b*s.points);
    auto graph=ggml_new_graph_custom(c,128,false);
    for(auto &[name,t]:taps)ggml_build_forward_expand(graph,t);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported conditioning projection operation");
    auto buffer=session.allocate(c);
    if(!buffer)throw std::bad_alloc();
    for(auto &[name,v]:session.evaluate_f32(graph,uploads,taps)){finite(v);result[name]=std::move(v);}
    const uint64_t n=condition_token_count(s);
    auto &tokens=result["30.tokens"],&augment=result["31.token_pe"];tokens.resize(b*n*d);augment.resize(b*n*d,0.f);
    for(uint64_t batch=0;batch<b;++batch) {
        uint64_t offset=batch*n*d;
        auto append=[&](std::span<const float> values,bool pe) {
            std::copy(values.begin(),values.end(),tokens.begin()+offset);
            if(pe)std::copy(values.begin(),values.end(),augment.begin()+offset);
            offset+=values.size();
        };
        append(std::span(result.at("01.pose_token")).subspan(batch*d,d),false);
        append(std::span(result.at("03.prev_token")).subspan(batch*d,d),true);
        append(std::span(result.at("12.prompt_token")).subspan(batch*s.points*d,s.points*d),true);
        if(s.hand_tokens)append(parameters.at("hand_box_embedding.weight"),false);
        append(parameters.at("keypoint_embedding.weight"),false);
        if(s.keypoints3d)append(parameters.at("keypoint3d_embedding.weight"),false);
        require(offset==(batch+1)*n*d,"conditioning token assembly mismatch");
    }
    return result;
}
}
