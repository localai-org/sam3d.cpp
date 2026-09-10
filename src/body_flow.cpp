// Copyright (c) Meta Platforms, Inc. and affiliates.
// SAM3DBody.forward_decoder / PromptableDecoder adaptation. SAM license and
// attribution: THIRD_PARTY_NOTICES.md. Composition has no reference injection.
#include "body_flow.hpp"
#include "finite.hpp"
#include "body_decoder.hpp"
#include "body_feedback.hpp"
#include "transpose.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
namespace sam3d {
namespace {
void require(bool v,const char *m){if(!v)throw std::invalid_argument(m);}
void finite(std::span<const float> v){require(all_finite_f32(v),"non-finite decoder flow tensor");}
decoder_shape layer_shape(body_flow_shape s,uint32_t layer){auto c=s.condition;return {c.batch,condition_token_count(c),(c.height/c.patch)*(c.width/c.patch),c.token_dim,c.context_dim,s.heads,s.head_dim,s.hidden,s.repeat_pe,layer==0,s.twoway};}
feedback_shape update_shape(body_flow_shape s,uint32_t layer){auto c=s.condition;const auto start=2+c.points+2*uint32_t(c.hand_tokens);return {c.batch,condition_token_count(c),c.token_dim,c.context_dim,c.height/c.patch,c.width/c.patch,70,70,70,start,start+70,9,10,layer,s.depth};}
weight_map subset(const weight_map &p,const std::vector<std::pair<std::string,uint64_t>> &sizes,const std::string &prefix={},bool hand=false){weight_map out;for(auto &[name,size]:sizes)out[name]=p.at(body_flow_parameter_name(prefix+name,hand));return out;}
std::vector<float> flatten(std::span<const float> x,uint32_t b,uint32_t c,uint32_t n){
    std::vector<float> out(x.size());const uint64_t size=uint64_t(c)*n;
    for(uint64_t z=0;z<b;++z)transpose_f32(x.subspan(z*size,size),std::span(out).subspan(z*size,size),c,n);
    return out;
}
}
std::string body_flow_parameter_name(const std::string &name,bool hand){
    if(!hand)return name;
    const auto dot=name.find('.');const auto module=name.substr(0,dot);
    for(const char *independent:{"init_pose","init_camera","init_to_token_mhr","prev_to_token_mhr","ray_cond_emb","keypoint_embedding","keypoint3d_embedding","keypoint_posemb_linear","keypoint_feat_linear","keypoint3d_posemb_linear","decoder","head_pose","head_camera"})
        if(module==independent)return module+"_hand"+(dot==std::string::npos?"":name.substr(dot));
    return name;
}
void validate_body_flow_shape(body_flow_shape s){
    validate_condition_shape(s.condition);auto c=s.condition;
    require(c.pose_dim==519 && c.joints==70 && c.keypoints==70 && c.keypoints3d && s.depth>=1 && s.depth<=16,"unsupported Body flow configuration");
    validate_decoder_shape(layer_shape(s,0));validate_feedback_shape(update_shape(s,0));
    validate_pose_shape({c.batch,c.token_dim,s.pose_hidden,s.pose_depth});
    validate_camera_head_shape({c.batch,c.token_dim,s.camera_hidden,s.camera_depth,70,s.camera_scale,s.intrinsics_center});
}
std::vector<std::pair<std::string,uint64_t>> body_flow_parameter_sizes(body_flow_shape s){
    validate_body_flow_shape(s);auto c=s.condition;auto out=condition_parameter_sizes(c);
    auto append=[&](auto sizes,const std::string &prefix){for(auto &[name,size]:sizes)out.emplace_back(prefix+name,size);};
    for(uint32_t i=0;i<s.depth;++i)append(decoder_parameter_sizes(layer_shape(s,i)),"decoder.layers."+std::to_string(i)+".");
    out.emplace_back("decoder.norm_final.weight",c.token_dim);out.emplace_back("decoder.norm_final.bias",c.token_dim);
    append(pose_parameter_sizes({c.batch,c.token_dim,s.pose_hidden,s.pose_depth}),"head_pose.");
    out.emplace_back("head_pose.keypoint_mapping",308*(18439+127));
    append(camera_head_parameter_sizes({c.batch,c.token_dim,s.camera_hidden,s.camera_depth,70,s.camera_scale,s.intrinsics_center}),"head_camera.");
    append(feedback_parameter_sizes(update_shape(s,0)),"");
    if(s.hand_branch){
        for(auto &[name,_]:out)name=body_flow_parameter_name(name,true);
        out.emplace_back("hand_pe_layer.positional_encoding_gaussian_matrix",c.context_dim);
        out.emplace_back("head_pose_hand.local_to_world_wrist",9);
        out.emplace_back("head_pose_hand.right_wrist_coords",3);out.emplace_back("head_pose_hand.root_coords",3);
    }
    return out;
}
std::vector<float> body_decoder_norm(neural_session &session,uint32_t b,uint32_t n,uint32_t d,
    std::span<const float> values,std::span<const float> weight,std::span<const float> bias){
    require(b>=1 && b<=2 && n>=1 && n<=256 && d>=1 && d<=1280 && values.size()==uint64_t(b)*n*d && weight.size()==d && bias.size()==d,"invalid decoder norm shape");
    finite(values);finite(weight);finite(bias);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
    if(!ctx)throw std::bad_alloc();auto c=ctx.get();auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,d,b*n),w=ggml_new_tensor_1d(c,GGML_TYPE_F32,d),bias_t=ggml_new_tensor_1d(c,GGML_TYPE_F32,d);
    auto y=ggml_add(c,ggml_mul(c,ggml_norm(c,x,1e-6f),w),bias_t);ggml_set_output(y);
    auto graph=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(graph,y);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported decoder norm operation");
    auto buffer=session.allocate(c);
    if(!buffer)throw std::bad_alloc();std::vector<float> result(values.size());
    const neural_session::upload inputs[]={{x,values.data(),values.size_bytes()},{w,weight.data(),weight.size_bytes()},{bias_t,bias.data(),bias.size_bytes()}};
    const neural_session::download outputs[]={{y,result.data(),result.size()*4}};
    if(session.compute(graph,inputs,outputs)!=GGML_STATUS_SUCCESS)throw std::runtime_error("decoder norm failed");
    finite(result);return result;
}
named_floats body_forward_decoder(neural_session &session,tensor_archive &archive,body_flow_shape s,
    std::span<const float> features,std::span<const float> rays,std::span<const float> cliff,
    std::span<const float> prompts,std::span<const float> previous,
    std::span<const float> box_center,std::span<const float> box_size,std::span<const float> image_size,
    std::span<const float> intrinsics,std::span<const float> affine,std::span<const float> crop_size,
    std::span<const int32_t> hand_indices,const weight_map &parameters,scalar_division arithmetic,std::span<const int32_t> nonhand_indices){
    const auto sizes=body_flow_parameter_sizes(s);auto c=s.condition;const uint64_t b=c.batch,d=c.token_dim,n=condition_token_count(c);
    require(parameters.size()==sizes.size(),"Body flow parameter set mismatch");
    for(auto &[name,size]:sizes){require(parameters.contains(name) && parameters.at(name).size()==size,"Body flow parameter shape mismatch");}
    require(box_center.size()==b*2 && box_size.size()==b && image_size.size()==b*2 && intrinsics.size()==b*9 && affine.size()==b*6 && crop_size.size()==b*2,"Body flow camera inputs mismatch");
    for(auto v:{box_center,box_size,image_size,intrinsics,affine,crop_size})finite(v);
    require(hand_indices.size()==54,"Body flow hand indices mismatch");
    require(s.hand_branch?nonhand_indices.size()==145:nonhand_indices.empty(),"Body flow nonhand mask mismatch");
    auto weights=[&](auto sizes,const std::string &prefix=std::string{}){return subset(parameters,sizes,prefix,s.hand_branch);};
    auto get=[&](const std::string &name)->const validated_weights&{return parameters.at(body_flow_parameter_name(name,s.hand_branch));};
    const auto dense=s.hand_branch?std::span<const float>(parameters.at("hand_pe_layer.positional_encoding_gaussian_matrix")):std::span<const float>{};
    hand_pose_config hand;
    if(s.hand_branch)hand={get("head_pose.local_to_world_wrist"),get("head_pose.right_wrist_coords"),get("head_pose.root_coords"),nonhand_indices};
    // Captures retain their original NCHW boundaries. Normal inference keeps
    // the camera and dense position encoding in their native token layout.
    const bool channels_last=!s.capture_boundaries;
    auto conditioned=body_condition(session,c,features,rays,cliff,prompts,previous,weights(condition_parameter_sizes(c)),arithmetic,dense,channels_last);
    named_floats result;if(s.capture_boundaries)for(auto &[name,v]:conditioned)result["condition."+name]=v;
    auto tokens=conditioned.at("30.tokens"),augment=conditioned.at("31.token_pe");
    const auto grid=(c.height/c.patch)*(c.width/c.patch);
    auto context=channels_last?std::move(conditioned.at("20.image")):flatten(conditioned.at("20.image"),c.batch,c.context_dim,grid);
    // Feedback samples the original conditioned image, never updated two-way
    // context. Reuse the existing contiguous-channel layout; only two-way
    // callers need a separate immutable copy before context changes.
    const auto feedback_context=s.twoway?context:std::vector<float>{};
    const auto context_pe=channels_last?std::move(conditioned.at("21.image_pe")):flatten(conditioned.at("21.image_pe"),1,c.context_dim,grid);
    // A one-way decoder reads the same image and positional embedding in all
    // six layers. Keep one immutable device snapshot for this invocation only.
    std::unique_ptr<decoder_image_snapshot> resident_image;
    if(!s.twoway)resident_image=std::make_unique<decoder_image_snapshot>(session,layer_shape(s,0),context,context_pe);
    std::vector<float> initial_pose(b*519),initial_camera(b*3);
    for(uint32_t z=0;z<b;++z){std::copy_n(get("init_pose.weight").begin(),519,initial_pose.begin()+z*519);std::copy_n(get("init_camera.weight").begin(),3,initial_camera.begin()+z*3);}
    const pose_shape ps{c.batch,c.token_dim,s.pose_hidden,s.pose_depth};
    auto pose_weights=weights(pose_parameter_sizes(ps),"head_pose.");
    camera_head_shape cs{c.batch,c.token_dim,s.camera_hidden,s.camera_depth,70,s.camera_scale,s.intrinsics_center};
    auto camera_weights=weights(camera_head_parameter_sizes(cs),"head_camera.");
    auto feedback_weights=weights(feedback_parameter_sizes(update_shape(s,0)));
    std::vector<int32_t> indices(70);std::iota(indices.begin(),indices.end(),0);
    for(uint32_t i=0;i<s.depth;++i){
        const auto prefix="layer."+std::to_string(i)+".";auto ls=layer_shape(s,i);
        auto layer=body_decoder_layer(session,ls,tokens,resident_image?std::span<const float>{}:std::span<const float>(context),augment,
            resident_image?std::span<const float>{}:std::span<const float>(context_pe),{},weights(decoder_parameter_sizes(ls),"decoder.layers."+std::to_string(i)+"."),s.capture_decoder_operations,s.twoway,resident_image.get());
        if(s.capture_decoder_operations)for(auto &[name,v]:layer)result[prefix+"decoder."+name]=v;
        tokens=std::move(layer.at("90.tokens"));
        // A one-way decoder never changes its image context. Keep the original
        // host value instead of downloading the same tensor after every layer.
        if(s.twoway)context=std::move(layer.at("91.context"));
        if(s.capture_boundaries){result[prefix+"00.tokens"]=tokens;result[prefix+"01.context"]=context;}
        auto normalized=body_decoder_norm(session,c.batch,uint32_t(n),c.token_dim,tokens,get("decoder.norm_final.weight"),get("decoder.norm_final.bias"));
        if(s.capture_boundaries)result[prefix+"02.normalized"]=normalized;std::vector<float> token(b*d);
        for(uint32_t z=0;z<b;++z)std::copy_n(normalized.begin()+z*n*d,d,token.begin()+z*d);
        // Original callback ignores prev_pose_output for both residuals.
        auto pose=body_pose_geometry(session,archive,ps,token,initial_pose,hand_indices,pose_weights,get("head_pose.keypoint_mapping"),arithmetic,s.hand_branch?&hand:nullptr);
        // Head operation taps have their own full capture. The flow contract
        // uses the same 23 boundary fields for body and hand heads.
        if(s.hand_branch)std::erase_if(pose,[](const auto &entry){const auto &key=entry.first;
            return key.starts_with("pose.") && key!="pose.10.pred" && key!="pose.24.shape" && key!="pose.25.scale" && key!="pose.26.hand" && key!="pose.27.face" && key!="pose.90.model_params";});
        const auto &world=pose.at("map.92.keypoints");
        auto camera=body_camera_head(session,cs,token,initial_camera,world,box_center,box_size,image_size,intrinsics,camera_weights);
        if(s.capture_boundaries || i+1==s.depth){
            auto vs=cs;vs.points=18439;
            auto vertices=body_camera_project(session,vs,camera.at("10.pred_cam"),pose.at("map.90.vertices"),box_center,box_size,image_size,intrinsics);
            for(auto &[name,v]:pose)result[prefix+"pose."+name]=v;
            for(auto &[name,v]:camera)result[prefix+"camera."+name]=v;
            result[prefix+"03.vertex_pixels"]=std::move(vertices.at("20.pixels"));
        }
        if(i+1<s.depth){
            auto feedback=body_feedback(session,update_shape(s,i),s.twoway?feedback_context:context,tokens,augment,camera.at("20.pixels"),camera.at("17.depth"),world,affine,crop_size,indices,indices,feedback_weights,false,true);
            if(s.capture_boundaries)result[prefix+"04.crop_points"]=feedback.at("02.crop_points");
            tokens=std::move(feedback.at("90.tokens"));augment=std::move(feedback.at("91.augment"));
            if(s.capture_boundaries){result[prefix+"05.feedback_tokens"]=tokens;result[prefix+"06.feedback_augment"]=augment;}
        }else{
            if(s.capture_boundaries)result[prefix+"04.crop_points"]=std::move(body_full_to_crop(session,c.batch,70,camera.at("20.pixels"),affine,crop_size).at("02.crop_points"));
            if(c.hand_tokens){std::vector<float> hands(b*2*d);for(uint32_t z=0;z<b;++z)std::copy_n(normalized.begin()+(z*n+2+c.points)*d,2*d,hands.begin()+z*2*d);result["90.output_tokens"]=std::move(hands);}
            else result["90.output_tokens"]=std::move(normalized);
        }
    }
    return result;
}
}
