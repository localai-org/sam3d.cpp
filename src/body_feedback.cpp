// Copyright (c) Meta Platforms, Inc. and affiliates.
// SAM3DBody crop/keypoint feedback adaptation; SAM license, THIRD_PARTY_NOTICES.md.
#include "body_feedback.hpp"
#include "finite.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sam3d {
namespace {
void require(bool b,const char *s){if(!b)throw std::invalid_argument(s);}
void finite(std::span<const float> v){require(all_finite_f32(v),"non-finite feedback tensor");}
struct graph_run {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx;
    std::map<std::string,ggml_tensor *> taps;
    std::vector<std::pair<ggml_tensor *,std::span<const float>>> uploads;
    bool capture_all=true;
    graph_run():ctx(ggml_init({ggml_tensor_overhead()*256+ggml_graph_overhead_custom(256,false),nullptr,true}),ggml_free){if(!ctx)throw std::bad_alloc();}
    ggml_context *c(){return ctx.get();}
    ggml_tensor *weight(neural_session &session,const validated_weights &v,int64_t d,int64_t n,int64_t b=1){
        if(d<1 || n<1 || b<1 || uint64_t(n)>v.size()/uint64_t(b))throw std::invalid_argument("feedback weight shape mismatch");
        return ggml_reshape_3d(c(),session.parameter(c(),v,d,n*b),d,n,b);
    }
    ggml_tensor *input(std::span<const float> v,int64_t d,int64_t n,int64_t b=1){auto t=ggml_new_tensor_3d(c(),GGML_TYPE_F32,d,n,b);uploads.emplace_back(t,v);return t;}
    ggml_tensor *tap(const std::string &name,ggml_tensor *t){
        if(capture_all || name.ends_with(".2.output") || name=="21.feature_update"){taps[name]=t;ggml_set_output(t);}
        ggml_set_name(t,name.c_str());return t;
    }
    ggml_tensor *mm(ggml_tensor *w,ggml_tensor *x){auto y=ggml_mul_mat(c(),w,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);return y;}
    named_floats run(neural_session &session){auto graph=ggml_new_graph_custom(c(),256,false);for(auto &[name,t]:taps)ggml_build_forward_expand(graph,t);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported feedback graph operation");
        auto buffer=session.allocate(c());
        if(!buffer)throw std::bad_alloc();auto result=session.evaluate_f32(graph,uploads,taps);
        for(auto &[_,v]:result)finite(v);return result;}
};
}
named_floats body_full_to_crop(neural_session &session,uint32_t batch,uint32_t count,
    std::span<const float> pixels,std::span<const float> affine,std::span<const float> size){
    require(batch>=1 && batch<=2 && count>=1 && count<=308 && pixels.size()==uint64_t(batch)*count*2 && affine.size()==batch*6 && size.size()==batch*2,"invalid crop-projection dimensions");
    finite(pixels);finite(affine);finite(size);for(float v:size)require(v>0 && v<=32766,"invalid crop dimensions");
    std::vector<float> homogeneous(uint64_t(batch)*count*3);
    for(uint64_t i=0;i<uint64_t(batch)*count;++i){homogeneous[i*3]=pixels[i*2];homogeneous[i*3+1]=pixels[i*2+1];homogeneous[i*3+2]=1.f;}
    graph_run g;auto x=g.tap("00.homogeneous",g.input(homogeneous,3,count,batch)),a=g.input(affine,3,2,batch);
    auto crop=g.tap("01.crop_pixels",g.mm(a,x));auto dimensions=g.input(size,2,1,batch);
    g.tap("02.crop_points",ggml_scale_bias(g.c(),ggml_div(g.c(),crop,dimensions),1.f,-.5f));return g.run(session);
}
void validate_feedback_shape(feedback_shape s){
    require(s.batch>=1 && s.batch<=2 && s.tokens>=1 && s.tokens<=256 && s.dim>=1 && s.dim<=1280 && s.context_dim>=1 && s.context_dim<=1280 &&
        s.height>=1 && s.height<=32 && s.width>=1 && s.width<=32 && s.points>=1 && s.points<=308 && s.keypoints>=1 && s.keypoints<=70 && s.keypoints3d<=70 &&
        s.start2d<=s.tokens && s.keypoints<=s.tokens-s.start2d && s.start3d<=s.tokens && s.keypoints3d<=s.tokens-s.start3d &&
        s.hip_left<s.points && s.hip_right<s.points && s.depth>=1 && s.depth<=16 && s.layer<s.depth,"invalid feedback shape");
    require(s.keypoints3d==0 || s.start2d+s.keypoints<=s.start3d || s.start3d+s.keypoints3d<=s.start2d,"overlapping 2D/3D token ranges");
}
std::vector<std::pair<std::string,uint64_t>> feedback_parameter_sizes(feedback_shape s){
    validate_feedback_shape(s);const uint64_t d=s.dim;std::vector<std::pair<std::string,uint64_t>> result;
    auto ffn=[&](const std::string &name,uint64_t input){result.emplace_back(name+".layers.0.0.weight",d*input);result.emplace_back(name+".layers.0.0.bias",d);
        result.emplace_back(name+".layers.1.weight",d*d);result.emplace_back(name+".layers.1.bias",d);};
    ffn("keypoint_posemb_linear",2);result.emplace_back("keypoint_feat_linear.weight",d*s.context_dim);result.emplace_back("keypoint_feat_linear.bias",d);
    if(s.keypoints3d)ffn("keypoint3d_posemb_linear",3);return result;
}
named_floats body_feedback(neural_session &session,feedback_shape s,
    std::span<const float> image,std::span<const float> tokens,std::span<const float> augment,
    std::span<const float> pixels,std::span<const float> depths,std::span<const float> world,
    std::span<const float> affine,std::span<const float> crop_size,std::span<const int32_t> indices2d,
    std::span<const int32_t> indices3d,const weight_map &parameters,bool capture_all,bool image_channels_last){
    const auto sizes=feedback_parameter_sizes(s);const uint64_t b=s.batch,k=s.keypoints,j=s.points,d=s.dim,c=s.context_dim,hw=s.height*s.width;
    require(image.size()==b*c*hw && tokens.size()==b*s.tokens*d && augment.size()==tokens.size() && pixels.size()==b*j*2 && depths.size()==b*j && world.size()==b*j*3 &&
        indices2d.size()==k && indices3d.size()==s.keypoints3d,"feedback input size mismatch");
    for(auto v:{image,tokens,augment,pixels,depths,world})finite(v);
    for(auto list:{indices2d,indices3d})for(int32_t idx:list)require(idx>=0 && uint32_t(idx)<s.points,"feedback keypoint index out of range");
    require(parameters.size()==sizes.size(),"feedback parameter set mismatch");for(auto &[name,size]:sizes){require(parameters.contains(name) && parameters.at(name).size()==size,"feedback parameter shape mismatch");}
    auto result=body_full_to_crop(session,s.batch,s.points,pixels,affine,crop_size);
    auto &output=result["90.tokens"];output.assign(tokens.begin(),tokens.end());auto &positions=result["91.augment"];positions.assign(augment.begin(),augment.end());
    if(s.layer+1==s.depth)return result;
    auto &selected=result["10.selected_2d"];selected.resize(b*k*2);auto &selected_depth=result["11.selected_depth"];selected_depth.resize(b*k);
    auto &invalid=result["12.invalid"];invalid.resize(b*k);auto &grid=result["13.grid"];grid.resize(b*k*2);
    auto &sampled=result["14.sampled"];sampled.resize(b*k*c,0.f);auto &masked=result["15.masked_features"];masked.resize(b*k*c);
    for(uint64_t batch=0;batch<b;++batch)for(uint64_t q=0;q<k;++q){const uint64_t target=batch*k+q,source=batch*j+indices2d[q];
        const float x=result.at("02.crop_points")[source*2],y=result.at("02.crop_points")[source*2+1];
        selected[target*2]=x;selected[target*2+1]=y;selected_depth[target]=depths[source];
        const float px=x+.5f,py=y+.5f;invalid[target]=(px<0 || px>1 || py<0 || py>1 || depths[source]<1e-5f)?1.f:0.f;
        grid[target*2]=x*2.f;grid[target*2+1]=y*2.f;finite(std::span(grid).subspan(target*2,2));
        // align_corners=False, zero padding: edge coordinates see half a pixel
        // outside the image. Do not clamp to the nearest border feature.
        const float ix=((grid[target*2]+1.f)*float(s.width)-1.f)*.5f;
        const float iy=((grid[target*2+1]+1.f)*float(s.height)-1.f)*.5f;
        require(std::isfinite(ix) && std::isfinite(iy),"sampling coordinate overflow");
        if(ix>-1 && ix<float(s.width) && iy>-1 && iy<float(s.height)){
            const int x0=int(std::floor(ix)),y0=int(std::floor(iy));
            const float weights[4]={(x0+1-ix)*(y0+1-iy),(ix-x0)*(y0+1-iy),(x0+1-ix)*(iy-y0),(ix-x0)*(iy-y0)};
            for(int corner=0;corner<4;++corner){const int xx=x0+(corner%2),yy=y0+(corner/2);
                if(xx>=0 && xx<int(s.width) && yy>=0 && yy<int(s.height)){
                    const uint64_t pixel=uint64_t(yy)*s.width+xx;
                    if(image_channels_last){
                        const auto *source=image.data()+(batch*hw+pixel)*c;
                        for(uint64_t channel=0;channel<c;++channel)
                            sampled[target*c+channel]+=source[channel]*weights[corner];
                    }else for(uint64_t channel=0;channel<c;++channel)
                        sampled[target*c+channel]+=image[(batch*c+channel)*hw+pixel]*weights[corner];
                }}
        }
        for(uint64_t channel=0;channel<c;++channel)masked[target*c+channel]=sampled[target*c+channel]*(1.f-invalid[target]);
    }
    graph_run g;g.capture_all=capture_all;
    auto linear=[&](ggml_tensor *v,const std::string &name,uint64_t input){auto w=g.weight(session,parameters.at(name+".weight"),input,d),bias=g.weight(session,parameters.at(name+".bias"),d,1);
        return ggml_add(g.c(),g.mm(w,v),bias);};
    auto ffn=[&](std::span<const float> values,const std::string &name,const std::string &tap,uint64_t input,uint64_t count){
        auto x=g.input(values,input,count);auto first=g.tap(tap+".0.linear",linear(x,name+".layers.0.0",input));
        auto relu=g.tap(tap+".1.relu",ggml_relu(g.c(),first));g.tap(tap+".2.output",linear(relu,name+".layers.1",d));};
    ffn(selected,"keypoint_posemb_linear","20.pose2d",2,b*k);
    g.tap("21.feature_update",linear(g.input(masked,c,b*k),"keypoint_feat_linear",c));
    if(s.keypoints3d){auto &relative=result["30.relative_3d"];relative.resize(b*s.keypoints3d*3);
        for(uint64_t batch=0;batch<b;++batch)for(uint64_t q=0;q<s.keypoints3d;++q)for(uint64_t axis=0;axis<3;++axis){
            const float pelvis=(world[(batch*j+s.hip_left)*3+axis]+world[(batch*j+s.hip_right)*3+axis])*.5f;
            relative[(batch*s.keypoints3d+q)*3+axis]=world[(batch*j+indices3d[q])*3+axis]-pelvis;}
        finite(relative);ffn(relative,"keypoint3d_posemb_linear","31.pose3d",3,b*s.keypoints3d);
    }
    auto learned=g.run(session);for(auto &[name,v]:learned)result[name]=std::move(v);
    const auto &pose2d=result.at("20.pose2d.2.output"),&feature_update=result.at("21.feature_update");
    const std::vector<float> empty_pose3d;
    const auto &pose3d=s.keypoints3d?result.at("31.pose3d.2.output"):empty_pose3d;
    for(uint64_t batch=0;batch<b;++batch){for(uint64_t q=0;q<k;++q)for(uint64_t channel=0;channel<d;++channel){
        const uint64_t dst=(batch*s.tokens+s.start2d+q)*d+channel,src=(batch*k+q)*d+channel;
        positions[dst]=pose2d[src]*(1.f-invalid[batch*k+q]);
        // Invalid samples were zeroed BEFORE linear: its bias still contributes.
        output[dst]+=feature_update[src];}
        for(uint64_t q=0;q<s.keypoints3d;++q)for(uint64_t channel=0;channel<d;++channel)
            positions[(batch*s.tokens+s.start3d+q)*d+channel]=pose3d[(batch*s.keypoints3d+q)*d+channel];
    }
    finite(output);finite(positions);return result;
}
}
