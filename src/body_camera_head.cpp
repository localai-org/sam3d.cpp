// Copyright (c) Meta Platforms, Inc. and affiliates.
// PerspectiveHead/geometry_utils adaptation; SAM license, THIRD_PARTY_NOTICES.md.
#include "body_camera_head.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sam3d {
namespace {
void require(bool v,const char *s){if(!v)throw std::invalid_argument(s);}
void finite(std::span<const float> v){require(std::all_of(v.begin(),v.end(),[](float f){return std::isfinite(f);}),"non-finite camera-head tensor");}
std::string layer_name(uint32_t i,uint32_t depth){return "proj.layers."+std::to_string(i)+(i+1<depth?".0":"");}
}
void validate_camera_head_shape(camera_head_shape s){
    require(s.batch>=1 && s.batch<=2 && s.dim>=1 && s.dim<=1280 && s.hidden>=1 && s.hidden<=1280 &&
        s.depth>=1 && s.depth<=3 && s.points>=1 && s.points<=100000 && std::isfinite(s.scale_factor) && s.scale_factor>0,
        "invalid camera-head shape/scale");
}
std::vector<std::pair<std::string,uint64_t>> camera_head_parameter_sizes(camera_head_shape s){
    validate_camera_head_shape(s);std::vector<std::pair<std::string,uint64_t>> result;uint64_t in=s.dim;
    for(uint32_t i=0;i<s.depth;++i){const uint64_t out=i+1==s.depth?3:s.hidden;auto name=layer_name(i,s.depth);
        result.emplace_back(name+".weight",in*out);result.emplace_back(name+".bias",out);in=out;}
    return result;
}
static named_floats camera_impl(neural_session &session,camera_head_shape s,
    std::span<const float> token,std::span<const float> initial,std::span<const float> points,
    std::span<const float> box_center,std::span<const float> box_size,std::span<const float> image_size,
    std::span<const float> intrinsics,const weight_map &parameters,bool projection_only){
    validate_camera_head_shape(s);
    const auto sizes=projection_only?std::vector<std::pair<std::string,uint64_t>>{}:camera_head_parameter_sizes(s);const uint64_t b=s.batch,n=s.points;
    require(token.size()==b*(projection_only?3:s.dim) && (initial.empty() || (!projection_only && initial.size()==b*3)) && points.size()==b*n*3 &&
        box_center.size()==b*2 && box_size.size()==b && image_size.size()==b*2 && intrinsics.size()==b*9,"camera-head input size mismatch");
    for(auto v:{token,initial,points,box_center,box_size,image_size,intrinsics})finite(v);
    for(uint32_t i=0;i<b;++i){require(box_size[i]>0 && image_size[i*2]>0 && image_size[i*2+1]>0,"camera image/box dimensions must be positive");
        require(intrinsics[i*9]>0 && intrinsics[i*9+4]>0 && intrinsics[i*9+6]==0 && intrinsics[i*9+7]==0 && intrinsics[i*9+8]==1,"invalid pinhole intrinsics");}
    require(parameters.size()==sizes.size(),"camera-head parameter set mismatch");
    for(auto &[name,size]:sizes){require(parameters.contains(name) && parameters.at(name).size()==size,"camera-head parameter shape mismatch");}
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({
        ggml_tensor_overhead()*256+ggml_graph_overhead_custom(256,false),nullptr,true}),ggml_free);
    if(!ctx)throw std::bad_alloc();auto c=ctx.get();std::map<std::string,ggml_tensor *> taps;
    std::vector<std::pair<ggml_tensor *,std::span<const float>>> uploads;
    auto tensor=[&](std::span<const float> v,uint64_t d,uint64_t rows,uint64_t batch){auto t=ggml_new_tensor_3d(c,GGML_TYPE_F32,d,rows,batch);uploads.emplace_back(t,v);return t;};
    auto tap=[&](const std::string &name,ggml_tensor *t){ggml_set_output(t);ggml_set_name(t,name.c_str());taps[name]=t;return t;};
    auto mm=[&](ggml_tensor *a,ggml_tensor *x){auto out=ggml_mul_mat(c,a,x);ggml_mul_mat_set_prec(out,GGML_PREC_F32);return out;};
    auto value=tensor(token,projection_only?3:s.dim,1,b);uint64_t in=s.dim;
    for(uint32_t i=0;i<(projection_only?0:s.depth);++i){const uint64_t out=i+1==s.depth?3:s.hidden;auto name=layer_name(i,s.depth);
        auto weight=session.parameter(c,parameters.at(name+".weight"),in,out),bias=session.parameter(c,parameters.at(name+".bias"),out,1);
        value=tap("00.ffn."+std::to_string(i)+".linear",ggml_add(c,mm(weight,value),bias));
        if(i+1<s.depth)value=tap("00.ffn."+std::to_string(i)+".relu",ggml_relu(c,value));in=out;
    }
    if(!initial.empty())value=ggml_add(c,value,tensor(initial,3,1,b));
    tap("10.pred_cam",value);
    // F32 views preserve B,N,C layout with channels on GGML's first axis.
    auto channel=[&](ggml_tensor *t,int64_t index){return ggml_cont(c,ggml_view_3d(c,t,1,t->ne[1],t->ne[2],t->nb[1],t->nb[2],index*4));};
    auto scale=ggml_neg(c,channel(value,0)),tx=channel(value,1),ty=ggml_neg(c,channel(value,2));
    auto corrected=tap("11.corrected_cam",ggml_concat(c,ggml_concat(c,scale,tx,0),ty,0));
    (void)corrected;
    auto box=tensor(box_size,1,1,b),center=tensor(box_center,2,1,b),size=tensor(image_size,2,1,b);
    // Each multiplication/addition is separate, as in original Tensor ops.
    auto scaled_box=ggml_scale(c,ggml_mul(c,box,scale),s.scale_factor);
    auto bs=tap("12.scaled_box",ggml_scale_bias(c,scaled_box,1.f,1e-8f));
    // Original K is [output,input] row-major, already GGML's [input,output]
    // storage convention for mul_mat. Do NOT transpose it a second time.
    std::vector<float> fx(b),principal(b*2);
    for(uint32_t i=0;i<b;++i){fx[i]=intrinsics[i*9];principal[i*2]=intrinsics[i*9+2];principal[i*2+1]=intrinsics[i*9+5];}
    auto focal=tap("13.focal",tensor(fx,1,1,b));
    auto origin=s.intrinsics_center?tensor(principal,2,1,b):ggml_scale(c,size,.5f);
    auto offset=tap("14.offset",ggml_div(c,ggml_scale(c,ggml_sub(c,center,origin),2.f),bs));
    auto depth=ggml_div(c,ggml_scale(c,focal,2.f),bs);
    auto translation=tap("15.translation",ggml_concat(c,ggml_add(c,ggml_concat(c,tx,ty,0),offset),depth,0));
    auto camera_points=tap("16.camera_points",ggml_add(c,tensor(points,3,n,b),translation));
    auto point_depth=tap("17.depth",channel(camera_points,2));
    auto normalized=tap("18.normalized",ggml_div(c,camera_points,point_depth));
    auto matrix=tensor(intrinsics,3,3,b);
    auto projected=tap("19.intrinsic_projection",mm(matrix,normalized));
    auto pixels=tap("20.pixels",ggml_cont(c,ggml_view_3d(c,projected,2,n,b,projected->nb[1],projected->nb[2],0)));
    auto graph=ggml_new_graph_custom(c,256,false);ggml_build_forward_expand(graph,pixels);
    // The corrected camera tap is diagnostic and otherwise not an ancestor.
    ggml_build_forward_expand(graph,corrected);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported camera-head operation");
    auto buffer=session.allocate(c);
    if(!buffer)throw std::bad_alloc();auto result=session.evaluate_f32(graph,uploads,taps);
    for(auto &[_,v]:result)finite(v);
    return result;
}
named_floats body_camera_head(neural_session &session,camera_head_shape s,
    std::span<const float> token,std::span<const float> initial,std::span<const float> points,
    std::span<const float> box_center,std::span<const float> box_size,std::span<const float> image_size,
    std::span<const float> intrinsics,const weight_map &parameters){
    return camera_impl(session,s,token,initial,points,box_center,box_size,image_size,intrinsics,parameters,false);
}
named_floats body_camera_project(neural_session &session,camera_head_shape s,
    std::span<const float> camera,std::span<const float> points,std::span<const float> box_center,
    std::span<const float> box_size,std::span<const float> image_size,std::span<const float> intrinsics){
    return camera_impl(session,s,camera,{},points,box_center,box_size,image_size,intrinsics,{},true);
}
}
