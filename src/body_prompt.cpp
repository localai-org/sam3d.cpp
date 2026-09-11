// Copyright (c) Meta Platforms, Inc. and affiliates.
// SAM 3D Body PromptEncoder adaptation; SAM license, NOTICE.
#include "body_prompt.hpp"
#include "transpose.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace sam3d {
namespace {
void require(bool b,const char *message) {if(!b) throw std::invalid_argument(message);}
void finite(std::span<const float> values) {
    require(std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}),"non-finite prompt tensor");
}
float divide(float value,uint32_t divisor,scalar_division mode) {
    require(mode==scalar_division::direct || mode==scalar_division::reciprocal_multiply,"invalid scalar division mode");
    return mode==scalar_division::direct?value/float(divisor):value*(1.f/float(divisor));
}
named_floats encode_position(neural_session &session,uint32_t dim,
    std::span<const float> coordinates,std::span<const float> gaussian,bool capture_all = true) {
    finite(coordinates);finite(gaussian);
    const int64_t n=coordinates.size()/2,f=dim/2;
    require(dim>=2 && dim<=1280 && dim%2==0 && n>=1 && n<=1024 && coordinates.size()==2*n && gaussian.size()==dim,
        "invalid position encoding dimensions");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({
        ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true}),ggml_free);
    if(!ctx) throw std::bad_alloc();auto c=ctx.get();
    std::map<std::string,ggml_tensor *> taps;
    auto tap=[&](const std::string &name,ggml_tensor *t) {
        ggml_set_name(t,name.c_str());
        if(capture_all || name=="07.encoding"){ggml_set_output(t);taps[name]=t;}
        return t;
    };
    auto xy=tap("00.coords",ggml_new_tensor_2d(c,GGML_TYPE_F32,2,n));
    auto matrix=ggml_new_tensor_2d(c,GGML_TYPE_F32,f,2);
    auto doubled=tap("01.doubled",ggml_scale(c,xy,2.f));
    auto centered=tap("02.centered",ggml_scale_bias(c,doubled,1.f,-1.f));
    auto projected=ggml_mul_mat(c,ggml_cont(c,ggml_transpose(c,matrix)),centered);
    ggml_mul_mat_set_prec(projected,GGML_PREC_F32);tap("03.projected",projected);
    auto angles=tap("04.angles",ggml_scale(c,projected,float(2*std::numbers::pi)));
    auto sine=tap("05.sin",ggml_sin(c,angles)),cosine=tap("06.cos",ggml_cos(c,angles));
    auto encoding=tap("07.encoding",ggml_concat(c,sine,cosine,0));
    auto graph=ggml_new_graph_custom(c,64,false);ggml_build_forward_expand(graph,encoding);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported prompt position operation");
    auto buffer=session.allocate(c);
    if(!buffer) throw std::bad_alloc();
    const std::pair<ggml_tensor *,std::span<const float>> uploads[]={{xy,coordinates},{matrix,gaussian}};
    auto result=session.evaluate_f32(graph,uploads,taps);
    for(auto &[_,v]:result)finite(v);
    return result;
}
}
void validate_prompt_shape(prompt_shape s) {
    require(s.batch>=1 && s.batch<=2 && s.points>=1 && s.points<=256 && s.dim>=2 && s.dim<=1280 && s.dim%2==0 &&
        s.joints>=1 && s.joints<=256 && s.height>=1 && s.height<=32 && s.width>=1 && s.width<=32,"invalid prompt shape");
}
std::vector<std::pair<std::string,uint64_t>> prompt_parameter_sizes(prompt_shape s) {
    validate_prompt_shape(s);
    std::vector<std::pair<std::string,uint64_t>> result{{"pe_layer.positional_encoding_gaussian_matrix",s.dim},
        {"invalid_point_embed.weight",s.dim},{"not_a_point_embed.weight",s.dim}};
    for(uint32_t j=0;j<s.joints;++j) result.emplace_back("point_embeddings."+std::to_string(j)+".weight",s.dim);
    return result;
}
named_floats body_prompt_encode(neural_session &session,prompt_shape s,std::span<const float> keypoints,
    const weight_map &parameters,scalar_division division,std::span<const float> dense_gaussian,bool capture_all,bool dense_channels_last) {
    const auto sizes=prompt_parameter_sizes(s);const uint64_t count=uint64_t(s.batch)*s.points,grid=s.height*s.width;
    require(keypoints.size()==count*3,"prompt keypoint size mismatch");finite(keypoints);
    require(parameters.size()==sizes.size(),"prompt parameter set mismatch");
    for(auto &[name,size]:sizes) {require(parameters.contains(name) && parameters.at(name).size()==size,"prompt parameter size/name mismatch");}
    std::vector<float> xy(count*2),dense_xy(grid*2);
    for(uint64_t i=0;i<count;++i) {
        for(uint64_t axis=0;axis<2;++axis) {float v=keypoints[i*3+axis];require(v>=0 && v<=1,"prompt coordinate outside [0,1]");xy[i*2+axis]=v;}
        float label=keypoints[i*3+2];require(label>=-2 && label<s.joints && std::trunc(label)==label,"invalid prompt joint label");
    }
    for(uint32_t y=0;y<s.height;++y) for(uint32_t x=0;x<s.width;++x) {
        dense_xy[(y*s.width+x)*2]=divide(float(x)+.5f,s.width,division);
        dense_xy[(y*s.width+x)*2+1]=divide(float(y)+.5f,s.height,division);
    }
    const auto &gaussian=parameters.at("pe_layer.positional_encoding_gaussian_matrix");
    // Hand image PE has its own trained Gaussian; sparse prompt PE still
    // belongs to the shared PromptEncoder. Do not replace both together.
    require(dense_gaussian.empty() || dense_gaussian.size()==s.dim,"invalid dense Gaussian shape");finite(dense_gaussian);
    auto dense=encode_position(session,s.dim,dense_xy,dense_gaussian.empty()?std::span<const float>(gaussian):dense_gaussian,capture_all),points=encode_position(session,s.dim,xy,gaussian,capture_all);
    named_floats result;
    auto &embedding=result["20.embeddings"];embedding=points.at("07.encoding");
    auto &mask=result["21.mask"];mask.resize(count);
    // Exact table selection and F32 addition; no learned computation is replaced
    // by geometric heuristics. Negative labels discard positional encoding.
    for(uint64_t i=0;i<count;++i) {
        const auto label=int(keypoints[i*3+2]);mask[i]=label>-2?1.f:0.f;
        const auto &weight=parameters.at(label==-2?"invalid_point_embed.weight":label==-1?"not_a_point_embed.weight":
            "point_embeddings."+std::to_string(label)+".weight");
        for(uint32_t channel=0;channel<s.dim;++channel)
            embedding[i*s.dim+channel]=label<0?weight[channel]:embedding[i*s.dim+channel]+weight[channel];
    }
    if(dense_channels_last){
        if(capture_all)result["22.dense_tokens"]=dense.at("07.encoding");
        else result["22.dense_tokens"]=std::move(dense.at("07.encoding"));
    }else{
        auto &nchw=result["22.dense_nchw"];nchw.resize(grid*s.dim);
        transpose_f32(dense.at("07.encoding"),nchw,grid,s.dim);
    }
    if(capture_all){
        for(auto &[name,v]:dense) result["00.dense."+name]=std::move(v);
        for(auto &[name,v]:points) result["10.point."+name]=std::move(v);
    }
    finite(embedding);return result;
}
named_floats body_position_pixels(neural_session &session,uint32_t batch,uint32_t points,uint32_t dim,
    uint32_t height,uint32_t width,std::span<const float> xy,std::span<const float> gaussian,scalar_division division) {
    require(batch>=1 && batch<=2 && points>=1 && points<=256 && height>=1 && height<=32766 && width>=1 && width<=32766 &&
        xy.size()==uint64_t(batch)*points*2,"invalid pixel position shape");finite(xy);
    std::vector<float> normalized(xy.begin(),xy.end());
    for(uint64_t i=0;i<uint64_t(batch)*points;++i) {normalized[i*2]=divide(normalized[i*2],width,division);normalized[i*2+1]=divide(normalized[i*2+1],height,division);}
    return encode_position(session,dim,normalized,gaussian);
}
}
