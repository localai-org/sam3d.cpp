// Copyright (c) Meta Platforms, Inc. and affiliates.
// CameraEncoder adaptation: SAM license, see THIRD_PARTY_NOTICES.md.
// Antialiased interpolation follows PyTorch v2.7: LICENSES/PyTorch.txt.
#include "camera_encoder.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace sam3d {
namespace {
void require(bool b,const char *s) { if (!b) throw std::invalid_argument(s); }
void finite(std::span<const float> values) {
    require(std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}),"non-finite camera tensor");
}
struct filter { uint32_t first; std::vector<float> weights; };
std::vector<filter> axis_filters(uint32_t input,uint32_t patch) {
    std::vector<filter> result(input/patch);
    const float scale=patch, inverse=1.f/scale;
    for (uint32_t i=0;i<result.size();++i) {
        auto &f=result[i]; const float center=scale*(i+.5);
        f.first=std::max<int64_t>(int64_t(center-scale+.5),0);
        const auto end=std::min<int64_t>(int64_t(center+scale+.5),input);
        float sum=0;
        for (auto j=f.first;j<end;++j) {
            const float distance=(j-center+.5)*inverse;
            const float w=std::max(0.f,1.f-std::abs(distance));
            f.weights.push_back(w); sum+=w;
        }
        require(sum>0,"empty antialias filter");
        for (auto &weight:f.weights) weight/=sum;
    }
    return result;
}
std::vector<float> downsample(camera_shape s,std::span<const float> input) {
    if (s.patch==1) return {input.begin(),input.end()};
    const auto horizontal=axis_filters(s.width,s.patch), vertical=axis_filters(s.height,s.patch);
    const uint64_t w=s.width/s.patch,h=s.height/s.patch;
    std::vector<float> intermediate(uint64_t(s.batch)*2*s.height*w), result(uint64_t(s.batch)*2*h*w);
    for (uint32_t plane=0;plane<s.batch*2;++plane) {
        for (uint32_t y=0;y<s.height;++y) for (uint32_t x=0;x<w;++x) {
            const auto &f=horizontal[x];
            const auto offset=(uint64_t(plane)*s.height+y)*s.width+f.first;
            float value=input[offset]*f.weights[0];
            for (size_t j=1;j<f.weights.size();++j) value+=input[offset+j]*f.weights[j];
            intermediate[(uint64_t(plane)*s.height+y)*w+x]=value;
        }
        for (uint32_t y=0;y<h;++y) for (uint32_t x=0;x<w;++x) {
            const auto &f=vertical[y];
            const auto offset=(uint64_t(plane)*s.height+f.first)*w+x;
            float value=intermediate[offset]*f.weights[0];
            for (size_t j=1;j<f.weights.size();++j) value+=intermediate[offset+j*w]*f.weights[j];
            result[(uint64_t(plane)*h+y)*w+x]=value;
        }
    }
    finite(result); return result;
}
}
void validate_camera_shape(camera_shape s) {
    require(s.batch>=1 && s.batch<=2 && s.patch>=1 && s.patch<=32 && s.dim>=1 && s.dim<=1280,
            "invalid camera batch/patch/channels");
    require(s.height>=s.patch && s.height<=512 && s.width>=s.patch && s.width<=512 &&
            s.height%s.patch==0 && s.width%s.patch==0 && s.height/s.patch<=32 && s.width/s.patch<=32,
            "invalid camera image/grid shape");
}
named_floats camera_encode(neural_session &session,camera_shape s,std::span<const float> features,
    std::span<const float> rays,const weight_map &parameters,bool capture_all,bool channels_last) {
    validate_camera_shape(s);
    const uint64_t n=(s.height/s.patch)*(s.width/s.patch),d=s.dim,b=s.batch;
    require(features.size()==b*n*d && rays.size()==b*2*s.height*s.width,"camera input size mismatch");
    finite(features); finite(rays);
    require(parameters.size()==3,"camera parameter count mismatch");
    for (auto &[name,count]:std::vector<std::pair<std::string,uint64_t>>{
            {"conv.weight",d*(d+99)},{"norm.weight",d},{"norm.bias",d}}) {
        require(parameters.contains(name) && parameters.at(name).size()==count,"camera parameter shape mismatch");
    }
    named_floats result;
    auto &sampled=result["00.rays_downsampled"]; sampled=downsample(s,rays); // B,2,H',W'
    auto &pos=result["01.positions"]; pos.resize(b*n*3);
    for (uint64_t batch=0;batch<b;++batch) for (uint64_t token=0;token<n;++token) {
        pos[(batch*n+token)*3]=sampled[(batch*2)*n+token];
        pos[(batch*n+token)*3+1]=sampled[(batch*2+1)*n+token];
        pos[(batch*n+token)*3+2]=1.f;
    }
    // PyTorch linspace uses symmetric endpoint evaluation for F32 frequencies.
    auto &frequencies=result["02.frequencies"]; frequencies.resize(48);
    const float step=31.f/15.f;
    for (uint32_t axis=0;axis<3;++axis) for (uint32_t i=0;i<16;++i)
        frequencies[axis*16+i]=i<8?1.f+step*i:32.f-step*(15-i);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({
        ggml_tensor_overhead()*128+ggml_graph_overhead_custom(128,false),nullptr,true}),ggml_free);
    if (!context) throw std::bad_alloc();
    auto c=context.get(); std::map<std::string,ggml_tensor *> taps;
    auto tap=[&](const std::string &name,ggml_tensor *value) {
        ggml_set_name(value,name.c_str());
        if(capture_all || name=="07.output"){ggml_set_output(value);taps[name]=value;}
        return value;
    };
    auto p=ggml_new_tensor_3d(c,GGML_TYPE_F32,1,3,b*n);
    auto f=ggml_new_tensor_2d(c,GGML_TYPE_F32,16,3);
    auto a=ggml_mul(c,ggml_repeat(c,p,ggml_new_tensor_3d(c,GGML_TYPE_F32,16,3,b*n)),f);
    auto angles=ggml_scale(c,ggml_reshape_2d(c,a,48,b*n),float(std::numbers::pi));
    auto trig=ggml_concat(c,ggml_sin(c,angles),ggml_cos(c,angles),0);
    auto encoding=tap("03.fourier",ggml_concat(c,ggml_reshape_2d(c,p,3,b*n),trig,0));
    // Pure layout conversion on the selected backend; avoid two large host
    // transposes per inference while preserving the public NCHW boundary.
    auto image_storage=ggml_new_tensor_3d(c,GGML_TYPE_F32,n,d,b);
    auto image=ggml_reshape_2d(c,ggml_cont(c,ggml_permute(c,image_storage,1,0,2,3)),d,b*n);
    auto joined=tap("04.joined",ggml_concat(c,image,encoding,0));
    auto weight=session.parameter(c,parameters.at("conv.weight"),d+99,d);
    auto projection=ggml_mul_mat(c,weight,joined); ggml_mul_mat_set_prec(projection,GGML_PREC_F32);
    tap("05.projection",projection);
    // Original LayerNorm2d is mean/squared-centered-mean/divide-sqrt, eps 1e-6.
    auto mean=ggml_mean(c,projection), centered=ggml_sub(c,projection,mean);
    auto variance=ggml_mean(c,ggml_sqr(c,centered));
    auto normalized=ggml_div(c,centered,ggml_sqrt(c,ggml_scale_bias(c,variance,1.f,1e-6f)));
    auto norm_weight=session.parameter(c,parameters.at("norm.weight"),d,1), norm_bias=session.parameter(c,parameters.at("norm.bias"),d,1);
    auto output=tap("06.normalized",ggml_add(c,ggml_mul(c,normalized,norm_weight),norm_bias));
    output=ggml_reshape_3d(c,output,d,n,b);
    output=tap("07.output",channels_last?output:ggml_cont(c,ggml_permute(c,output,1,0,2,3)));
    auto graph=ggml_new_graph_custom(c,128,false); ggml_build_forward_expand(graph,output);
    for (int i=0;i<ggml_graph_n_nodes(graph);++i)
        require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported camera operation");
    auto buffer=session.allocate(c);
    if (!buffer) throw std::bad_alloc();
    const std::pair<ggml_tensor *,std::span<const float>> uploads[]={{p,pos},{f,frequencies},{image_storage,features}};
    for(auto &[name,v]:session.evaluate_f32(graph,uploads,taps)){finite(v);result[name]=std::move(v);}
    if(!capture_all)std::erase_if(result,[](const auto &item){return item.first!="07.output";});
    return result;
}
}
