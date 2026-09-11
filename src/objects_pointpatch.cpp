// Copyright (c) Meta Platforms, Inc. and affiliates.
// PointPatchEmbed adaptation: SAM License; NOTICE.
// Window Block semantics follow Ross Wightman's timm 0.9.16 (Apache-2.0).
// Copyright 2019 Ross Wightman. Modified: C++/GGML graph and bounded chunks.
#include "objects_pointpatch.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tuple>
namespace sam3d {
namespace {
void require(bool b,const char *m){if(!b)throw std::invalid_argument(m);}
void finite(std::span<const float> a){require(std::all_of(a.begin(),a.end(),[](float v){return std::isfinite(v);}),"nonfinite PointPatch parameter/output");}
struct graph {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx{ggml_init({ggml_tensor_overhead()*256+ggml_graph_overhead_custom(256,false),nullptr,true}),ggml_free};
    std::map<std::string,ggml_tensor *> taps;
    std::vector<std::pair<ggml_tensor *,std::span<const float>>> uploads;
    graph(){if(!ctx)throw std::bad_alloc();}
    ggml_context *c(){return ctx.get();}
    ggml_tensor *input(std::span<const float> v,int64_t d,int64_t n=1,int64_t b=1){
        auto t=ggml_new_tensor_3d(c(),GGML_TYPE_F32,d,n,b);require(ggml_nelements(t)==int64_t(v.size()),"PointPatch graph input shape mismatch");uploads.emplace_back(t,v);return t;
    }
    ggml_tensor *tap(const std::string &k,ggml_tensor *v,bool enabled){if(enabled){ggml_set_output(v);ggml_set_name(v,k.c_str());taps[k]=v;}return v;}
    ggml_tensor *mm(ggml_tensor *w,ggml_tensor *x){auto v=ggml_mul_mat(c(),w,x);ggml_mul_mat_set_prec(v,GGML_PREC_F32);return v;}
    named_floats run(neural_session &session,ggml_tensor *out){
        auto g=ggml_new_graph_custom(c(),256,false);ggml_build_forward_expand(g,out);
        for(int i=0;i<ggml_graph_n_nodes(g);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(g,i)),"unsupported PointPatch operation");
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(c(),session.backend()),ggml_backend_buffer_free);
        if(!buffer)throw std::bad_alloc();
        for(auto &[t,v]:uploads)ggml_backend_tensor_set(t,v.data(),0,v.size_bytes());
        if(ggml_backend_graph_compute(session.backend(),g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("PointPatch graph failed");
        named_floats result;for(auto &[key,t]:taps){auto &v=result[key];v.resize(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);}return result;
    }
};
std::vector<float> projection(neural_session &session,std::span<const float> xyz,uint32_t d,const named_floats &p){
    graph g;auto x=g.input(xyz,3,xyz.size()/3),w=g.input(p.at("point_proj.weight"),3,d),bias=g.input(p.at("point_proj.bias"),d);
    auto out=g.tap("projection",ggml_add(g.c(),g.mm(w,x),bias),true);
    return std::move(g.run(session,out).at("projection"));
}
named_floats block(neural_session &session,uint32_t b,uint32_t n,uint32_t d,std::span<const float> tokens,const named_floats &p,bool observe){
    graph g;auto c=g.c();const uint32_t dh=d/16;auto x=g.input(tokens,d,n,b);
    auto param=[&](const std::string &name,int64_t a,int64_t z=1){return g.input(p.at("blocks.0."+name),a,z);};
    auto linear=[&](ggml_tensor *v,const std::string &name,uint32_t in,uint32_t out){return ggml_add(c,g.mm(param(name+".weight",in,out),v),param(name+".bias",out));};
    auto norm=[&](ggml_tensor *v,const std::string &name){return ggml_add(c,ggml_mul(c,ggml_norm(c,v,1e-6f),param(name+".weight",d)),param(name+".bias",d));};
    auto tap=[&](const std::string &key,ggml_tensor *v){return g.tap(key,v,observe || key=="30.block_output");};
    auto norm1=tap("11.norm1",norm(x,"norm1"));
    auto qkv=tap("12.qkv",linear(norm1,"attn.qkv",d,3*d));
    auto head=[&](uint32_t i){auto v=ggml_view_4d(c,qkv,dh,16,n,b,dh*4,3*d*4,3*d*n*4,i*d*4);return ggml_cont(c,ggml_permute(c,v,0,2,1,3));};
    auto q=tap("13.q",head(0)),k=tap("14.k",head(1)),v=tap("15.v",head(2));
    auto logits=tap("16.logits",ggml_scale(c,g.mm(k,q),1.f/std::sqrt(float(dh))));
    auto probs=tap("17.probs",ggml_soft_max(c,logits));
    auto attn=g.mm(ggml_cont(c,ggml_transpose(c,v)),probs);
    attn=tap("18.attention",ggml_reshape_3d(c,ggml_cont(c,ggml_permute(c,attn,0,2,1,3)),d,n,b));
    auto projected=tap("19.attn_projected",linear(attn,"attn.proj",d,d));
    auto residual=tap("20.residual",ggml_add(c,x,projected));
    auto norm2=tap("21.norm2",norm(residual,"norm2"));
    auto hidden=tap("22.fc1",linear(norm2,"mlp.fc1",d,2*d));
    auto act=tap("23.gelu",ggml_gelu_erf(c,hidden));
    auto fc2=tap("24.fc2",linear(act,"mlp.fc2",2*d,d));
    auto out=tap("30.block_output",ggml_add(c,residual,fc2));
    auto result=g.run(session,out);for(auto &[key,values]:result)finite(values);return result;
}
std::array<float,3> remap(std::array<float,3> x,point_remapping mode){
    if(mode==point_remapping::sinh){for(auto &v:x)v=std::asinh(v);}
    else if(mode==point_remapping::exp){float den=1.f+x[2];x={x[0]/den,x[1]/den,std::log1p(x[2])};}
    else if(mode==point_remapping::sinh_exp){x={std::asinh(x[0]),std::asinh(x[1]),std::log(std::max(x[2],1e-8f))};}
    else if(mode==point_remapping::exp_disparity){x={x[0]/x[2],x[1]/x[2],std::log(x[2])};}
    return x;
}
}
void validate_pointpatch_shape(pointpatch_shape s){
    require(s.batch>=1 && s.batch<=2 && s.height>=1 && s.width>=1 && s.height<=2048 && s.width<=2048,"invalid PointPatch input shape");
    require(s.side>=1 && s.side<=256 && s.patch>=1 && s.patch<=16 && s.side%s.patch==0 && s.dim>=16 && s.dim<=1024 && s.dim%16==0,"invalid PointPatch architecture");
    require(uint32_t(s.remap)<=uint32_t(point_remapping::exp_disparity) && (!s.force_dropout || s.dropout_enabled),"invalid PointPatch remap/dropout mode");
}
std::vector<std::pair<std::string,uint64_t>> pointpatch_parameter_sizes(pointpatch_shape s){
    validate_pointpatch_shape(s);uint64_t d=s.dim,g=s.side/s.patch;
    std::vector<std::pair<std::string,uint64_t>> out={{"point_proj.weight",3*d},{"point_proj.bias",d},{"invalid_xyz_token",d},{"pos_embed",d*g*g},{"pos_embed_window",(1+s.patch*s.patch)*d},{"cls_token",d}};
    if(s.dropout_enabled)out.emplace_back("dropped_xyz_token",d);
    for(auto name:{"norm1","norm2"}){out.emplace_back(std::string("blocks.0.")+name+".weight",d);out.emplace_back(std::string("blocks.0.")+name+".bias",d);}
    for(auto &[name,in,n]:std::vector<std::tuple<std::string,uint64_t,uint64_t>>{{"attn.qkv",d,3*d},{"attn.proj",d,d},{"mlp.fc1",d,2*d},{"mlp.fc2",2*d,d}}){out.emplace_back("blocks.0."+name+".weight",in*n);out.emplace_back("blocks.0."+name+".bias",n);}
    std::sort(out.begin(),out.end());return out;
}
std::vector<float> objects_pointpatch(neural_session &session,pointpatch_shape s,std::span<const float> xyz,std::span<const uint8_t> mask,const named_floats &p,const pointpatch_observer &observe,uint32_t chunk){
    auto sizes=pointpatch_parameter_sizes(s);require(chunk>=1 && chunk<=64,"invalid PointPatch chunk size");
    require(xyz.size()==uint64_t(s.batch)*3*s.height*s.width && (mask.empty() || mask.size()==uint64_t(s.batch)*s.side*s.side),"PointPatch input/mask extent mismatch");
    for(auto v:mask)require(v<=1,"PointPatch mask must be binary");
    require(p.size()==sizes.size(),"PointPatch parameter set mismatch");for(auto &[key,n]:sizes){require(p.contains(key) && p.at(key).size()==n,"PointPatch parameter shape mismatch");finite(p.at(key));}
    const uint32_t g=s.side/s.patch,pp=s.patch*s.patch,n=pp+1,d=s.dim,windows=s.batch*g*g;
    std::vector<float> result(uint64_t(windows)*d);
    for(uint32_t first=0;first<windows;first+=chunk){
        const auto b=std::min(chunk,windows-first);
        auto emit=[&](const std::string &key,std::span<const float> values){if(observe){require(values.size()%b==0,"PointPatch diagnostic shape mismatch");const auto stride=values.size()/b;observe(key,values,uint64_t(first)*stride,uint64_t(windows)*stride);}};
        std::vector<float> resized(uint64_t(b)*pp*3),valid(uint64_t(b)*pp),safe(resized.size()),mapped(resized.size());
        for(uint32_t i=0;i<b;++i){auto win=first+i;auto batch=win/(g*g),wy=(win/g)%g,wx=win%g;
            for(uint32_t py=0;py<s.patch;++py)for(uint32_t px=0;px<s.patch;++px){auto y=wy*s.patch+py,x=wx*s.patch+px,point=(i*pp+py*s.patch+px);
                auto iy=std::min(uint32_t(std::floor(y*(float(s.height)/s.side))),s.height-1),ix=std::min(uint32_t(std::floor(x*(float(s.width)/s.side))),s.width-1);
                std::array<float,3> v;for(uint32_t c=0;c<3;++c)v[c]=xyz[(uint64_t(batch)*3+c)*s.height*s.width+uint64_t(iy)*s.width+ix];
                bool ok=mask.empty()?std::all_of(v.begin(),v.end(),[](float z){return std::isfinite(z);}):mask[(uint64_t(batch)*s.side+y)*s.side+x]!=0;
                valid[point]=ok?1.f:0.f;for(uint32_t c=0;c<3;++c)resized[point*3+c]=v[c];
                if(!ok)v={0,0,0};else finite(v);
                for(uint32_t c=0;c<3;++c)safe[point*3+c]=v[c];v=remap(v,s.remap);
                if(ok)finite(v);for(uint32_t c=0;c<3;++c)mapped[point*3+c]=v[c];
            }
        }
        emit("00.resized",resized);emit("01.valid",valid);emit("02.safe",safe);emit("03.remapped",mapped);
        auto projected=projection(session,mapped,d,p);emit("04.projected",projected);
        for(uint32_t i=0;i<b*pp;++i)if(!valid[i])std::copy_n(p.at("invalid_xyz_token").begin(),d,projected.begin()+uint64_t(i)*d);
        finite(projected);emit("05.embedded",projected);
        std::vector<float> tokens(uint64_t(b)*n*d);
        for(uint32_t i=0;i<b;++i){std::copy_n(p.at("cls_token").begin(),d,tokens.begin()+uint64_t(i)*n*d);std::copy_n(projected.begin()+uint64_t(i)*pp*d,uint64_t(pp)*d,tokens.begin()+(uint64_t(i)*n+1)*d);}
        emit("06.cls_joined",tokens);
        for(uint32_t i=0;i<b;++i)for(uint32_t j=0;j<n*d;++j)tokens[uint64_t(i)*n*d+j]+=p.at("pos_embed_window")[j];emit("10.window_tokens",tokens);
        auto block_taps=block(session,b,n,d,tokens,p,bool(observe));for(auto &[key,v]:block_taps)emit(key,v);
        auto &output=block_taps.at("30.block_output");std::vector<float> cls(uint64_t(b)*d),pe(cls.size()),positioned(cls.size()),final(cls.size());
        for(uint32_t i=0;i<b;++i)for(uint32_t j=0;j<d;++j){auto at=uint64_t(i)*d+j;cls[at]=output[uint64_t(i)*n*d+j];pe[at]=p.at("pos_embed")[uint64_t(j)*g*g+(first+i)%(g*g)];positioned[at]=cls[at]+pe[at];final[at]=s.force_dropout?p.at("dropped_xyz_token")[j]+pe[at]:positioned[at];}
        finite(final);emit("31.cls",cls);emit("32.position",pe);emit("33.positioned",positioned);emit("90.output",final);
        std::copy(final.begin(),final.end(),result.begin()+uint64_t(first)*d);
    }
    return result;
}
}
