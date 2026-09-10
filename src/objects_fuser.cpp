// Copyright (c) Meta Platforms, Inc. and affiliates.
// Adapted EmbedderFuser/FeedForward eval graph; SAM License.
#include "objects_fuser.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
namespace sam3d {
namespace {
void require(bool b,const char *m){if(!b)throw std::invalid_argument(m);}
void finite(std::span<const float> v){require(std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}),"nonfinite fusion tensor");}
uint32_t width(const fuser_shape &s){return *std::max_element(s.embed_dims.begin(),s.embed_dims.end());}
uint32_t groups(const fuser_shape &s){uint32_t n=0;for(auto x:s.inputs)if(x.position>=0)n=std::max(n,uint32_t(x.position)+1);return n;}
uint32_t hidden(uint32_t d,double multiplier){auto initial=uint32_t(multiplier*d);auto h=uint32_t(2.*initial/3.);return 256*((h+255)/256);}
}
void validate_fuser_shape(const fuser_shape &s){
    require(s.batch>=1 && s.batch<=2 && !s.embed_dims.empty() && s.embed_dims.size()<=8 && !s.inputs.empty() && s.inputs.size()<=16,"invalid fusion counts");
    for(auto d:s.embed_dims)require(d>=1 && d<=2048,"invalid fusion width");
    for(auto m:{s.projection_multiplier,s.compression_multiplier})require(std::isfinite(m) && m>=0 && m<=8 && (m==0 || hidden(width(s),m)>0),"invalid fusion projection multiplier");
    uint32_t next=0,previous=0,total=0,original_width=0,actual_width=0;
    for(auto x:s.inputs){
        require(x.embedder<s.embed_dims.size() && x.embedder>=previous && x.tokens>=1 && x.tokens<=4096,"invalid fusion modality/order");previous=x.embedder;total+=x.tokens;
        require(x.position>=-1 && x.position<=int32_t(next),"noncanonical fusion position ID");if(x.position==int32_t(next))++next;
        original_width+=s.embed_dims[x.embedder];actual_width+=s.projection_multiplier>0?width(s):s.embed_dims[x.embedder];
        if(s.compression_multiplier>0)require(x.tokens==s.inputs[0].tokens,"compression requires equal token counts");
        else if(s.projection_multiplier==0)require(s.embed_dims[x.embedder]==width(s),"unprojected token concatenation requires equal widths");
        if(x.position>=0 && s.projection_multiplier==0)require(s.embed_dims[x.embedder]==width(s),"position width differs from unprojected modality");
    }
    require(total<=16384,"too many fusion tokens");
    require(s.compression_multiplier==0 || original_width==actual_width,"original compression projector width mismatch");
}
std::vector<std::pair<std::string,uint64_t>> fuser_parameter_sizes(const fuser_shape &s){
    validate_fuser_shape(s);uint32_t d=width(s);std::vector<std::pair<std::string,uint64_t>> out{{"idx_emb",uint64_t(groups(s)+1)*d}};
    auto projection=[&](const std::string &p,uint32_t input,double mult){uint32_t h=hidden(d,mult);
        if(s.pre_norm){out.emplace_back(p+".0.weight",input);out.emplace_back(p+".0.bias",input);}
        out.emplace_back(p+".1.w1.weight",uint64_t(input)*h);out.emplace_back(p+".1.w3.weight",uint64_t(input)*h);out.emplace_back(p+".1.w2.weight",uint64_t(h)*d);
    };
    if(s.projection_multiplier>0)for(size_t i=0;i<s.embed_dims.size();++i)projection("projection_nets."+std::to_string(i),s.embed_dims[i],s.projection_multiplier);
    if(s.compression_multiplier>0){uint32_t input=0;for(auto x:s.inputs)input+=s.embed_dims[x.embedder];projection("compression_projector",input,s.compression_multiplier);}
    std::sort(out.begin(),out.end());return out;
}
named_floats objects_fuse(neural_session &session,const fuser_shape &s,const std::vector<std::vector<float>> &input,const named_floats &params,bool capture){
    auto sizes=fuser_parameter_sizes(s);require(params.size()==sizes.size() && input.size()==s.inputs.size(),"fusion input/parameter count mismatch");
    for(auto &[key,n]:sizes){require(params.contains(key) && params.at(key).size()==n,"fusion parameter extent mismatch");finite(params.at(key));}
    for(size_t i=0;i<input.size();++i){require(input[i].size()==uint64_t(s.batch)*s.inputs[i].tokens*s.embed_dims[s.inputs[i].embedder],"fusion input extent mismatch");finite(input[i]);}
    constexpr size_t nodes=2048;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({ggml_tensor_overhead()*nodes+ggml_graph_overhead_custom(nodes,false),nullptr,true}),ggml_free);
    if(!ctx)throw std::bad_alloc();auto c=ctx.get();uint32_t d=width(s);
    std::map<std::string,ggml_tensor *> weights,taps;std::vector<ggml_tensor *> inputs;
    auto tap=[&](const std::string &key,ggml_tensor *v){if(capture || key=="90.output"){ggml_set_output(v);taps.emplace(key,v);}return v;};
    auto weight=[&](const std::string &key,uint32_t a,uint32_t b=1){if(weights.contains(key))return weights.at(key);auto v=ggml_new_tensor_2d(c,GGML_TYPE_F32,a,b);weights.emplace(key,v);return v;};
    auto linear=[&](ggml_tensor *x,const std::string &key,uint32_t output){auto v=ggml_mul_mat(c,weight(key,x->ne[0],output),x);ggml_mul_mat_set_prec(v,GGML_PREC_F32);return v;};
    auto project=[&](ggml_tensor *x,const std::string &p,const std::string &prefix,double mult){
        tap(prefix+".00.input",x);
        if(s.pre_norm)x=ggml_add(c,ggml_mul(c,ggml_norm(c,x,1e-5f),weight(p+".0.weight",x->ne[0])),weight(p+".0.bias",x->ne[0]));
        tap(prefix+".01.norm",x);uint32_t h=hidden(d,mult);
        auto a=tap(prefix+".02.w1",linear(x,p+".1.w1.weight",h));
        auto silu=tap(prefix+".03.silu",ggml_silu(c,a));auto b=tap(prefix+".04.w3",linear(x,p+".1.w3.weight",h));
        auto gated=tap(prefix+".05.gated",ggml_mul(c,silu,b));return tap(prefix+".06.projected",linear(gated,p+".1.w2.weight",d));
    };
    auto idx=weight("idx_emb",d,groups(s)+1);std::vector<ggml_tensor *> tokens;
    for(size_t i=0;i<input.size();++i){auto x=s.inputs[i];std::string prefix="10.modality."+std::to_string(i);
        auto v=ggml_new_tensor_3d(c,GGML_TYPE_F32,s.embed_dims[x.embedder],x.tokens,s.batch);inputs.push_back(v);tap(prefix+".input",v);
        if(s.projection_multiplier>0)v=project(v,"projection_nets."+std::to_string(x.embedder),prefix+".projection",s.projection_multiplier);
        if(x.position>=0){auto pos=ggml_view_1d(c,idx,d,uint64_t(x.position)*d*4);v=ggml_add(c,v,pos);}
        tap(prefix+".positioned",v);
        // Multiply rather than replace, preserving upstream signed-zero semantics.
        v=tap(prefix+".dropped",ggml_scale(c,v,x.forced_drop?0.f:1.f));tokens.push_back(v);
    }
    auto joined=tokens[0];for(size_t i=1;i<tokens.size();++i)joined=ggml_concat(c,joined,tokens[i],s.compression_multiplier>0?0:1);
    tap("20.joined",joined);auto output=s.compression_multiplier>0?project(joined,"compression_projector","30.compression",s.compression_multiplier):joined;
    tap("90.output",output);auto graph=ggml_new_graph_custom(c,nodes,false);ggml_build_forward_expand(graph,output);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported fusion operation");
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(c,session.backend()),ggml_backend_buffer_free);if(!buffer)throw std::bad_alloc();
    for(auto &[key,v]:weights)ggml_backend_tensor_set(v,params.at(key).data(),0,ggml_nbytes(v));
    for(size_t i=0;i<inputs.size();++i)ggml_backend_tensor_set(inputs[i],input[i].data(),0,input[i].size()*4);
    if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("fusion graph failed");
    named_floats result;for(auto &[key,v]:taps){auto &out=result[key];out.resize(ggml_nelements(v));ggml_backend_tensor_get(v,out.data(),0,out.size()*4);finite(out);}return result;
}
}
