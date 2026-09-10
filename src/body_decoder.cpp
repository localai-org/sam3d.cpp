// Copyright (c) Meta Platforms, Inc. and affiliates.
// SAM 3D Body transformer adaptation; SAM license, THIRD_PARTY_NOTICES.md.
#include "body_decoder.hpp"
#include "finite.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace sam3d {
namespace {
void require(bool b,const char *s) { if (!b) throw std::invalid_argument(s); }
void finite(std::span<const float> v) {
    require(all_finite_f32(v),"non-finite decoder tensor");
}
}
void validate_decoder_shape(decoder_shape s) {
    require(s.batch>=1 && s.batch<=2 && s.tokens>=1 && s.tokens<=256 &&
        s.context_tokens>=1 && s.context_tokens<=1024,"invalid decoder token shape");
    require(s.token_dim>=1 && s.token_dim<=1280 && s.context_dim>=1 && s.context_dim<=1280 &&
        s.heads>=1 && s.heads<=20 && s.head_dim>=1 && s.head_dim<=128 &&
        s.heads*s.head_dim<=1280 && s.hidden>=1 && s.hidden<=4096,"invalid decoder channel shape");
}
std::vector<std::pair<std::string,uint64_t>> decoder_parameter_sizes(decoder_shape s) {
    validate_decoder_shape(s);
    std::vector<std::pair<std::string,uint64_t>> sizes;
    auto norm=[&](const std::string &n,uint64_t d){sizes.emplace_back(n+".weight",d);sizes.emplace_back(n+".bias",d);};
    auto linear=[&](const std::string &n,uint64_t in,uint64_t out){sizes.emplace_back(n+".weight",in*out);sizes.emplace_back(n+".bias",out);};
    auto attention=[&](const std::string &n,uint64_t q,uint64_t kv) {
        const uint64_t e=s.heads*s.head_dim;
        linear(n+".q_proj",q,e); linear(n+".k_proj",kv,e); linear(n+".v_proj",kv,e); linear(n+".proj",e,q);
    };
    if(s.repeat_pe) {norm("ln_pe_1",s.token_dim);norm("ln_pe_2",s.context_dim);}
    norm("ln1",s.token_dim);attention("self_attn",s.token_dim,s.token_dim);
    norm("ln2_1",s.token_dim);norm("ln2_2",s.context_dim);attention("cross_attn",s.token_dim,s.context_dim);
    norm("ln3",s.token_dim);linear("ffn.layers.0.0",s.token_dim,s.hidden);linear("ffn.layers.1",s.hidden,s.token_dim);
    if(s.twoway) {norm("ln4_1",s.context_dim);norm("ln4_2",s.token_dim);attention("cross_attn_2",s.context_dim,s.token_dim);}
    return sizes;
}
struct decoder_image_snapshot::impl {
    neural_session *session;
    decoder_shape shape;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context{nullptr,ggml_free};
    neural_session::scratch_buffer buffer;
    ggml_tensor *image=nullptr,*pe=nullptr;
};
decoder_image_snapshot::decoder_image_snapshot(neural_session &session,decoder_shape s,
    std::span<const float> image,std::span<const float> pe):impl_(std::make_unique<impl>()){
    validate_decoder_shape(s);
    const uint64_t plane=uint64_t(s.context_tokens)*s.context_dim;
    require(image.size()==s.batch*plane,"resident decoder image size mismatch");
    require(pe.empty() || pe.size()==plane || pe.size()==s.batch*plane,"resident decoder PE size mismatch");
    finite(image);finite(pe);
    auto &p=*impl_;p.session=&session;p.shape=s;
    p.context.reset(ggml_init({ggml_tensor_overhead()*4,nullptr,true}));if(!p.context)throw std::bad_alloc();
    p.image=ggml_new_tensor_3d(p.context.get(),GGML_TYPE_F32,s.context_dim,s.context_tokens,s.batch);
    if(!pe.empty())p.pe=ggml_new_tensor_3d(p.context.get(),GGML_TYPE_F32,s.context_dim,s.context_tokens,pe.size()/plane);
    p.buffer=session.allocate(p.context.get());if(!p.buffer)throw std::bad_alloc();
    ggml_backend_tensor_set(p.image,image.data(),0,image.size_bytes());
    if(p.pe)ggml_backend_tensor_set(p.pe,pe.data(),0,pe.size_bytes());
}
decoder_image_snapshot::~decoder_image_snapshot()=default;
bool decoder_image_snapshot::has_position() const{return impl_->pe!=nullptr;}
std::pair<ggml_tensor *,ggml_tensor *> decoder_image_snapshot::views(neural_session &session,decoder_shape s,ggml_context *c) const{
    const auto &p=*impl_;
    require(c && p.session==&session && s.batch==p.shape.batch && s.context_tokens==p.shape.context_tokens &&
        s.context_dim==p.shape.context_dim,"resident decoder image session/shape mismatch");
    return {ggml_view_tensor(c,p.image),p.pe?ggml_view_tensor(c,p.pe):nullptr};
}
named_floats body_decoder_layer(neural_session &session,decoder_shape s,
    std::span<const float> input,std::span<const float> image,
    std::span<const float> token_pe,std::span<const float> image_pe,
    std::span<const float> mask,const weight_map &parameters,bool capture_all,bool return_context,const decoder_image_snapshot *resident_image) {
    const auto sizes=decoder_parameter_sizes(s);
    const int64_t b=s.batch,n=s.tokens,m=s.context_tokens,d=s.token_dim,cx=s.context_dim,e=s.heads*s.head_dim;
    require(input.size()==b*n*d && (resident_image?image.empty():image.size()==b*m*cx),"decoder input size mismatch");
    require(!resident_image || image_pe.empty(),"resident decoder PE requires empty host input");
    require(token_pe.empty() || token_pe.size()==n*d || token_pe.size()==b*n*d,"decoder token PE size mismatch");
    require(image_pe.empty() || image_pe.size()==m*cx || image_pe.size()==b*m*cx,"decoder image PE size mismatch");
    require(!s.repeat_pe || !(resident_image?resident_image->has_position():!image_pe.empty()) || !token_pe.empty(),"decoder image PE requires token PE");
    require(mask.empty() || mask.size()==b*n,"decoder mask size mismatch");
    for(float v:mask) require(v==0 || v==1,"decoder mask must contain zero/one");
    for(auto v:{input,image,token_pe,image_pe}) finite(v);
    require(parameters.size()==sizes.size(),"decoder parameter set mismatch");
    for(auto &[name,size]:sizes) {
        require(parameters.contains(name) && parameters.at(name).size()==size,"decoder parameter shape mismatch");
    }
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({
        ggml_tensor_overhead()*1024+ggml_graph_overhead_custom(1024,false),nullptr,true}),ggml_free);
    if(!ctx) throw std::bad_alloc();
    auto c=ctx.get(); std::map<std::string,ggml_tensor *> p,taps;
    std::vector<std::pair<ggml_tensor *,std::span<const float>>> uploads;
    auto tensor=[&](std::span<const float> values,int64_t dim,int64_t count,int64_t batch) {
        auto t=ggml_new_tensor_3d(c,GGML_TYPE_F32,dim,count,batch);uploads.emplace_back(t,values);return t;
    };
    for(auto &[name,size]:sizes) p[name]=session.parameter(c,parameters.at(name),size,1);
    auto x=tensor(input,d,n,b);
    auto xp=token_pe.empty()?nullptr:tensor(token_pe,d,n,token_pe.size()/(n*d));
    const auto image_views=resident_image?resident_image->views(session,s,c):
        std::pair{tensor(image,cx,m,b),image_pe.empty()?nullptr:tensor(image_pe,cx,m,image_pe.size()/(m*cx))};
    auto context=image_views.first,cp=image_views.second;
    auto tap=[&](std::string name,ggml_tensor *t) {
        ggml_set_name(t,name.c_str());
        if(capture_all || name=="90.tokens" || (return_context && name=="91.context")) {ggml_set_output(t);taps.emplace(name,t);}
        return t;
    };
    auto mm=[&](ggml_tensor *a,ggml_tensor *v) {
        auto t=ggml_mul_mat(c,a,v);ggml_mul_mat_set_prec(t,GGML_PREC_F32);return t;
    };
    auto linear=[&](ggml_tensor *v,const std::string &name,int64_t in,int64_t out) {
        return ggml_add(c,mm(ggml_reshape_2d(c,p.at(name+".weight"),in,out),v),p.at(name+".bias"));
    };
    auto norm=[&](ggml_tensor *v,const std::string &name) {
        return ggml_add(c,ggml_mul(c,ggml_norm(c,v,1e-6f),p.at(name+".weight")),p.at(name+".bias"));
    };
    // A masked-out token still attends to itself in the self-attention pass.
    // SDPA returns zero for all-masked rows in reverse attention. Avoid a NaN
    // softmax by permitting one dummy key, then zeroing that row's probabilities.
    std::vector<float> self_mask, reverse_mask, reverse_keep;
    if(!mask.empty()) {
        self_mask.resize(b*n*n);reverse_mask.resize(b*m*n);reverse_keep.resize(b);
        for(int64_t z=0;z<b;++z) {
            bool any=false;for(int64_t k=0;k<n;++k) any|=mask[z*n+k]>0;
            reverse_keep[z]=any?1.f:0.f;
            for(int64_t q=0;q<n;++q) for(int64_t k=0;k<n;++k)
                self_mask[(z*n+q)*n+k]=(q==k || (mask[z*n+q]>0 && mask[z*n+k]>0))?0.f:-std::numeric_limits<float>::infinity();
            for(int64_t q=0;q<m;++q) for(int64_t k=0;k<n;++k)
                reverse_mask[(z*m+q)*n+k]=(mask[z*n+k]>0 || (!any && k==0))?0.f:-std::numeric_limits<float>::infinity();
        }
    }
    auto attention=[&](const std::string &stage,const std::string &name,ggml_tensor *q,ggml_tensor *k,ggml_tensor *v,
                       const std::vector<float> &attention_mask,bool reverse) {
        const int64_t nq=q->ne[1],nk=k->ne[1],dq=q->ne[0],dk=k->ne[0];
        tap(stage+".00.q_input",q);tap(stage+".01.k_input",k);tap(stage+".02.v_input",v);
        auto heads=[&](ggml_tensor *value,int64_t count) {
            return ggml_cont(c,ggml_permute(c,ggml_reshape_4d(c,value,s.head_dim,s.heads,count,b),0,2,1,3));
        };
        q=tap(stage+".03.q",heads(linear(q,name+".q_proj",dq,e),nq));
        k=tap(stage+".04.k",heads(linear(k,name+".k_proj",dk,e),nk));
        v=tap(stage+".05.v",heads(linear(v,name+".v_proj",dk,e),nk));
        // Match the observed PyTorch math-SDPA order: scale each operand by
        // sqrt(1/sqrt(head_dim)) before GEMM, not the product afterwards.
        const float operand_scale=float(std::sqrt(1.0/std::sqrt(double(s.head_dim))));
        auto logits=tap(stage+".06.logits",mm(ggml_scale(c,k,operand_scale),ggml_scale(c,q,operand_scale)));
        if(!attention_mask.empty()) {
            auto a=ggml_new_tensor_4d(c,GGML_TYPE_F32,nk,nq,1,b);uploads.emplace_back(a,attention_mask);
            logits=ggml_add(c,logits,a);
        }
        auto probs=ggml_soft_max(c,logits);
        if(reverse && !attention_mask.empty()) {
            auto keep=ggml_new_tensor_4d(c,GGML_TYPE_F32,1,1,1,b);uploads.emplace_back(keep,reverse_keep);
            probs=ggml_mul(c,probs,keep);
        }
        tap(stage+".07.probs",probs);
        auto attended=mm(ggml_cont(c,ggml_transpose(c,v)),probs);
        attended=tap(stage+".08.attended",ggml_reshape_3d(c,ggml_cont(c,ggml_permute(c,attended,0,2,1,3)),e,nq,b));
        return tap(stage+".09.output",linear(attended,name+".proj",e,dq));
    };
    if(s.repeat_pe && cp) {xp=tap("00.token_pe",norm(xp,"ln_pe_1"));cp=tap("01.context_pe",norm(cp,"ln_pe_2"));}
    auto normalized=tap("02.ln1",norm(x,"ln1"));
    auto q=(s.repeat_pe && !s.skip_first_pe && xp)?ggml_add(c,normalized,xp):normalized;
    x=tap("20.self_residual",ggml_add(c,x,attention("10.self","self_attn",q,q,normalized,self_mask,false)));
    auto qn=tap("21.ln2_1",norm(x,"ln2_1")),kn=tap("22.ln2_2",norm(context,"ln2_2"));
    q=(s.repeat_pe && cp)?ggml_add(c,qn,xp):qn;
    auto k=(s.repeat_pe && cp)?ggml_add(c,kn,cp):kn;
    x=tap("40.cross_residual",ggml_add(c,x,attention("30.cross","cross_attn",q,k,kn,{},false)));
    auto f=tap("41.ln3",norm(x,"ln3"));
    f=tap("42.ffn_linear1",linear(f,"ffn.layers.0.0",d,s.hidden));
    f=tap("43.ffn_gelu",ggml_gelu_erf(c,f));
    f=tap("44.ffn_linear2",linear(f,"ffn.layers.1",s.hidden,d));
    x=tap("45.ffn_residual",ggml_add(c,x,f));
    if(s.twoway) {
        qn=tap("50.ln4_1",norm(context,"ln4_1"));kn=tap("51.ln4_2",norm(x,"ln4_2"));
        q=(s.repeat_pe && cp)?ggml_add(c,qn,cp):qn;k=(s.repeat_pe && cp)?ggml_add(c,kn,xp):kn;
        context=ggml_add(c,context,attention("60.reverse","cross_attn_2",q,k,kn,reverse_mask,true));
    }
    tap("90.tokens",x);tap("91.context",context);
    auto graph=ggml_new_graph_custom(c,1024,false);ggml_build_forward_expand(graph,x);ggml_build_forward_expand(graph,context);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto node=ggml_graph_node(graph,i);
        if(!ggml_backend_supports_op(session.backend(),node))
            throw std::runtime_error(std::string("unsupported decoder operation: ")+ggml_op_name(node->op));
    }
    auto buffer=session.allocate(c);
    if(!buffer) throw std::bad_alloc();
    auto result=session.evaluate_f32(graph,uploads,taps);
    for(auto &[_,v]:result)finite(v);
    return result;
}
}
