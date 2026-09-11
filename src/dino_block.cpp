// Copyright (c) Meta Platforms, Inc. and affiliates.
// DINOv3 contract adaptation: LICENSES/DINOv3.md and NOTICE.
#include "dino_block.hpp"
#include "bf16.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace sam3d {
namespace {
void require(bool b, const char *message) { if (!b) throw std::invalid_argument(message); }
void finite(std::span<const float> v) {
    require(std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}),"non-finite block tensor");
}
}
void validate_dino_shape(dino_shape s) {
    require(s.batch>=1 && s.batch<=2 && s.height>=1 && s.width>=1 &&
            s.height<=32 && s.width<=32 && s.prefix<=8,"invalid DINO token shape");
    require(s.dim>=4 && s.dim<=1280 && s.heads>=1 && s.heads<=20 &&
            s.dim%(4*s.heads)==0,"invalid DINO head shape");
    require(s.hidden>=8 && s.hidden<=5120 && s.hidden%8==0,"invalid DINO SwiGLU size");
}
std::vector<std::pair<std::string,uint64_t>> dino_parameter_sizes(dino_shape s) {
    validate_dino_shape(s);
    const uint64_t d=s.dim, f=s.hidden;
    return {{"norm1.weight",d},{"norm1.bias",d},{"attn.qkv.weight",3*d*d},
        {"attn.qkv.bias",3*d},{"attn.qkv.bias_mask",3*d},{"attn.proj.weight",d*d},
        {"attn.proj.bias",d},{"ls1.gamma",d},{"norm2.weight",d},{"norm2.bias",d},
        {"mlp.w1.weight",f*d},{"mlp.w1.bias",f},{"mlp.w2.weight",f*d},
        {"mlp.w2.bias",f},{"mlp.w3.weight",d*f},{"mlp.w3.bias",d},
        {"ls2.gamma",d},{"periods",d/s.heads/4}};
}
std::vector<float> dino_angles(dino_shape s,std::span<const float> periods) {
    validate_dino_shape(s);
    const int64_t n=s.height*s.width+s.prefix,dh=s.dim/s.heads;
    require(periods.size()==uint64_t(dh/4),"DINO periods size mismatch");finite(periods);
    for(float p:periods)require(p>0,"DINO period must be positive");
    std::vector<float> values(n*dh,0.f);
    for (uint32_t y=0;y<s.height;++y) for (uint32_t z=0;z<s.width;++z) {
        const float coords[2]={2.f*((float(y)+.5f)/float(s.height))-1.f,
                               2.f*((float(z)+.5f)/float(s.width))-1.f};
        const auto token=s.prefix+y*s.width+z;
        for(int axis=0;axis<2;++axis)for(int64_t j=0;j<dh/4;++j){
            const float a=(float(2*std::numbers::pi)*coords[axis])/periods[j];
            values[token*dh+axis*(dh/4)+j]=a;
            values[token*dh+dh/2+axis*(dh/4)+j]=a;
        }
    }
    return values;
}
named_tensors dino_block_graph(neural_session &session,ggml_context *c,dino_shape s,
    ggml_tensor *x,const named_tensors &p,ggml_tensor *angles,bool capture_all,bool bf16) {
    validate_dino_shape(s);
    const int64_t d=s.dim, n=s.height*s.width+s.prefix, dh=s.dim/s.heads;
    named_tensors taps;
    auto rounded=[&](ggml_tensor *t){return bf16?bf16_round(c,t):t;};
    auto tap=[&](std::string name,ggml_tensor *t) {
        ggml_set_name(t,name.c_str());
        if (capture_all || name=="21.output") { ggml_set_output(t); taps.emplace(name,t); }
        return t;
    };
    auto mm=[&](ggml_tensor *a,ggml_tensor *b) {
        auto product=[&](ggml_tensor *left,ggml_tensor *right) {
            auto t=ggml_mul_mat(c,left,right); ggml_mul_mat_set_prec(t,GGML_PREC_F32); return t;
        };
        // The strict-F32 Vulkan shader accumulates long dot products serially.
        // Eight independent partial sums plus a balanced reduction reduce this
        // rounding error without reduced-precision operands or CPU fallback.
        // Keep the already validated CPU path unchanged. This initial accurate
        // path materializes the slices; optimizing copies is a later benchmark.
        if (!bf16 && std::string_view(ggml_backend_name(session.backend())).starts_with("Vulkan") &&
            a->ne[0]>=1024 && a->ne[0]%8==0) {
            const auto k=a->ne[0]/8;
            ggml_tensor *partial[8];
            auto slice=[&](ggml_tensor *t,int i) {
                return ggml_cont(c,ggml_view_4d(c,t,k,t->ne[1],t->ne[2],t->ne[3],
                    t->nb[1],t->nb[2],t->nb[3],i*k*t->nb[0]));
            };
            for (int i=0;i<8;++i) partial[i]=product(slice(a,i),slice(b,i));
            for (int width=8;width>1;width/=2)
                for (int i=0;i<width/2;++i) partial[i]=ggml_add(c,partial[2*i],partial[2*i+1]);
            return partial[0];
        }
        return product(a,b);
    };
    auto linear=[&](ggml_tensor *a,const std::string &name,int64_t in,int64_t out) {
        return rounded(ggml_add(c,mm(ggml_reshape_2d(c,p.at(name+".weight"),in,out),a),p.at(name+".bias")));
    };
    auto norm=[&](ggml_tensor *a,const std::string &name) {
        return rounded(ggml_add(c,ggml_mul(c,ggml_norm(c,a,1e-5f),p.at(name+".weight")),p.at(name+".bias")));
    };
    // Model periods are loaded parameters; positions use the original eval
    // separate-axis [-1,1] normalization. Training-only rescale is NOT applied.
    auto sin=tap("00.rope_sin",ggml_sin(c,angles));
    auto cos=tap("01.rope_cos",ggml_cos(c,angles));
    auto norm1=tap("02.norm1",norm(x,"norm1"));
    auto qkv=tap("03.qkv",rounded(ggml_add(c,mm(ggml_reshape_2d(c,p.at("attn.qkv.weight"),d,3*d),norm1),
                                     rounded(ggml_mul(c,p.at("attn.qkv.bias"),p.at("attn.qkv.bias_mask"))))));
    auto head=[&](int index) {
        auto v=ggml_view_4d(c,qkv,dh,s.heads,n,s.batch,dh*4,3*d*4,3*d*n*4,index*d*4);
        return ggml_cont(c,ggml_permute(c,v,0,2,1,3));
    };
    auto q=tap("04.q",head(0)), k=tap("05.k",head(1)), v=tap("06.v",head(2));
    auto rotate=[&](ggml_tensor *a) {
        auto first=ggml_view_4d(c,a,dh/2,n,s.heads,s.batch,a->nb[1],a->nb[2],a->nb[3],0);
        auto second=ggml_view_4d(c,a,dh/2,n,s.heads,s.batch,a->nb[1],a->nb[2],a->nb[3],dh/2*4);
        auto rotated=ggml_concat(c,ggml_neg(c,second),first,0);
        return ggml_add(c,ggml_mul(c,a,cos),ggml_mul(c,rotated,sin));
    };
    q=tap("07.q_rope",rounded(rotate(q))); k=tap("08.k_rope",rounded(rotate(k)));
    // PyTorch math SDPA splits the scale between its operands before GEMM.
    // Scaling the product instead is mathematically equivalent, but not F32
    // equivalent for the large trained logits; see inspect_math_sdpa.py.
    const auto flash_env=std::getenv("SAM3D_BF16_FLASH_ATTENTION");
    const bool flash=bf16 && flash_env && std::string_view(flash_env)=="1";
    ggml_tensor *attended=nullptr;
    if(flash){
        // Experimental, explicitly selected original automatic-SDPA analogue.
        // Do not silently substitute math attention when capturing operations:
        // fused kernels do not expose their internal logits/softmax boundaries.
        require(!capture_all,"BF16 fused attention cannot expose math-SDPA operation taps");
        const auto prefix_env=std::getenv("SAM3D_BF16_PRECISE_PREFIX");
        const int64_t precise_prefix=prefix_env && std::string_view(prefix_env)=="1"?s.prefix:0;
        auto queries=q;
        ggml_tensor *prefix_output=nullptr;
        if(precise_prefix){
            // Class/storage tokens develop much larger magnitudes than image
            // patches. Use the validated math-SDPA path for these few queries,
            // retaining every key/value and the original token order. This is
            // a precision choice, not masking/removing register interactions.
            auto prefix_q=ggml_cont(c,ggml_view_4d(c,q,dh,precise_prefix,s.heads,s.batch,q->nb[1],q->nb[2],q->nb[3],0));
            const float operand_scale=float(std::sqrt(1.0/std::sqrt(double(dh))));
            auto logits=mm(ggml_scale(c,k,operand_scale),ggml_scale(c,prefix_q,operand_scale));
            auto probabilities=ggml_soft_max(c,logits);
            auto prefix_attended=mm(ggml_cont(c,ggml_transpose(c,v)),probabilities);
            prefix_output=ggml_reshape_3d(c,ggml_cont(c,ggml_permute(c,prefix_attended,0,2,1,3)),d,precise_prefix,s.batch);
            queries=ggml_cont(c,ggml_view_4d(c,q,dh,n-precise_prefix,s.heads,s.batch,q->nb[1],q->nb[2],q->nb[3],precise_prefix*q->nb[1]));
        }
        attended=ggml_flash_attn_ext(c,queries,ggml_cast(c,k,GGML_TYPE_BF16),
            ggml_cast(c,v,GGML_TYPE_BF16),nullptr,float(1.0/std::sqrt(double(dh))),0.f,0.f);
        ggml_flash_attn_ext_set_prec(attended,GGML_PREC_F32);
        require(ggml_backend_supports_op(session.backend(),attended),"backend does not support requested BF16 fused attention");
        // GGML returns [head_dim, heads, tokens, batch], already contiguous.
        attended=ggml_reshape_3d(c,attended,d,n-precise_prefix,s.batch);
        if(prefix_output)attended=ggml_concat(c,prefix_output,attended,1);
    }else{
        const float operand_scale=float(std::sqrt(1.0/std::sqrt(double(dh))));
        auto logits=tap("09.logits",mm(ggml_scale(c,k,operand_scale),ggml_scale(c,q,operand_scale)));
        auto probs=tap("10.probs",ggml_soft_max(c,logits));
        attended=mm(ggml_cont(c,ggml_transpose(c,v)),probs);
        attended=ggml_reshape_3d(c,ggml_cont(c,ggml_permute(c,attended,0,2,1,3)),d,n,s.batch);
    }
    attended=rounded(attended);
    tap("11.attention",attended);
    auto projected=tap("12.attn_proj",linear(attended,"attn.proj",d,d));
    auto scaled=tap("13.ls1",rounded(ggml_mul(c,projected,p.at("ls1.gamma"))));
    auto residual=tap("14.residual1",rounded(ggml_add(c,x,scaled)));
    auto norm2=tap("15.norm2",norm(residual,"norm2"));
    auto w1=tap("16.w1",linear(norm2,"mlp.w1",d,s.hidden));
    auto w2=tap("17.w2",linear(norm2,"mlp.w2",d,s.hidden));
    auto activated=rounded(ggml_silu(c,w1));
    // Computing w2 first makes the SiLU/round/multiply/round chain adjacent in
    // the graph. BF16 carriers are F32 here: swapping equal-shaped multiply
    // operands preserves the value and both explicit rounding boundaries.
    auto multiplied=bf16?ggml_mul(c,w2,activated):ggml_mul(c,activated,w2);
    // Widen back into the dead F32 multiply result. This explicit GGML copy
    // keeps the final destination from partially overlapping a live gate
    // input when the backend fuses the six operations. CPU/unfused execution
    // still performs both ordinary casts; no precision boundary is removed.
    auto hidden=tap("18.hidden",bf16?ggml_cpy(c,ggml_cast(c,multiplied,GGML_TYPE_BF16),multiplied):multiplied);
    auto w3=tap("19.w3",linear(hidden,"mlp.w3",s.hidden,d));
    auto ls2=tap("20.ls2",rounded(ggml_mul(c,w3,p.at("ls2.gamma"))));
    auto out=tap("21.output",rounded(ggml_add(c,residual,ls2)));
    (void)out;
    return taps;
}
static named_floats dino_block_impl(neural_session &session,dino_shape s,std::span<const float> input,
    const std::map<std::string,std::span<const float>> &parameters,bool capture_all,bool bf16=false) {
    const auto sizes=dino_parameter_sizes(s);
    const int64_t d=s.dim,n=s.height*s.width+s.prefix,dh=s.dim/s.heads;
    require(input.size()==uint64_t(s.batch)*n*d,"DINO input size mismatch");finite(input);
    require(parameters.size()==sizes.size(),"DINO parameter set mismatch");
    for(const auto &[name,size]:sizes){auto it=parameters.find(name);
        require(it!=parameters.end() && it->second.size()==size,"DINO parameter size/name mismatch");}
    const auto mask=parameters.at("attn.qkv.bias_mask");
    const bool all_zero=std::all_of(mask.begin(),mask.end(),[](float v){return v==0.f;});
    for(int64_t i=0;i<3*d;++i)require(all_zero || mask[i]==(i>=d && i<2*d?0.f:1.f),"invalid DINO key bias mask");
    auto angle_values=dino_angles(s,parameters.at("periods"));
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({
        ggml_tensor_overhead()*512+ggml_graph_overhead_custom(512,false),nullptr,true}),ggml_free);
    if(!ctx)throw std::bad_alloc();auto c=ctx.get();
    named_tensors p;
    for(const auto &[name,size]:sizes)p[name]=ggml_new_tensor_1d(c,bf16 && name.ends_with(".weight") && !name.starts_with("norm")?GGML_TYPE_BF16:GGML_TYPE_F32,size);
    auto x=ggml_new_tensor_3d(c,GGML_TYPE_F32,d,n,s.batch);
    auto angles=ggml_new_tensor_2d(c,GGML_TYPE_F32,dh,n);
    auto taps=dino_block_graph(session,c,s,x,p,angles,capture_all,bf16);
    auto out=taps.at("21.output");
    auto graph=ggml_new_graph_custom(c,512,false); ggml_build_forward_expand(graph,out);
    for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto node=ggml_graph_node(graph,i);
        if (!ggml_backend_supports_op(session.backend(),node))
            throw std::runtime_error(std::string("DINO backend unsupported operation: ")+ggml_op_name(node->op));
    }
    auto buffer=session.allocate(c);
    if (!buffer) throw std::bad_alloc();
    for (auto &[name,t]:p) {
        if(t->type==GGML_TYPE_BF16)set_bf16_tensor(t,parameters.at(name));
        else ggml_backend_tensor_set(t,parameters.at(name).data(),0,ggml_nbytes(t));
    }
    ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
    ggml_backend_tensor_set(angles,angle_values.data(),0,ggml_nbytes(angles));
    if (ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("DINO block graph failed");
    named_floats result;
    for (auto &[name,t]:taps) {
        auto &values=result[name]; values.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t)); finite(values);
    }
    return result;
}
named_floats dino_block(neural_session &session,dino_shape s,std::span<const float> input,
                       const named_floats &parameters,bool capture_all) {
    std::map<std::string,std::span<const float>> views;
    for(auto &[name,values]:parameters){finite(values);views.emplace(name,values);}
    return dino_block_impl(session,s,input,views,capture_all);
}
named_floats dino_block(neural_session &session,dino_shape s,std::span<const float> input,
                       const named_weights &parameters,bool capture_all) {
    std::map<std::string,std::span<const float>> views;
    for(auto &[name,values]:parameters)views.emplace(name,values.values());
    return dino_block_impl(session,s,input,views,capture_all);
}
named_floats dino_block_bf16(neural_session &session,dino_shape s,std::span<const float> input,const named_weights &parameters,bool capture_all){
    named_floats rounded;
    for(auto &[name,v]:parameters)rounded[name]=bf16_values(v.values());
    std::map<std::string,std::span<const float>> views;
    for(auto &[name,v]:rounded)views[name]=v;
    auto x=bf16_values(input);return dino_block_impl(session,s,x,views,capture_all,true);
}
}
