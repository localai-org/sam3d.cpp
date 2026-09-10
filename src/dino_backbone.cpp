// Copyright (c) Meta Platforms, Inc. and affiliates.
// DINOv3/Body contract adaptation: see THIRD_PARTY_NOTICES.md.
#include "dino_backbone.hpp"
#include "bf16.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sam3d {
namespace {
void require(bool b,const char *message) { if (!b) throw std::invalid_argument(message); }
void finite(std::span<const float> values) {
    require(std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}),
            "non-finite backbone tensor");
}
dino_shape block_shape(backbone_shape s) {
    return {s.batch,s.height/s.patch,s.width/s.patch,s.dim,s.heads,1+s.storage,s.hidden};
}
std::vector<float> final_norm(neural_session &session,uint32_t dim,
    std::span<const float> tokens,std::span<const float> weight,std::span<const float> bias,bool bf16=false) {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({
        ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
    if (!context) throw std::bad_alloc();
    auto c=context.get();
    auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,dim,tokens.size()/dim);
    auto w=ggml_new_tensor_1d(c,GGML_TYPE_F32,dim), b=ggml_new_tensor_1d(c,GGML_TYPE_F32,dim);
    auto output=ggml_add(c,ggml_mul(c,ggml_norm(c,x,1e-5f),w),b);
    if(bf16)output=bf16_round(c,output);
    auto graph=ggml_new_graph_custom(c,32,false); ggml_build_forward_expand(graph,output);
    for (int i=0;i<ggml_graph_n_nodes(graph);++i)
        require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),
                "backend does not support final norm operation");
    auto buffer=session.allocate(c);
    if (!buffer) throw std::bad_alloc();
    ggml_backend_tensor_set(x,tokens.data(),0,ggml_nbytes(x));
    ggml_backend_tensor_set(w,weight.data(),0,ggml_nbytes(w));
    ggml_backend_tensor_set(b,bias.data(),0,ggml_nbytes(b));
    if (ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("backbone final norm failed");
    std::vector<float> result(tokens.size());
    ggml_backend_tensor_get(output,result.data(),0,result.size()*4); finite(result);
    return result;
}
}

void validate_backbone_shape(backbone_shape s) {
    require(s.patch>=1 && s.patch<=32 && s.height>=s.patch && s.width>=s.patch,
            "invalid backbone image/patch shape");
    require(s.height<=512 && s.width<=512 && s.height%s.patch==0 && s.width%s.patch==0,
            "backbone image must be divisible by patch size and at most 512x512");
    require(s.depth>=1 && s.depth<=32 && s.storage<=7,"invalid backbone depth/storage tokens");
    validate_dino_shape(block_shape(s));
}
std::vector<std::pair<std::string,uint64_t>> backbone_parameter_sizes(backbone_shape s) {
    validate_backbone_shape(s);
    const uint64_t d=s.dim;
    std::vector<std::pair<std::string,uint64_t>> sizes={
        {"patch_embed.proj.weight",d*3*s.patch*s.patch},{"patch_embed.proj.bias",d},
        {"cls_token",d},{"mask_token",d}};
    if (s.storage) sizes.emplace_back("storage_tokens",s.storage*d);
    sizes.emplace_back("rope_embed.periods",d/s.heads/4);
    for (uint32_t i=0;i<s.depth;++i) for (auto &[name,size]:dino_parameter_sizes(block_shape(s)))
        if (name!="periods") sizes.emplace_back("blocks."+std::to_string(i)+"."+name,size);
    sizes.emplace_back("norm.weight",d); sizes.emplace_back("norm.bias",d);
    return sizes;
}
std::vector<float> dino_backbone(neural_session &session,backbone_shape s,
    std::span<const float> image,const parameter_reader &reader,const backbone_observer &observer,
    const backbone_block_observer &block_observer,dino_resident_stack *resident,bool bf16) {
    validate_backbone_shape(s); require(bool(reader),"missing backbone parameter reader");
    require(image.size()==uint64_t(s.batch)*3*s.height*s.width,"backbone image size mismatch"); finite(image);
    if(resident){
        require(!block_observer,"resident backbone does not capture intra-block operations");
        return resident->run_image(s,image,observer);
    }
    auto load=[&](const std::string &name,uint64_t count) {
        auto values=reader(name,count);
        require(values.values().size()==count,"backbone parameter size mismatch");
        return bf16?validated_weights::checked(bf16_values(values.values())):values;
    };
    auto observe=[&](const std::string &name,std::span<const float> data) {
        if (observer) observer(name,data);
    };
    const uint64_t d=s.dim, n=(s.height/s.patch)*(s.width/s.patch), prefix=1+s.storage;
    std::vector<float> tokens(uint64_t(s.batch)*(n+prefix)*d);
    {
        auto weight=load("patch_embed.proj.weight",d*3*s.patch*s.patch);
        auto bias=load("patch_embed.proj.bias",d);
        auto patch=patch_embed(session,{s.batch,3,s.height,s.width,s.dim,s.patch},image,weight.values(),bias.values(),false,bf16);
        observe("00.patch",patch.tokens);
        auto cls_weight=load("cls_token",d), mask=load("mask_token",d);
        std::vector<float> cls(cls_weight.values().begin(),cls_weight.values().end());
        std::vector<float> storage;
        if (s.storage) {auto stored=load("storage_tokens",s.storage*d);storage.assign(stored.values().begin(),stored.values().end());}
        // Upstream no-mask path adds 0*mask_token to CLS before concatenation.
        for (uint64_t j=0;j<d;++j) cls[j]+=0.f*mask.values()[j];
        for (uint32_t batch=0;batch<s.batch;++batch) {
            auto start=tokens.begin()+batch*(n+prefix)*d;
            std::copy(cls.begin(),cls.end(),start);
            std::copy(storage.begin(),storage.end(),start+d);
            std::copy_n(patch.tokens.begin()+batch*n*d,n*d,start+prefix*d);
        }
    }
    observe("01.tokens",tokens);
    const auto periods=load("rope_embed.periods",d/s.heads/4);
    for (uint32_t i=0;i<s.depth;++i) {
        named_weights parameters; parameters.emplace("periods",periods);
        const auto block="blocks."+std::to_string(i)+".";
        for (auto &[name,count]:dino_parameter_sizes(block_shape(s)))
            if (name!="periods") parameters.emplace(name,load(block+name,count));
        auto output=bf16?dino_block_bf16(session,block_shape(s),tokens,parameters,bool(block_observer)):dino_block(session,block_shape(s),tokens,parameters,bool(block_observer));
        if (block_observer) for (const auto &[name,values]:output) block_observer(i,name,values);
        tokens=std::move(output.at("21.output"));
        const auto key="02.block."+std::string(i<10?"0":"")+std::to_string(i);
        observe(key,tokens);
    }
    auto weight=load("norm.weight",d), bias=load("norm.bias",d);
    auto normalized=final_norm(session,s.dim,tokens,weight.values(),bias.values(),bf16);
    observe("03.norm",normalized);
    std::vector<float> features(uint64_t(s.batch)*n*d);
    for (uint32_t batch=0;batch<s.batch;++batch) for (uint64_t channel=0;channel<d;++channel)
        for (uint64_t token=0;token<n;++token)
            features[(batch*d+channel)*n+token]=normalized[(batch*(n+prefix)+prefix+token)*d+channel];
    observe("04.features",features);
    return features;
}

struct dino_resident_stack::impl {
    using context_ptr=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using buffer_ptr=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    struct weights {
        context_ptr context{nullptr,ggml_free};
        buffer_ptr buffer{nullptr,ggml_backend_buffer_free};
    };
    neural_session &session;
    backbone_shape shape;
    bool bf16=false;
    named_weights fixed;
    std::vector<weights> layers;
    context_ptr context{nullptr,ggml_free};
    std::unique_ptr<ggml_gallocr,decltype(&ggml_gallocr_free)> allocator{nullptr,ggml_gallocr_free};
    ggml_cgraph *graph=nullptr;
    ggml_tensor *input=nullptr;
    ggml_tensor *normalized=nullptr,*features=nullptr;
    std::vector<ggml_tensor *> outputs;
    // This context owns the stem tensors; its output copy writes the existing
    // stack input. Keep the stem allocation independent of stack lifetime reuse.
    context_ptr image_context{nullptr,ggml_free};
    buffer_ptr image_buffer{nullptr,ggml_backend_buffer_free};
    ggml_cgraph *image_graph=nullptr;
    ggml_tensor *image_input=nullptr,*patch_tokens=nullptr,*image_tokens=nullptr;
    impl(neural_session &session_,backbone_shape s,const parameter_reader &reader,bool low_precision):session(session_),shape(s),bf16(low_precision){
        validate_backbone_shape(s);require(bool(reader),"missing resident backbone parameter reader");
        auto bs=block_shape(s);const int64_t n=bs.height*bs.width+bs.prefix;
        const size_t capacity=512*s.depth+64;
        context.reset(ggml_init({ggml_tensor_overhead()*capacity+ggml_graph_overhead_custom(capacity,false),nullptr,true}));
        if(!context)throw std::bad_alloc();auto c=context.get();
        input=ggml_new_tensor_3d(c,GGML_TYPE_F32,s.dim,n,s.batch);ggml_set_input(input);
        // Keep the small stem/norm parameters immutable too: never mix cached
        // transformer weights with freshly reread (possibly modified) files.
        for(auto &[name,count]:backbone_parameter_sizes(s))if(!name.starts_with("blocks.")){
            auto value=reader(name,count);require(value.values().size()==count,"resident stem parameter size mismatch");
            fixed.emplace(name,bf16?validated_weights::checked(bf16_values(value.values())):std::move(value));
        }
        auto periods=fixed.at("rope_embed.periods");
        auto angle_values=dino_angles(bs,periods.values());
        auto angles=ggml_new_tensor_2d(c,GGML_TYPE_F32,s.dim/s.heads,n);
        // An input flag alone allows lifetime reuse after its last consumer.
        // Preserve these immutable angles across repeated graph evaluations.
        ggml_set_input(angles);ggml_set_output(angles);
        auto x=input;uint64_t weight_bytes=0;
        for(uint32_t i=0;i<s.depth;++i){
            weights layer;layer.context.reset(ggml_init({ggml_tensor_overhead()*32,nullptr,true}));
            if(!layer.context)throw std::bad_alloc();
            named_tensors p;
            for(auto &[name,count]:dino_parameter_sizes(bs))if(name!="periods"){
                weight_bytes+=count*4;
                require(weight_bytes<=uint64_t(4)*1024*1024*1024,"resident backbone weights exceed 4 GiB budget");
                p[name]=ggml_new_tensor_1d(layer.context.get(),bf16 && name.ends_with(".weight") && !name.starts_with("norm")?GGML_TYPE_BF16:GGML_TYPE_F32,count);
            }
            layer.buffer.reset(ggml_backend_alloc_ctx_tensors(layer.context.get(),session.backend()));
            if(!layer.buffer)throw std::bad_alloc();
            ggml_backend_buffer_set_usage(layer.buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for(auto &[name,t]:p){
                auto values=reader("blocks."+std::to_string(i)+"."+name,ggml_nelements(t));
                require(values.values().size()==uint64_t(ggml_nelements(t)),"resident backbone parameter size mismatch");
                if(name=="attn.qkv.bias_mask"){
                    const auto mask=values.values();
                    const bool zero=std::all_of(mask.begin(),mask.end(),[](float v){return v==0.f;});
                    for(uint64_t j=0;j<mask.size();++j)
                        require(zero || mask[j]==(j>=s.dim && j<2*s.dim?0.f:1.f),"invalid DINO key bias mask");
                }
                if(t->type==GGML_TYPE_BF16)set_bf16_tensor(t,values.values());
                else if(bf16){auto rounded=bf16_values(values.values());ggml_backend_tensor_set(t,rounded.data(),0,ggml_nbytes(t));}
                else ggml_backend_tensor_set(t,values.values().data(),0,ggml_nbytes(t));
            }
            layers.push_back(std::move(layer));
            x=dino_block_graph(session,c,bs,x,p,angles,false,bf16).at("21.output");
            outputs.push_back(x);
        }
        // Preserve final normalization and the prefix-removing NCHW layout in
        // this resident graph. Previously these boundaries made a full host
        // download/upload and a strided CPU transpose on every request.
        auto norm_weight=session.parameter(c,fixed.at("norm.weight"),s.dim,1);
        auto norm_bias=session.parameter(c,fixed.at("norm.bias"),s.dim,1);
        normalized=ggml_add(c,ggml_mul(c,ggml_norm(c,x,1e-5f),norm_weight),norm_bias);
        if(bf16)normalized=bf16_round(c,normalized);
        ggml_set_output(normalized);
        auto patches=ggml_view_3d(c,normalized,s.dim,n-bs.prefix,s.batch,
            s.dim*sizeof(float),s.dim*n*sizeof(float),bs.prefix*s.dim*sizeof(float));
        features=ggml_cont(c,ggml_permute(c,patches,1,0,2,3));ggml_set_output(features);
        graph=ggml_new_graph_custom(c,capacity,false);ggml_build_forward_expand(graph,features);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)
            require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported resident DINO operation");
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(session.backend())));
        if(!allocator)throw std::bad_alloc();
        size_t bytes=0;ggml_gallocr_reserve_n_size(allocator.get(),graph,nullptr,nullptr,&bytes);
        require(bytes<=uint64_t(1024)*1024*1024,"resident backbone activations exceed 1 GiB budget");
        // The measurement API records a matching layout without allocating it;
        // explicitly reserve before alloc_graph (which sees that layout match).
        if(!ggml_gallocr_reserve(allocator.get(),graph) || !ggml_gallocr_alloc_graph(allocator.get(),graph))throw std::bad_alloc();
        ggml_backend_tensor_set(angles,angle_values.data(),0,ggml_nbytes(angles));
        build_image_graph(capacity,bytes);
    }
    void build_image_graph(size_t stack_capacity,size_t stack_bytes){
        const auto s=shape;
        const int64_t d=s.dim,prefix=1+s.storage,grid=(s.height/s.patch)*(s.width/s.patch);
        const int64_t k=3*s.patch*s.patch;
        image_context.reset(ggml_init({ggml_tensor_overhead()*64+
            ggml_graph_overhead_custom(stack_capacity+64,false),nullptr,true}));
        if(!image_context)throw std::bad_alloc();auto c=image_context.get();
        image_input=ggml_new_tensor_4d(c,GGML_TYPE_F32,s.width,s.height,3,s.batch);
        ggml_set_input(image_input);
        auto w=ggml_new_tensor_4d(c,bf16?GGML_TYPE_BF16:GGML_TYPE_F32,s.patch,s.patch,3,d);
        auto bias=ggml_new_tensor_1d(c,GGML_TYPE_F32,d);
        auto prefix_values=ggml_new_tensor_3d(c,GGML_TYPE_F32,d,prefix,s.batch);
        auto pixels=bf16?bf16_round(c,image_input):image_input;
        auto patches=ggml_im2col(c,w,pixels,s.patch,s.patch,0,0,1,1,true,GGML_TYPE_F32);
        auto projection=ggml_mul_mat(c,ggml_reshape_2d(c,w,k,d),ggml_reshape_2d(c,patches,k,grid*s.batch));
        ggml_mul_mat_set_prec(projection,GGML_PREC_F32);
        // Conv2d rounds before its separate bias add, unlike Linear. Retain
        // both boundaries exactly as patch_embed and original BF16 execution.
        if(bf16)projection=bf16_round(c,projection);
        patch_tokens=ggml_add(c,projection,bias);
        if(bf16)patch_tokens=bf16_round(c,patch_tokens);
        ggml_set_output(patch_tokens);
        image_tokens=ggml_concat(c,prefix_values,ggml_reshape_3d(c,patch_tokens,d,grid,s.batch),1);
        ggml_set_output(image_tokens);
        auto copied=ggml_cpy(c,image_tokens,input);ggml_set_output(copied);
        // Build in execution order: the explicit copy populates the existing
        // stack input before any block consumes it. Backend overlap tracking
        // supplies the device dependency; no host token buffer is involved.
        image_graph=ggml_new_graph_custom(c,stack_capacity+64,false);
        ggml_build_forward_expand(image_graph,copied);
        ggml_build_forward_expand(image_graph,features);
        for(int i=0;i<ggml_graph_n_nodes(image_graph);++i)
            require(ggml_backend_supports_op(session.backend(),ggml_graph_node(image_graph,i)),"unsupported resident image operation");
        image_buffer.reset(ggml_backend_alloc_ctx_tensors(c,session.backend()));
        if(!image_buffer)throw std::bad_alloc();
        require(stack_bytes+ggml_backend_buffer_get_size(image_buffer.get())<=uint64_t(1024)*1024*1024,
                "resident image activations exceed 1 GiB budget");
        const auto &weight=fixed.at("patch_embed.proj.weight");
        if(bf16)set_bf16_tensor(w,weight.values());
        else ggml_backend_tensor_set(w,weight.values().data(),0,ggml_nbytes(w));
        ggml_backend_tensor_set(bias,fixed.at("patch_embed.proj.bias").values().data(),0,ggml_nbytes(bias));
        std::vector<float> pref(s.batch*prefix*d);
        const auto cls=fixed.at("cls_token").values(),mask=fixed.at("mask_token").values();
        for(uint32_t b=0;b<s.batch;++b){
            auto start=pref.begin()+b*prefix*d;
            for(int64_t j=0;j<d;++j)start[j]=cls[j]+0.f*mask[j];
            if(s.storage){const auto stored=fixed.at("storage_tokens").values();std::copy(stored.begin(),stored.end(),start+d);}
        }
        ggml_backend_tensor_set(prefix_values,pref.data(),0,ggml_nbytes(prefix_values));
    }
};
dino_resident_stack::dino_resident_stack(neural_session &session,backbone_shape s,const parameter_reader &reader,bool bf16)
    :impl_(std::make_unique<impl>(session,s,reader,bf16)){}
dino_resident_stack::~dino_resident_stack()=default;
bool dino_resident_stack::bf16()const{return impl_->bf16;}
validated_weights dino_resident_stack::parameter(const std::string &name,uint64_t count)const{
    auto it=impl_->fixed.find(name);
    require(it!=impl_->fixed.end() && it->second.values().size()==count,"resident fixed parameter mismatch");
    return it->second;
}
std::vector<float> dino_resident_stack::run(backbone_shape s,std::span<const float> tokens,const backbone_observer &observer){
    return execute(s,tokens,observer,false);
}
std::vector<float> dino_resident_stack::run_features(backbone_shape s,std::span<const float> tokens,const backbone_observer &observer){
    return execute(s,tokens,observer,true);
}
std::vector<float> dino_resident_stack::execute(backbone_shape s,std::span<const float> tokens,const backbone_observer &observer,bool features){
    const auto a=impl_->shape;
    require(s.batch==a.batch && s.height==a.height && s.width==a.width && s.patch==a.patch && s.dim==a.dim &&
        s.heads==a.heads && s.hidden==a.hidden && s.depth==a.depth && s.storage==a.storage,"resident backbone shape mismatch");
    require(tokens.size()==uint64_t(ggml_nelements(impl_->input)),"resident backbone input size mismatch");finite(tokens);
    if(impl_->bf16){
        auto rounded=bf16_values(tokens);ggml_backend_tensor_set(impl_->input,rounded.data(),0,tokens.size_bytes());
    }else ggml_backend_tensor_set(impl_->input,tokens.data(),0,tokens.size_bytes());
    if(ggml_backend_graph_compute(impl_->session.backend(),impl_->graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("resident DINO graph failed");
    return collect(s,observer,features);
}
std::vector<float> dino_resident_stack::run_image(backbone_shape s,std::span<const float> image,const backbone_observer &observer){
    const auto a=impl_->shape;
    require(s.batch==a.batch && s.height==a.height && s.width==a.width && s.patch==a.patch && s.dim==a.dim &&
        s.heads==a.heads && s.hidden==a.hidden && s.depth==a.depth && s.storage==a.storage,"resident backbone shape mismatch");
    require(image.size()==uint64_t(ggml_nelements(impl_->image_input)),"resident image size mismatch");finite(image);
    ggml_backend_tensor_set(impl_->image_input,image.data(),0,image.size_bytes());
    if(ggml_backend_graph_compute(impl_->session.backend(),impl_->image_graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("resident image encoder graph failed");
    if(observer)for(auto [name,t]:{std::pair{"00.patch",impl_->patch_tokens},std::pair{"01.tokens",impl_->image_tokens}}){
        std::vector<float> values(ggml_nelements(t));ggml_backend_tensor_get(t,values.data(),0,values.size()*4);finite(values);
        observer(name,values);
    }
    return collect(s,observer,true);
}
std::vector<float> dino_resident_stack::collect(backbone_shape s,const backbone_observer &observer,bool features){
    std::vector<float> values;
    if(observer || !features)values.resize(ggml_nelements(impl_->input));
    for(uint32_t i=0;i<s.depth;++i)if(observer || (!features && i+1==s.depth)){
        ggml_backend_tensor_get(impl_->outputs[i],values.data(),0,values.size()*4);finite(values);
        if(observer)observer("02.block."+std::string(i<10?"0":"")+std::to_string(i),values);
    }
    if(!features)return values;
    if(observer){
        ggml_backend_tensor_get(impl_->normalized,values.data(),0,values.size()*4);finite(values);
        observer("03.norm",values);
    }
    values.resize(uint64_t(ggml_nelements(impl_->features)));
    ggml_backend_tensor_get(impl_->features,values.data(),0,values.size()*4);finite(values);
    if(observer)observer("04.features",values);
    return values;
}
std::vector<float> body_backbone(neural_session &session,tensor_archive &archive,
    std::span<const float> image,const backbone_observer &observer,const backbone_block_observer &block_observer) {
    return dino_backbone(session,{1,512,512,16,1280,20,5120,32,4},image,
        [&](const std::string &name,uint64_t count){return archive.load(name,count*4);},observer,block_observer);
}
}
