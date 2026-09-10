#pragma once
#include "body_condition.hpp"
#include "body_decoder.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>

struct condition_case {
    sam3d::condition_shape shape;
    bool previous;
    uint32_t heads,head_dim,hidden;
    explicit condition_case(const std::array<uint32_t,16> &v):
        shape{v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],bool(v[10]),bool(v[11])},
        previous(v[12]),heads(v[13]),head_dim(v[14]),hidden(v[15]) {
        if(v[10]>1 || v[11]>1 || v[12]>1)throw std::invalid_argument("invalid conditioning flags");
        sam3d::validate_condition_shape(shape);sam3d::validate_decoder_shape(layer());
    }
    sam3d::decoder_shape layer() const {
        return {shape.batch,sam3d::condition_token_count(shape),(shape.height/shape.patch)*(shape.width/shape.patch),
            shape.token_dim,shape.context_dim,heads,head_dim,hidden,true,true,true};
    }
    std::vector<std::pair<std::string,uint64_t>> sizes() const {
        auto result=sam3d::condition_parameter_sizes(shape);
        for(auto &[name,size]:sam3d::decoder_parameter_sizes(layer()))result.emplace_back("decoder.layers.0."+name,size);
        std::sort(result.begin(),result.end());return result;
    }
    sam3d::named_floats run(sam3d::neural_session &session,const sam3d::named_floats &input,
                          const sam3d::named_floats &parameters,sam3d::scalar_division division,bool channels_last=false) const {
        sam3d::named_floats body,decoder;
        const std::string prefix="decoder.layers.0.";
        for(auto &[name,v]:parameters) {
            if(name.starts_with(prefix))decoder[name.substr(prefix.size())]=v;else body[name]=v;
        }
        const auto &s=shape;
        const auto prev=input.contains("previous")?std::span<const float>(input.at("previous")):std::span<const float>{};
        auto taps=sam3d::body_condition(session,s,input.at("features"),input.at("rays"),input.at("cliff"),input.at("keypoints"),prev,body,division,{},channels_last);
        const auto grid=(s.height/s.patch)*(s.width/s.patch);
        auto bnc=[&](const std::vector<float> &nchw,uint32_t batch) {
            std::vector<float> result(nchw.size());
            for(uint32_t b=0;b<batch;++b)for(uint32_t n=0;n<grid;++n)for(uint32_t c=0;c<s.context_dim;++c)
                result[(uint64_t(b)*grid+n)*s.context_dim+c]=nchw[(uint64_t(b)*s.context_dim+c)*grid+n];
            return result;
        };
        auto image=channels_last?taps.at("20.image"):bnc(taps.at("20.image"),s.batch);
        auto pe=channels_last?taps.at("21.image_pe"):bnc(taps.at("21.image_pe"),1);
        // Deliberately NO prompt mask: upstream sets token_mask=None, including
        // when point_mask contains zeros. Native intermediates only, no oracle.
        auto output=sam3d::body_decoder_layer(session,layer(),taps.at("30.tokens"),image,taps.at("31.token_pe"),pe,{},decoder,false);
        taps["40.layer_tokens"]=std::move(output.at("90.tokens"));taps["41.layer_context"]=std::move(output.at("91.context"));
        // Canonicalize only the diagnostic outputs after the real first layer
        // consumed the selected layout. A scalar inverse is independent of the
        // native transpose helper, catching batch/channel/token permutation bugs.
        if(channels_last)for(auto key:{"20.image","21.image_pe"}){
            const auto tokens=taps.at(key);auto &nchw=taps.at(key);
            const uint32_t batches=std::string_view(key)=="20.image"?s.batch:1;
            for(uint32_t b=0;b<batches;++b)for(uint32_t n=0;n<grid;++n)for(uint32_t d=0;d<s.context_dim;++d)
                nchw[(uint64_t(b)*s.context_dim+d)*grid+n]=tokens[(uint64_t(b)*grid+n)*s.context_dim+d];
        }
        return taps;
    }
};
