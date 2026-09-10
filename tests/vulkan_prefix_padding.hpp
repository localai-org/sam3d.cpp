#pragma once
#include "neural.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

// Model-free F32 P x V. Probabilities are normalized before padding: extending
// the softmax domain would change this oracle. Include the actual 1029-key,
// 20-head, five-register-query shape, not just tiny blocks.
inline void test_prefix_padding(sam3d::neural_session &session){
    double maximum=0;
    for(int keys:{1,5,17,1029})for(int heads:{1,20})for(int queries:{1,5}){
        constexpr int dim=64;const int tail=(4-keys%4)%4;
        std::vector<float> av(keys*dim*heads),bv(keys*queries*heads);
        std::vector<double> expected(dim*queries*heads,0);
        for(int h=0;h<heads;++h){
            for(int j=0;j<dim;++j)for(int k=0;k<keys;++k)
                av[(h*dim+j)*keys+k]=float(((k*17+j*11+h*3)%79)-23)/32.f;
            for(int q=0;q<queries;++q){
                double sum=0;
                for(int k=0;k<keys;++k)sum+=1+(k*13+q*7+h)%31;
                for(int k=0;k<keys;++k)bv[(h*queries+q)*keys+k]=float((1+(k*13+q*7+h)%31)/sum);
                for(int j=0;j<dim;++j)for(int k=0;k<keys;++k)
                    expected[(h*queries+q)*dim+j]+=double(av[(h*dim+j)*keys+k])*bv[(h*queries+q)*keys+k];
            }
        }
        for(bool padded:{false,true})for(int layout=0;layout<3;++layout){
            std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
            if(!c)throw std::bad_alloc();auto ctx=c.get();
            auto a=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,keys,dim,heads);
            auto b=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,keys,queries,heads);
            auto left=padded?ggml_pad(ctx,a,tail,0,0,0):a;
            auto right=padded?ggml_pad(ctx,b,tail,0,0,0):b;
            if(layout==1){
                left=ggml_reshape_4d(ctx,left,left->ne[0],dim,1,heads);
                right=ggml_reshape_4d(ctx,right,right->ne[0],1,queries,heads);
            }
            auto out=layout==2?ggml_mul_mat(ctx,right,left):ggml_mul_mat(ctx,left,right);
            ggml_mul_mat_set_prec(out,GGML_PREC_F32);
            if(layout==2)out=ggml_cont(ctx,ggml_transpose(ctx,out));
            auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,out);
            auto buffer=session.allocate(ctx);if(!buffer)throw std::bad_alloc();
            ggml_backend_tensor_set(a,av.data(),0,av.size()*4);ggml_backend_tensor_set(b,bv.data(),0,bv.size()*4);
            if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("prefix padding graph failed");
            std::vector<float> got(expected.size());ggml_backend_tensor_get(out,got.data(),0,got.size()*4);
            for(size_t i=0;i<got.size();++i){
                const auto error=std::abs(double(got[i])-expected[i]);maximum=std::max(maximum,error);
                if(!std::isfinite(got[i]) || error>1e-5)
                    throw std::runtime_error("prefix padding F32 oracle mismatch: keys="+std::to_string(keys)+" padded="+std::to_string(padded)+" layout="+std::to_string(layout));
            }
        }
    }
    std::cout<<"96 F32 prefix P x V cases passed against double oracle: padding, vector batching, transposed GEMM; max abs "<<maximum<<'\n';
}
