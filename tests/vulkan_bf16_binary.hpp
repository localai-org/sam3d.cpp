#pragma once
#include "neural.hpp"
#include "ggml.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

// No models: exercise fusion and refusal paths against both separate device
// operations and an independent scalar F32-then-BF16 oracle.
inline void test_bf16_binary(sam3d::neural_session &session) {
    const auto old=std::getenv("GGML_VK_FUSE_BF16_BINARY");
    const bool had=old!=nullptr;const std::string saved=had?old:"";
    auto flag=[](const char *v){
#if defined(_WIN32)
        _putenv_s("GGML_VK_FUSE_BF16_BINARY",v?v:"");
#else
        if(v)setenv("GGML_VK_FUSE_BF16_BINARY",v,1);else unsetenv("GGML_VK_FUSE_BF16_BINARY");
#endif
    };
    for(bool mul:{false,true})for(int n:{1,7,513,1280})for(int broadcast=0;broadcast<3;++broadcast)for(int mode=0;mode<8;++mode){
        const int rows=6,stride=n+(mode==5?3:0),bn=broadcast==2?1:n,br=broadcast==0?rows:1;
        std::vector<float> av(rows*stride,0),bv(bn*br),expected(rows*n),raw(rows*n),unfused;
        for(int r=0;r<rows;++r)for(int i=0;i<n;++i){
            const uint32_t bits=0x3f800000u+uint32_t((r*n+i)*9973%0x1800000);
            av[r*stride+i]=std::bit_cast<float>(bits|((r+i)%2?0x80000000u:0));
        }
        constexpr std::array<float,8> operands{1.f,-1.f,.5f,1.00390625f,-2.f,0.f,-0.f,.000030517578125f};
        for(size_t i=0;i<bv.size();++i)bv[i]=operands[i%operands.size()];
        for(int r=0;r<rows;++r)for(int i=0;i<n;++i){
            const float a=av[r*stride+i],b=bv[(r%br)*bn+i%bn];
            volatile float operation=mul?a*b:a+b;raw[r*n+i]=operation;
            expected[r*n+i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(operation));
        }
        for(bool fused:{false,true}){
            flag(fused?"1":nullptr);
            std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true}),ggml_free);
            if(!c)throw std::bad_alloc();auto ctx=c.get();
            auto storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,stride,rows);
            auto a=mode==5?ggml_view_2d(ctx,storage,n,rows,stride*4,0):storage;
            auto b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,bn,br);
            auto op=mul?(mode==6?ggml_mul_inplace(ctx,a,b):ggml_mul(ctx,a,b)):
                        (mode==6?ggml_add_inplace(ctx,a,b):ggml_add(ctx,a,b));
            if(mode==1)ggml_set_output(op);
            auto cast_input=mode==7?ggml_reshape_3d(ctx,op,n,3,2):op;
            auto narrow=ggml_cast(ctx,cast_input,GGML_TYPE_BF16);
            if(mode==3)ggml_set_output(narrow);
            auto out=ggml_cast(ctx,narrow,GGML_TYPE_F32);ggml_set_output(out);
            auto extra=mode==2?ggml_scale(ctx,op,2.f):mode==4?ggml_cast(ctx,narrow,GGML_TYPE_F32):nullptr;
            if(extra)ggml_set_output(extra);
            auto graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
            if(extra)ggml_build_forward_expand(graph,extra);
            auto buffer=session.allocate(ctx);if(!buffer)throw std::bad_alloc();
            ggml_backend_tensor_set(storage,av.data(),0,av.size()*4);ggml_backend_tensor_set(b,bv.data(),0,bv.size()*4);
            if(std::getenv("GGML_VK_PERF_LOGGER"))
                std::cerr<<"BF16_BINARY_CASE op="<<(mul?"MUL":"ADD")<<" width="<<n<<" broadcast="<<broadcast
                         <<" mode="<<mode<<" fused="<<fused<<" expected="<<(fused&&mode==0)<<std::endl;
            if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("binary BF16 graph failed");
            if(std::getenv("GGML_VK_PERF_LOGGER"))std::cerr<<"BF16_BINARY_CASE_END"<<std::endl;
            auto check=[&](ggml_tensor *t,const std::vector<float> &want){
                std::vector<float> got(want.size());ggml_backend_tensor_get(t,got.data(),0,got.size()*4);
                for(size_t i=0;i<got.size();++i)if(std::bit_cast<uint32_t>(got[i])!=std::bit_cast<uint32_t>(want[i]))
                    throw std::runtime_error("binary BF16 bit mismatch: op="+std::to_string(mul)+" width="+std::to_string(n)+" broadcast="+std::to_string(broadcast)+" mode="+std::to_string(mode)+" fused="+std::to_string(fused)+" index="+std::to_string(i));
                return got;
            };
            auto got=check(out,expected);
            if(!fused)unfused=got;else check(out,unfused);
            if(mode==1)check(op,raw);
            // GGML SCALE includes its zero bias: fma(-0, 2, +0) is +0.
            if(mode==2){auto twice=raw;for(auto &x:twice)x=std::fma(x,2.f,0.f);check(extra,twice);}
            if(mode==4)check(extra,expected);
            if(mode==3){
                std::vector<ggml_bf16_t> observed(expected.size());ggml_backend_tensor_get(narrow,observed.data(),0,observed.size()*2);
                for(size_t i=0;i<observed.size();++i)if(std::bit_cast<uint32_t>(ggml_bf16_to_fp32(observed[i]))!=std::bit_cast<uint32_t>(expected[i]))
                    throw std::runtime_error("binary BF16 observed narrow intermediate lost");
            }
        }
    }
    flag(had?saved.c_str():nullptr);
    std::cout<<"192 binary BF16 cases passed both fusion settings: exact scalar/device bits, broadcasts, tails, observed/shared values, strides, aliases and reshape guards\n";
}
