#pragma once
#include "neural.hpp"
#include "ggml.h"
#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

// Exercise dense 4D indexing, multidimensional-broadcast fallback, descriptor
// offsets, tails, and the 262144-element dispatch boundary. Compare BOTH shader
// paths with an independent F32-operation + BF16-round scalar oracle.
inline void test_bf16_linear(sam3d::neural_session &session) {
    const char *key="GGML_VK_BF16_BINARY_LINEAR";
    const auto previous=std::getenv(key);
    const bool had=previous!=nullptr;
    const std::string saved=had?previous:"";
    auto flag=[&](const char *value){
#ifdef _WIN32
        _putenv_s(key,value?value:"");
#else
        if(value)setenv(key,value,1);else unsetenv(key);
#endif
    };
    if(!std::getenv("GGML_VK_FUSE_BF16_ROUND") || !std::getenv("GGML_VK_FUSE_BF16_BINARY"))
        throw std::runtime_error("linear precision tests require both BF16 fusion flags");
    for(bool mul:{false,true})for(int n:{1,7,513,1280})for(int broadcast=0;broadcast<4;++broadcast)for(int offset:{0,3}){
        const int rows=n==1280?129:3;
        const std::array<int64_t,4> dims{n,rows,4,2};
        auto bd=dims;
        if(broadcast==1)bd={n,1,1,1};
        if(broadcast==2)bd={1,1,1,1};
        if(broadcast==3)bd={n,1,2,1}; // valid but not a linear/row/scalar repeat
        const size_t count=size_t(n)*rows*4*2,bcount=bd[0]*bd[1]*bd[2]*bd[3];
        std::vector<float> av(count+offset+3,-12345.f),bv(bcount+offset+3,-54321.f),expected(count),baseline;
        for(size_t i=0;i<count;++i)av[offset+i]=std::bit_cast<float>((0x3f000000u+uint32_t((i*9973)%0x1800000))|((i%3)?0x80000000u:0));
        constexpr std::array<float,8> operands{1.f,-1.f,.5f,1.00390625f,-2.f,0.f,-0.f,.000030517578125f};
        for(size_t i=0;i<bcount;++i)bv[offset+i]=operands[i%operands.size()];
        for(size_t i=0;i<count;++i){
            const size_t x=i%n,y=(i/n)%rows,z=(i/(n*rows))%4,w=i/(n*rows*4);
            const size_t bi=(((w%bd[3])*bd[2]+z%bd[2])*bd[1]+y%bd[1])*bd[0]+x%bd[0];
            volatile float result=mul?av[offset+i]*bv[offset+bi]:av[offset+i]+bv[offset+bi];
            expected[i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(result));
        }
        for(bool linear:{false,true}){
            flag(linear?"1":nullptr);
            std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
            if(!c)throw std::bad_alloc();
            auto a_storage=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,av.size());
            auto b_storage=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,bv.size());
            auto view=[&](ggml_tensor *t,const std::array<int64_t,4> &d){return ggml_view_4d(c.get(),t,d[0],d[1],d[2],d[3],d[0]*4,d[0]*d[1]*4,d[0]*d[1]*d[2]*4,offset*4);};
            auto a=view(a_storage,dims),b=view(b_storage,bd);
            auto op=mul?ggml_mul(c.get(),a,b):ggml_add(c.get(),a,b);
            auto out=ggml_cast(c.get(),ggml_cast(c.get(),op,GGML_TYPE_BF16),GGML_TYPE_F32);
            ggml_set_output(out);
            auto graph=ggml_new_graph_custom(c.get(),32,false);ggml_build_forward_expand(graph,out);
            auto buffer=session.allocate(c.get());if(!buffer)throw std::bad_alloc();
            ggml_backend_tensor_set(a_storage,av.data(),0,av.size()*4);ggml_backend_tensor_set(b_storage,bv.data(),0,bv.size()*4);
            if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("linear BF16 graph failed");
            auto check=[&](ggml_tensor *t,const std::vector<float> &want){
                std::vector<float> got(want.size());ggml_backend_tensor_get(t,got.data(),0,got.size()*4);
                for(size_t i=0;i<got.size();++i)if(std::bit_cast<uint32_t>(got[i])!=std::bit_cast<uint32_t>(want[i]))
                    throw std::runtime_error("linear BF16 mismatch: mul="+std::to_string(mul)+" width="+std::to_string(n)+" broadcast="+std::to_string(broadcast)+" offset="+std::to_string(offset)+" linear="+std::to_string(linear)+" index="+std::to_string(i));
                return got;
            };
            auto got=check(out,expected);
            if(!linear)baseline=got;else check(out,baseline);
            check(a_storage,av);check(b_storage,bv); // including descriptor guard regions
        }
    }
    flag(had?saved.c_str():nullptr);
    std::cout<<"64 linear BF16 cases passed both indexing settings: exact scalar/device bits, 4D batches, repeated rows/scalars, general-broadcast fallback, descriptor offsets and large dispatch tails\n";
}
