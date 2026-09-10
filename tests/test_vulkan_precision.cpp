#include "neural.hpp"
#include "dino_block.hpp"
#include "ggml.h"
#include <cstdlib>
#include <cmath>
#include <bit>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "vulkan_bf16_binary.hpp"
#include "vulkan_bf16_linear.hpp"
#include "vulkan_prefix_padding.hpp"
#include "vulkan_bf16_silu.hpp"
#include "vulkan_bf16_affine.hpp"
#include "vulkan_bf16_norm_affine.hpp"

// Opt-in real-device test: no models. Large and sub-F16-resolution operands
// expose an implicit half conversion even when accumulation remains F32.
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected Vulkan module and exact device description");
    sam3d::neural_session session(argv[1],"Vulkan",0,1,argv[2],true);
    for(int n:{1,13,37})for(bool strided:{false,true})for(bool large:{false,true}){
        const int k=64,m=33,batch=2,stride=k+(strided?3:0);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
        if(!c)throw std::bad_alloc();
        auto storage=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,stride,m);
        auto a=ggml_view_2d(c.get(),storage,k,m,stride*4,0);
        auto bstorage=ggml_new_tensor_3d(c.get(),GGML_TYPE_F32,stride,n,batch);
        auto b=ggml_view_3d(c.get(),bstorage,k,n,batch,stride*4,stride*n*4,0);
        auto out=ggml_mul_mat(c.get(),a,b);ggml_mul_mat_set_prec(out,GGML_PREC_F32);
        auto graph=ggml_new_graph_custom(c.get(),32,false);ggml_build_forward_expand(graph,out);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)if(!ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)))throw std::runtime_error("precision test operation unsupported");
        auto buffer=session.allocate(c.get());if(!buffer)throw std::bad_alloc();
        std::vector<float> av(m*stride),bv(stride*n*batch),expected(m*n*batch),got(expected.size());
        for(int row=0;row<m;++row)for(int col=0;col<k;++col)
            av[row*stride+col]=large?70000.f+float((row+3*col)%13):2.f+float((row+3*col)%13)/2048.f;
        for(size_t i=0;i<bv.size();++i)bv[i]=float(int(i%7)-3)/64.f;
        for(int sample=0;sample<batch;++sample)for(int col=0;col<n;++col)for(int row=0;row<m;++row){
            double sum=0;for(int j=0;j<k;++j)sum+=double(av[row*stride+j])*bv[(sample*n+col)*stride+j];
            expected[(sample*n+col)*m+row]=float(sum);
        }
        ggml_backend_tensor_set(storage,av.data(),0,av.size()*4);ggml_backend_tensor_set(bstorage,bv.data(),0,bv.size()*4);
        if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("precision graph failed");
        ggml_backend_tensor_get(out,got.data(),0,got.size()*4);
        for(size_t i=0;i<got.size();++i)if(!std::isfinite(got[i]) || std::abs(got[i]-expected[i])>(large?.002f:1e-6f))
            throw std::runtime_error("F32 operands altered: n="+std::to_string(n)+" strided="+std::to_string(strided)+" large="+std::to_string(large)+" expected="+std::to_string(expected[i])+" got="+std::to_string(got[i]));
    }
    std::cout<<"12 F32 precision cases passed: vector/matrix, broadcast, strided, range and sub-F16 resolution\n";
    // Exact BF16-roundtrip fusion guard. Include padded source layouts,
    // observed/shared intermediate casts (must not be elided), tail sizes,
    // signed zero, subnormals, ties and overflow to infinity.
    for(int n:{1,7,511,512,513,8193})for(int mode=0;mode<4;++mode){
        const int rows=3,stride=n+(mode==1?5:0);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
        if(!c)throw std::bad_alloc();
        auto storage=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,stride,rows);
        auto source=ggml_view_2d(c.get(),storage,n,rows,stride*4,0);
        auto narrow=ggml_cast(c.get(),source,GGML_TYPE_BF16);
        if(mode==2)ggml_set_output(narrow);
        auto wide=ggml_cast(c.get(),narrow,GGML_TYPE_F32);ggml_set_output(wide);
        auto extra=mode==3?ggml_cast(c.get(),narrow,GGML_TYPE_F32):nullptr;
        if(extra)ggml_set_output(extra);
        auto graph=ggml_new_graph_custom(c.get(),32,false);ggml_build_forward_expand(graph,wide);
        if(extra)ggml_build_forward_expand(graph,extra);
        auto buffer=session.allocate(c.get());if(!buffer)throw std::bad_alloc();
        std::vector<float> input(rows*stride),expected(rows*n),got(rows*n);
        constexpr std::array<uint32_t,12> special{0,0x80000000,1,0x80000001,0x7fff,0x8000,0x8001,0x3f808000,0x3f818000,0x7f7fffff,0xff7fffff,0x00800000};
        for(size_t i=0;i<input.size();++i){
            const uint32_t bits=i<special.size()?special[i]:uint32_t((i%255)<<23)|uint32_t((i*9973)&0x7fffff)|uint32_t((i%2)<<31);
            input[i]=std::bit_cast<float>(bits);
        }
        for(int row=0;row<rows;++row)for(int col=0;col<n;++col)
            expected[row*n+col]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(input[row*stride+col]));
        ggml_backend_tensor_set(storage,input.data(),0,input.size()*4);
        if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("BF16 roundtrip graph failed");
        auto check=[&](ggml_tensor *t){
            ggml_backend_tensor_get(t,got.data(),0,got.size()*4);
            for(size_t i=0;i<got.size();++i)if(std::bit_cast<uint32_t>(got[i])!=std::bit_cast<uint32_t>(expected[i]))
                throw std::runtime_error("BF16 roundtrip changed bits: mode="+std::to_string(mode)+" n="+std::to_string(n)+" index="+std::to_string(i));
        };
        check(wide);if(extra)check(extra);
        if(mode==2){
            std::vector<ggml_bf16_t> observed(rows*n);ggml_backend_tensor_get(narrow,observed.data(),0,observed.size()*sizeof(ggml_bf16_t));
            for(size_t i=0;i<observed.size();++i)if(std::bit_cast<uint32_t>(ggml_bf16_to_fp32(observed[i]))!=std::bit_cast<uint32_t>(expected[i]))
                throw std::runtime_error("observed BF16 intermediate was lost");
        }
    }
    std::cout<<"24 BF16 roundtrip cases passed: exact bits, tail sizes, padded layouts, shared/observed intermediates\n";
    test_bf16_binary(session);
    // Test a math-SDPA numerator and denominator on BF16 Q/K/V inputs.
    // Constant V exposes a rounded numerator with an F32 denominator; varied
    // V additionally exposes the older rounded-numerator/rounded-denominator
    // path that masks its quantization error with constant values.
    // Q*K and max are exact here; every key tile has the same row maximum.
    for(int keys:{17,63,129})for(float logit_shift:{-8.f,0.f,8.f})for(bool varied:{false,true}){
        constexpr int d=64,queries=7,heads=2;
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
        if(!c)throw std::bad_alloc();
        auto q=ggml_new_tensor_4d(c.get(),GGML_TYPE_F32,d,queries,heads,1);
        auto k=ggml_new_tensor_4d(c.get(),GGML_TYPE_BF16,d,keys,heads,1);
        auto v=ggml_new_tensor_4d(c.get(),GGML_TYPE_BF16,d,keys,heads,1);
        auto out=ggml_flash_attn_ext(c.get(),q,k,v,nullptr,.125f,0,0);
        ggml_flash_attn_ext_set_prec(out,GGML_PREC_F32);
        if(!ggml_backend_supports_op(session.backend(),out))throw std::runtime_error("BF16 attention precision operation unsupported");
        auto graph=ggml_new_graph_custom(c.get(),32,false);ggml_build_forward_expand(graph,out);
        auto buffer=session.allocate(c.get());if(!buffer)throw std::bad_alloc();
        std::vector<float> qv(d*queries*heads,0.f),got(d*queries*heads);
        std::vector<ggml_bf16_t> kv(d*keys*heads),vv(d*keys*heads);
        for(int h=0;h<heads;++h){
            for(int i=0;i<queries;++i)qv[(h*queries+i)*d]=8.f;
            for(int i=0;i<keys;++i)for(int j=0;j<d;++j){
                kv[(h*keys+i)*d+j]=ggml_fp32_to_bf16(j==0?logit_shift-float(i%8)/4.f:0.f);
                const float value=varied?float((3*i)%7-3)/4.f:1.f;
                vv[(h*keys+i)*d+j]=ggml_fp32_to_bf16(j%2?-value:value);
            }
        }
        double sum=0,numerator=0;
        for(int i=0;i<keys;++i){
            const double p=std::exp(-double(i%8)/4.);
            sum+=p;numerator+=p*(varied?double((3*i)%7-3)/4.:1.);
        }
        const float expected=float(numerator/sum);
        ggml_backend_tensor_set(q,qv.data(),0,qv.size()*4);
        ggml_backend_tensor_set(k,kv.data(),0,kv.size()*sizeof(ggml_bf16_t));
        ggml_backend_tensor_set(v,vv.data(),0,vv.size()*sizeof(ggml_bf16_t));
        if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("attention denominator graph failed");
        ggml_backend_tensor_get(out,got.data(),0,got.size()*4);
        for(size_t i=0;i<got.size();++i)if(!std::isfinite(got[i]) || std::abs(got[i]-(i%2?-expected:expected))>1e-5f)
            throw std::runtime_error("BF16 attention lost probability precision: keys="+std::to_string(keys)+" expected="+std::to_string(expected)+" got="+std::to_string(got[i]));
    }
    std::cout<<"18 BF16 attention probability-precision cases passed: multiple heads, constant/varied signed values, padded key tails and logit-shift invariance\n";
    test_bf16_linear(session);
    test_prefix_padding(session);
    // Environment switches affect only this synchronous test process. Verify
    // the production block path, including prefix slicing/concatenation and
    // that register queries still see image keys and values.
    auto mode=[](const char *key,const char *value){
#ifdef _WIN32
        if(_putenv_s(key,value))throw std::runtime_error("test environment update failed");
#else
        if(setenv(key,value,1))throw std::runtime_error("test environment update failed");
#endif
    };
    for(uint32_t batch:{1u,2u})for(uint32_t prefix:{1u,5u}){
        sam3d::dino_shape s{batch,2,2,128,2,prefix,256};
        sam3d::named_floats data;
        for(const auto &[name,count]:sam3d::dino_parameter_sizes(s)){
            auto &v=data[name];v.resize(count);
            for(size_t i=0;i<v.size();++i){
                if(name=="periods")v[i]=1.f+float(i);
                else if(name.starts_with("norm") && name.ends_with("weight"))v[i]=1.f;
                else if(name.ends_with("bias_mask"))v[i]=(i>=s.dim && i<2*s.dim)?0.f:1.f;
                else if(name.ends_with("bias"))v[i]=0.f;
                else if(name.starts_with("ls"))v[i]=.25f;
                else v[i]=float(int((i*17+name.size()*13)%101)-50)/512.f;
            }
        }
        sam3d::weight_map weights(std::move(data));
        const uint64_t n=4+prefix;
        std::vector<float> input(batch*n*s.dim);
        for(uint32_t b=0;b<batch;++b)for(uint64_t t=0;t<n;++t)for(uint32_t j=0;j<s.dim;++j)
            input[(b*n+t)*s.dim+j]=float(int(((t+3+b)*(j+11))%37)-18)/8.f;
        mode("SAM3D_BF16_FLASH_ATTENTION","0");mode("SAM3D_BF16_PRECISE_PREFIX","0");
        auto expected=sam3d::dino_block_bf16(session,s,input,weights,false).at("21.output");
        mode("SAM3D_BF16_FLASH_ATTENTION","1");mode("SAM3D_BF16_PRECISE_PREFIX","1");
        auto got=sam3d::dino_block_bf16(session,s,input,weights,false).at("21.output");
        if(got.size()!=input.size())throw std::runtime_error("precise-prefix output shape changed");
        for(uint32_t b=0;b<batch;++b)for(uint64_t t=0;t<prefix;++t)for(uint32_t j=0;j<s.dim;++j){
            const auto i=(b*n+t)*s.dim+j;
            if(got[i]!=expected[i])throw std::runtime_error("precise-prefix math output/layout changed");
        }
        for(uint32_t b=0;b<batch;++b)for(uint64_t t=prefix;t<n;++t)for(uint32_t j=0;j<s.dim;++j)
            input[(b*n+t)*s.dim+j]*=-1.f;
        auto changed=sam3d::dino_block_bf16(session,s,input,weights,false).at("21.output");
        bool responds=false;
        for(uint32_t b=0;b<batch;++b)for(uint64_t t=0;t<prefix;++t)for(uint32_t j=0;j<s.dim;++j){
            const auto i=(b*n+t)*s.dim+j;responds|=got[i]!=changed[i];
        }
        if(!responds)throw std::runtime_error("precise-prefix attention lost image interactions");
    }
    std::cout<<"4 precise-prefix block cases passed: exact math prefix, multiple heads/batches, original ordering and image interactions\n";
    test_bf16_silu_gate(session);
    test_bf16_affine(session);
    test_bf16_norm_affine(session);
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
