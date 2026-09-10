#include "neural.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// Model-free tile/layout oracle plus matched resident-kernel timing. The
// periodic dyadic operands have an exact analytic product, including tails.
// Performance here is diagnostic only; full trained trajectories remain gates.
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected Vulkan module and exact device description");
    sam3d::neural_session session(argv[1],"Vulkan",0,1,argv[2],true);
    struct shape{int m,n,k,batch;};
    const std::array<shape,8> cases{{{1280,1029,5120,1},{5120,1029,1280,1},
        {3840,1029,1280,1},{1280,1029,1280,1},{33,129,17,2},{67,65,63,2},{5,7,9,2},{1,65,1,1}}};
    for(const auto &s:cases){
        std::vector<ggml_bf16_t> a(size_t(s.m)*s.k*s.batch),b(size_t(s.n)*s.k*s.batch);
        std::vector<float> expected(size_t(s.m)*s.n*s.batch);
        for(int z=0;z<s.batch;++z){
            for(int m=0;m<s.m;++m)for(int k=0;k<s.k;++k)
                a[(size_t(z)*s.m+m)*s.k+k]=ggml_fp32_to_bf16(float((m+k+z)%7-3)/64.f);
            for(int n=0;n<s.n;++n)for(int k=0;k<s.k;++k)
                b[(size_t(z)*s.n+n)*s.k+k]=ggml_fp32_to_bf16(k%7==(n+z)%7?1.f:0.f);
            for(int n=0;n<s.n;++n)for(int m=0;m<s.m;++m){
                const int residue=(n+z)%7,count=s.k>residue?(s.k-1-residue)/7+1:0;
                expected[(size_t(z)*s.n+n)*s.m+m]=float(count*((m+residue+z)%7-3))/64.f;
            }
        }
        for(bool f32_carrier:{false,true})for(const char *mode:{"auto","small","medium","large"}){
#ifdef _WIN32
            _putenv_s("GGML_VK_BF16_MATMUL_TILE",mode);
#else
            setenv("GGML_VK_BF16_MATMUL_TILE",mode,1);
#endif
            std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*16+ggml_graph_overhead_custom(16,false),nullptr,true}),ggml_free);
            if(!c)throw std::bad_alloc();
            auto at=ggml_new_tensor_3d(c.get(),GGML_TYPE_BF16,s.k,s.m,s.batch);
            auto bt=ggml_new_tensor_3d(c.get(),f32_carrier?GGML_TYPE_F32:GGML_TYPE_BF16,s.k,s.n,s.batch);
            auto out=ggml_mul_mat(c.get(),at,bt);ggml_mul_mat_set_prec(out,GGML_PREC_F32);ggml_set_output(out);
            auto graph=ggml_new_graph_custom(c.get(),16,false);ggml_build_forward_expand(graph,out);
            auto buffer=session.allocate(c.get());if(!buffer)throw std::bad_alloc();
            ggml_backend_tensor_set(at,a.data(),0,a.size()*sizeof(ggml_bf16_t));
            if(f32_carrier){
                std::vector<float> carrier(b.size());
                for(size_t i=0;i<b.size();++i)carrier[i]=ggml_bf16_to_fp32(b[i]);
                ggml_backend_tensor_set(bt,carrier.data(),0,carrier.size()*sizeof(float));
            }else ggml_backend_tensor_set(bt,b.data(),0,b.size()*sizeof(ggml_bf16_t));
            std::vector<double> times;
            for(int run=0;run<25;++run){
                const auto begin=std::chrono::steady_clock::now();
                if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("BF16 matmul failed");
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
                if(run>=5)times.push_back(us);
                if(run==0 || run==24){
                    std::vector<float> actual(expected.size());ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
                    for(size_t i=0;i<actual.size();++i)if(actual[i]!=expected[i] || !std::isfinite(actual[i]))
                        throw std::runtime_error("BF16 tile oracle mismatch: mode="+std::string(mode)+" m="+std::to_string(s.m)+" n="+std::to_string(s.n)+" k="+std::to_string(s.k)+" index="+std::to_string(i));
                }
            }
            std::sort(times.begin(),times.end());
            std::cout<<"BF16_TILE_BENCH mode="<<mode<<" carrier="<<(f32_carrier?"f32":"bf16")<<" m="<<s.m<<" n="<<s.n<<" k="<<s.k<<" batch="<<s.batch
                     <<" median_us="<<(times[9]+times[10])*.5<<" min_us="<<times.front()<<" max_us="<<times.back()<<'\n';
        }
    }
    std::cout<<"64 BF16 tile cases passed: exact dyadic oracle, BF16/F32 carriers, large trained shapes, odd dimensions, batches and repeated resident computation\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
