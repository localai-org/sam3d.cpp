#include "neural.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

// Independent double-precision oracle and serialized output for comparison
// between process-start pipeline specializations. No model weights needed.
int main(int argc,char **argv){try{
    if(argc!=4)throw std::invalid_argument("expected module exact-device-description output.bin");
    sam3d::neural_session session(argv[1],"Vulkan",0,1,argv[2],true);
    std::ofstream saved(argv[3],std::ios::binary);if(!saved)throw std::runtime_error("output open failed");
    struct shape{int m,n,k,h,b;};
    const std::array<shape,14> cases{{{64,5,1029,20,1},{1029,5,64,20,1},{65,5,1029,2,2},
        {1,1,1,1,1},{3,2,3,2,2},{5,3,17,1,2},{7,4,63,2,1},{9,6,127,1,1},
        {17,7,128,1,2},{33,8,1024,1,1},{257,1,3000,1,1},{309,3,18566,1,1},
        {65,8,3000,2,2},{17,5,1029,2,1}}};
    size_t checked=0;double worst=0;
    for(auto s:cases){
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*16+ggml_graph_overhead_custom(16,false),nullptr,true}),ggml_free);
        if(!c)throw std::bad_alloc();
        auto a=ggml_new_tensor_4d(c.get(),GGML_TYPE_F32,s.k,s.m,s.h,1);
        auto b=ggml_new_tensor_4d(c.get(),GGML_TYPE_F32,s.k,s.n,s.h,s.b);
        auto out=ggml_mul_mat(c.get(),a,b);ggml_mul_mat_set_prec(out,GGML_PREC_F32);ggml_set_output(out);
        auto graph=ggml_new_graph_custom(c.get(),16,false);ggml_build_forward_expand(graph,out);
        auto allocation=session.allocate(c.get());
        std::vector<float> av(ggml_nelements(a)),bv(ggml_nelements(b)),actual(ggml_nelements(out));
        for(int repeat=0;repeat<4;++repeat){
            const bool dyadic=repeat%2==0;const float factor=dyadic?.03125f:.037f;
            for(size_t i=0;i<av.size();++i)av[i]=float(int((i*13+repeat*11)%31)-15)*factor;
            for(size_t i=0;i<bv.size();++i)bv[i]=float(int((i*7+repeat*3)%23)-11)*factor;
            ggml_backend_tensor_set(a,av.data(),0,av.size()*4);ggml_backend_tensor_set(b,bv.data(),0,bv.size()*4);
            auto begin=std::chrono::steady_clock::now();
            if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("matvec failed");
            const auto us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
            ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
            for(int z=0;z<s.b*s.h;++z)for(int n=0;n<s.n;++n)for(int m=0;m<s.m;++m){double expected=0;
                for(int k=0;k<s.k;++k)expected+=double(av[(size_t(z%s.h)*s.m+m)*s.k+k])*bv[(size_t(z)*s.n+n)*s.k+k];
                const auto i=(size_t(z)*s.n+n)*s.m+m;const double error=std::abs(double(actual[i])-expected);worst=std::max(worst,error);
                if(!std::isfinite(actual[i]) || (dyadic?actual[i]!=float(expected):error>5e-5*(1+std::abs(expected))))
                    throw std::runtime_error("matvec oracle failed m="+std::to_string(s.m)+" n="+std::to_string(s.n)+" k="+std::to_string(s.k)+" index="+std::to_string(i));
            }
            saved.write(reinterpret_cast<const char *>(actual.data()),actual.size()*4);if(!saved)throw std::runtime_error("output write failed");
            std::cout<<"MATVEC m="<<s.m<<" n="<<s.n<<" k="<<s.k<<" heads="<<s.h<<" batches="<<s.b<<" repeat="<<repeat<<" us="<<us<<'\n';++checked;
        }
    }
    std::cout<<checked<<" matvec cases passed, max_abs="<<worst<<" (dyadic exact, non-dyadic double oracle, row/column/reduction tails, broadcast batches)\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
