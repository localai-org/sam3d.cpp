#pragma once
#include "bf16.hpp"
#include "ggml-alloc.h"
#include <cstring>

inline void test_bf16_affine(sam3d::neural_session &session){
    const char *old=std::getenv("GGML_VK_FUSE_BF16_AFFINE");
    struct restore{
        bool present;std::string value;
        static void set(bool exists,const char *v){
#ifdef _WIN32
            _putenv_s("GGML_VK_FUSE_BF16_AFFINE",exists?v:"");
#else
            if(exists)setenv("GGML_VK_FUSE_BF16_AFFINE",v,1);else unsetenv("GGML_VK_FUSE_BF16_AFFINE");
#endif
        }
        ~restore(){set(present,value.c_str());}
    } guard{old!=nullptr,old?old:""};
    auto rounded=[](float v){return ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));};
    std::cout<<"AFFINE_TEST_BEGIN"<<std::endl;size_t cases=0,rounding_differences=0,fma_differences=0;
    for(bool reuse:{false,true})for(bool first_round:{false,true})
    for(int n:{1,7,511,512,513,1280})for(int mode=0;mode<26;++mode){
        const int sizes[]={n*6,n,1,n*3};
        const int bn=mode<16?sizes[mode%4]:mode==23?6:n*6;
        const int cn=mode<16?sizes[mode/4]:n*6;
        const int stride=n+(mode==24?5:0),xo=mode==25?3:0,bo=mode==25?5:0,co=mode==25?7:0;
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true}),ggml_free);
        if(!c)throw std::bad_alloc();
        auto xs=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,stride*6+xo);
        auto bs=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,bn+bo),cs=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,cn+co);
        ggml_set_input(xs);ggml_set_input(bs);ggml_set_input(cs);
        auto x=ggml_view_3d(c.get(),xs,n,3,2,stride*4,stride*3*4,xo*4);
        auto operand=[&](ggml_tensor *storage,int count,int offset,bool column){
            int dim0=count==1?1:n,dim1=count<=n?1:count==n*3?3:3,dim2=count==n*6?2:1;
            if(column){dim0=1;dim1=3;dim2=2;}
            return ggml_view_3d(c.get(),storage,dim0,dim1,dim2,dim0*4,dim0*dim1*4,offset*4);
        };
        auto b=operand(bs,bn,bo,mode==23),bias=operand(cs,cn,co,false);
        auto mul=ggml_mul(c.get(),x,b);ggml_tensor *middle=mul;
        if(first_round)middle=sam3d::bf16_round(c.get(),mul);
        // Reversed ADD only accepts a full-sized first operand.
        auto add=(mode%2 && cn==n*6)?ggml_add(c.get(),bias,middle):ggml_add(c.get(),middle,bias);
        auto narrow=ggml_cast(c.get(),add,GGML_TYPE_BF16);
        auto out=ggml_cast(c.get(),narrow,GGML_TYPE_F32);
        std::vector<ggml_tensor *> observed{out};
        if(mode==16)observed.push_back(mul);
        if(mode==17)observed.push_back(add);
        if(mode==18)observed.push_back(narrow);
        if(mode==19)observed.push_back(middle);
        if(mode==20)observed.push_back(ggml_scale(c.get(),mul,2.f));
        if(mode==21)observed.push_back(ggml_scale(c.get(),middle,2.f));
        if(mode==22)observed.push_back(ggml_scale(c.get(),add,2.f));
        auto graph=ggml_new_graph_custom(c.get(),64,false);
        ggml_build_forward_expand(graph,b);ggml_build_forward_expand(graph,bias);
        for(auto t:observed){ggml_set_output(t);ggml_build_forward_expand(graph,t);}
        sam3d::neural_session::scratch_buffer allocation;
        std::unique_ptr<ggml_gallocr,decltype(&ggml_gallocr_free)> planner(nullptr,ggml_gallocr_free);
        if(reuse){planner.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(session.backend())));
            if(!planner || !ggml_gallocr_alloc_graph(planner.get(),graph))throw std::bad_alloc();}
        else{allocation=session.allocate(c.get());if(!allocation)throw std::bad_alloc();}
        std::vector<float> xv(ggml_nelements(xs)),bv(bn+bo),cv(cn+co),result(n*6);
        for(int repeat=0;repeat<2;++repeat){
            for(size_t i=0;i<xv.size();++i)xv[i]=float(int((i*37+repeat*19)%1027)-513)/91.f;
            for(size_t i=0;i<bv.size();++i)bv[i]=float(int((i*13+repeat*31)%127)-63)/39.f;
            for(size_t i=0;i<cv.size();++i)cv[i]=float(int((i*71+repeat*11)%419)-209)/123.f;
            // Separate F32 multiply/add produces zero; FMA produces -2^-46.
            // BF16 can represent that residual, so contraction is observable.
            xv[xo]=1.f+0x1p-23f;bv[bo]=1.f-0x1p-23f;cv[co]=-1.f;
            std::vector<std::vector<unsigned char>> baseline;
            for(bool on:{false,true}){
                restore::set(true,on?"1":"0");
                ggml_backend_tensor_set(xs,xv.data(),0,xv.size()*4);
                ggml_backend_tensor_set(bs,bv.data(),0,bv.size()*4);
                ggml_backend_tensor_set(cs,cv.data(),0,cv.size()*4);
                if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("affine graph failed");
                for(size_t j=0;j<observed.size();++j){std::vector<unsigned char> bytes(ggml_nbytes(observed[j]));
                    ggml_backend_tensor_get(observed[j],bytes.data(),0,bytes.size());
                    if(!on)baseline.push_back(bytes);
                    else if(bytes!=baseline[j])throw std::runtime_error("affine changed bits n="+std::to_string(n)+" mode="+std::to_string(mode));}
                ggml_backend_tensor_get(out,result.data(),0,result.size()*4);
                for(int i=0;i<n*6;++i){
                    volatile float product=xv[xo+(i/n)*stride+i%n]*bv[bo+(mode==23?i/n:i%bn)];
                    const float sum=(first_round?rounded(product):product)+cv[co+i%cn];
                    const float expected=rounded(sum);
                    if(std::memcmp(&expected,&result[i],4))throw std::runtime_error("affine independent oracle failed");
                    if(rounded(product+cv[co+i%cn])!=rounded(rounded(product)+cv[co+i%cn]))++rounding_differences;
                    if(!first_round && rounded(std::fma(xv[xo+(i/n)*stride+i%n],bv[bo+(mode==23?i/n:i%bn)],cv[co+i%cn]))!=expected)++fma_differences;
                }
            }
            ++cases;
        }
    }
    if(!rounding_differences)throw std::runtime_error("affine corpus does not distinguish the two rounding contracts");
    if(!fma_differences)throw std::runtime_error("affine corpus does not detect FMA contraction");
    std::cout<<cases<<" exact affine off/on cases passed; "<<rounding_differences<<" rounding-boundary and "<<fma_differences<<" FMA negative controls\nAFFINE_TEST_END"<<std::endl;
}
