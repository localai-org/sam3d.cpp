#pragma once
#include "ggml-alloc.h"
#include <cstring>

inline void test_bf16_norm_affine(sam3d::neural_session &session) {
    const char *old=std::getenv("GGML_VK_FUSE_BF16_NORM_AFFINE");
    struct restore {
        bool present;std::string value;
        static void set(bool exists,const char *v) {
#ifdef _WIN32
            _putenv_s("GGML_VK_FUSE_BF16_NORM_AFFINE",exists?v:"");
#else
            if(exists)setenv("GGML_VK_FUSE_BF16_NORM_AFFINE",v,1);else unsetenv("GGML_VK_FUSE_BF16_NORM_AFFINE");
#endif
        }
        ~restore(){set(present,value.c_str());}
    } guard{old!=nullptr,old?old:""};
    size_t cases=0;
    std::cout<<"NORM_AFFINE_TEST_BEGIN"<<std::endl;
    for(bool reuse:{false,true})for(int n:{1,7,511,512,513,1024})for(int rows:{1,3,513})for(int mode=0;mode<10;++mode) {
        const int total=n*rows,stride=n+(mode==9?3:0),offset=mode==8?3:0;
        const int bn=mode==1?1:mode==2?total:n,cn=mode==3?1:mode==4?total:n;
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true}),ggml_free);
        if(!c)throw std::bad_alloc();
        auto xs=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,stride*rows+offset);
        auto bs=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,bn+offset),cs=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,cn+offset);
        ggml_set_input(xs);ggml_set_input(bs);ggml_set_input(cs);
        auto x=ggml_view_2d(c.get(),xs,n,rows,stride*4,offset*4);
        auto b=ggml_view_2d(c.get(),bs,bn==1?1:n,bn==total?rows:1,(bn==1?1:n)*4,offset*4);
        auto bias=ggml_view_2d(c.get(),cs,cn==1?1:n,cn==total?rows:1,(cn==1?1:n)*4,offset*4);
        auto norm=ggml_norm(c.get(),x,1e-5f);
        auto mul=ggml_mul(c.get(),norm,b);
        auto add=ggml_add(c.get(),mul,bias);
        auto narrow=ggml_cast(c.get(),add,GGML_TYPE_BF16);
        auto out=ggml_cast(c.get(),narrow,GGML_TYPE_F32);
        std::vector<ggml_tensor *> observed{out};
        if(mode==5)observed.push_back(norm);
        if(mode==6)observed.push_back(ggml_scale(c.get(),norm,2.f));
        if(mode==7)observed.push_back(mul);
        auto graph=ggml_new_graph_custom(c.get(),64,false);
        ggml_build_forward_expand(graph,b);ggml_build_forward_expand(graph,bias);
        for(auto t:observed){ggml_set_output(t);ggml_build_forward_expand(graph,t);}
        sam3d::neural_session::scratch_buffer allocation;
        std::unique_ptr<ggml_gallocr,decltype(&ggml_gallocr_free)> planner(nullptr,ggml_gallocr_free);
        if(reuse){planner.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(session.backend())));
            if(!planner || !ggml_gallocr_alloc_graph(planner.get(),graph))throw std::bad_alloc();}
        else{allocation=session.allocate(c.get());if(!allocation)throw std::bad_alloc();}
        std::vector<float> xv(ggml_nelements(xs)),bv(bn+offset),cv(cn+offset),result(total);
        for(int repeat=0;repeat<2;++repeat) {
            for(size_t j=0;j<xv.size();++j)xv[j]=float(int((j*37+repeat*19)%1027)-513)/91.f;
            for(size_t j=0;j<bv.size();++j)bv[j]=float(int((j*13+repeat*31)%127)-63)/39.f;
            for(size_t j=0;j<cv.size();++j)cv[j]=float(int((j*71+repeat*11)%419)-209)/123.f;
            std::vector<std::vector<unsigned char>> baseline;
            for(bool on:{false,true}) {
                restore::set(true,on?"1":"0");
                ggml_backend_tensor_set(xs,xv.data(),0,xv.size()*4);
                ggml_backend_tensor_set(bs,bv.data(),0,bv.size()*4);
                ggml_backend_tensor_set(cs,cv.data(),0,cv.size()*4);
                if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("norm-affine graph failed");
                for(size_t j=0;j<observed.size();++j) {
                    std::vector<unsigned char> bytes(ggml_nbytes(observed[j]));ggml_backend_tensor_get(observed[j],bytes.data(),0,bytes.size());
                    if(!on)baseline.push_back(bytes);
                    else if(bytes!=baseline[j]) {
                        size_t k=0;while(k<bytes.size() && bytes[k]==baseline[j][k])++k;
                        float before=0,after=0;const size_t at=k/4*4;
                        std::memcpy(&before,baseline[j].data()+at,4);std::memcpy(&after,bytes.data()+at,4);
                        throw std::runtime_error("norm-affine changed bits n="+std::to_string(n)+" rows="+std::to_string(rows)+" mode="+std::to_string(mode)+" reuse="+std::to_string(reuse)+" index="+std::to_string(at/4)+" before="+std::to_string(before)+" after="+std::to_string(after));
                    }
                }
                ggml_backend_tensor_get(out,result.data(),0,result.size()*4);
                // Independent double oracle checks the mathematical operation;
                // the stricter device off/on check above preserves actual bits.
                for(int row=0;row<rows;++row) {
                    double mean=0,var=0;
                    for(int j=0;j<n;++j)mean+=xv[offset+row*stride+j];
                    mean/=n;
                    for(int j=0;j<n;++j){double d=xv[offset+row*stride+j]-mean;var+=d*d;}
                    var/=n;
                    for(int j=0;j<n;++j){int i=row*n+j;
                        double expected=(xv[offset+row*stride+j]-mean)/std::sqrt(var+1e-5)*bv[offset+i%bn]+cv[offset+i%cn];
                        if(!std::isfinite(result[i]) || std::abs(result[i]-expected)>.008*std::max(1.,std::abs(expected)))throw std::runtime_error("norm-affine mathematical oracle failed");
                    }
                }
            }
            ++cases;
        }
    }
    std::cout<<cases<<" exact norm-affine off/on cases passed\nNORM_AFFINE_TEST_END"<<std::endl;
}
