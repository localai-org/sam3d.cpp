#include "neural.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void enabled(bool on){
#ifdef _WIN32
    _putenv_s("SAM3D_BATCHED_TRANSFERS",on?"1":"0");
#else
    setenv("SAM3D_BATCHED_TRANSFERS",on?"1":"0",1);
#endif
}
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
}
int main(int argc,char **argv){try{
    if(argc!=2 && argc!=4)throw std::invalid_argument("expected module [backend exact-device-description]");
    const bool gpu=argc==4;
    sam3d::neural_session session(argv[1],gpu?argv[2]:"CPU",0,1,gpu?argv[3]:"",gpu);
    using context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    size_t runs=0;
    for(size_t count:{1,7,513,50000,3,80000,9}){
        context c(ggml_init({ggml_tensor_overhead()*20+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
        if(!c)throw std::bad_alloc();
        auto storage=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,count+4);
        auto x=ggml_view_1d(c.get(),storage,count,4*sizeof(float));
        auto y=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,count);
        auto sum=ggml_add(c.get(),x,y),relu=ggml_relu(c.get(),sum);
        ggml_set_output(sum);ggml_set_output(relu);
        auto graph=ggml_new_graph_custom(c.get(),32,false);ggml_build_forward_expand(graph,relu);
        auto buffer=session.allocate(c.get());check(bool(buffer),"allocation failed");
        std::vector<float> a(count),b(count),out(count+2),positive(count+2);
        const float guard=std::bit_cast<float>(uint32_t(0x7fc12345));
        for(int repeat=0;repeat<12;++repeat){
            enabled(repeat%2!=0);
            for(size_t i=0;i<count;++i){a[i]=float(int((i+repeat)%31)-15)*.25f;b[i]=float(int((i*3+repeat)%17)-8)*.125f;}
            std::fill(out.begin(),out.end(),guard);std::fill(positive.begin(),positive.end(),guard);
            std::array<sam3d::neural_session::upload,2> input{{{x,a.data(),a.size()*4},{y,b.data(),b.size()*4}}};
            std::array<sam3d::neural_session::download,2> output{{{sum,out.data()+1,count*4},{relu,positive.data()+1,count*4}}};
            check(session.compute(graph,input,output)==GGML_STATUS_SUCCESS,"compute failed");
            for(size_t i=0;i<count;++i){const float expected=a[i]+b[i];check(out[i+1]==expected && positive[i+1]==std::max(expected,0.f),"stale/incorrect batched output");}
            for(auto *v:{&out,&positive})for(size_t i:{size_t(0),count+1})check(std::bit_cast<uint32_t>((*v)[i])==0x7fc12345,"output guard overwritten");
            ++runs;
            const auto batches=session.batched_transfer_count();
            auto reject=[&]{bool rejected=false;try{session.compute(graph,input,output);}catch(const std::invalid_argument &){rejected=true;}check(rejected,"invalid transfer accepted");check(session.batched_transfer_count()==batches,"invalid request submitted");};
            output[1].bytes-=4;reject();output[1].bytes+=4;
            output[1].data=nullptr;reject();output[1].data=positive.data()+1;
            input[0].tensor=nullptr;reject();input[0].tensor=x;
            input[1].data=nullptr;reject();input[1].data=b.data();
            check(session.compute(graph,input,output)==GGML_STATUS_SUCCESS,"recovery failed");
        }
    }
    // A valid large transfer exceeds the pinned pool budget and must use the
    // synchronous path without truncation or an unbounded staging allocation.
    enabled(true);
    constexpr size_t large=9*1024*1024;
    context c(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead_custom(8,false),nullptr,true}),ggml_free);
    auto x=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,large),y=ggml_dup(c.get(),x);
    ggml_set_output(y);auto graph=ggml_new_graph_custom(c.get(),8,false);ggml_build_forward_expand(graph,y);
    auto buffer=session.allocate(c.get());check(bool(buffer),"large allocation failed");
    std::vector<float> source(large,.125f),target(large,0.f);source.front()=-2.f;source.back()=7.f;
    std::array<sam3d::neural_session::upload,1> input{{{x,source.data(),source.size()*4}}};
    std::array<sam3d::neural_session::download,1> output{{{y,target.data(),target.size()*4}}};
    auto before=session.batched_transfer_count();
    check(session.compute(graph,input,output)==GGML_STATUS_SUCCESS && std::memcmp(source.data(),target.data(),source.size()*4)==0,"large fallback mismatch");
    check(session.batched_transfer_count()==before,"oversized pinned batch");
    check(gpu?before>0:before==0,"requested backend batching/fallback was not exercised");
    std::cout<<runs<<" alternating transfer cases passed, pinned batches="<<before<<", guards/rejections/recovery/large fallback passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
