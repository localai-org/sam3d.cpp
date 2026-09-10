#pragma once
#include "bf16.hpp"
#include "ggml-alloc.h"
#include <cstring>

// Independent off/on graph execution, with explicit observation/consumer guards.
// The full trained pipeline remains a separate required acceptance gate.
inline void test_bf16_silu_gate(sam3d::neural_session &session){
    std::cout<<"SILU_GATE_TEST_BEGIN"<<std::endl;
    const char *previous=std::getenv("GGML_VK_FUSE_BF16_SILU_GATE");
    const std::string saved=previous?previous:"";const bool present=previous;
    struct restore{
        bool present;std::string value;
        static void set(bool exists,const char *value){
#ifdef _WIN32
            _putenv_s("GGML_VK_FUSE_BF16_SILU_GATE",exists?value:"");
#else
            if(exists)setenv("GGML_VK_FUSE_BF16_SILU_GATE",value,1);else unsetenv("GGML_VK_FUSE_BF16_SILU_GATE");
#endif
        }
        ~restore(){set(present,value.c_str());}
    } guard{present,saved};
    size_t cases=0;
    for(bool reuse:{false,true})for(int n:{1,7,511,512,513,8193})for(int mode=0;mode<17;++mode){
        if(std::getenv("GGML_VK_PERF_LOGGER"))std::cout<<"SILU_GATE_CASE n="<<n<<" mode="<<mode<<" reuse="<<reuse<<std::endl;
        const int rows=3,stride=n+(mode==1?5:0);
        const int input_offset=mode==13?3:0,gate_offset=mode==14?5:0;
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true}),ggml_free);
        if(!c)throw std::bad_alloc();
        auto storage=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,stride*rows+input_offset);
        auto x=ggml_view_2d(c.get(),storage,n,rows,stride*4,input_offset*4);
        auto gate_storage=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,n*(mode==12?1:rows)+gate_offset);
        auto gate=ggml_view_2d(c.get(),gate_storage,n,mode==12?1:rows,n*4,gate_offset*4);
        ggml_set_input(storage);ggml_set_input(gate_storage);
        auto silu=ggml_silu(c.get(),x);
        auto narrow=ggml_cast(c.get(),silu,GGML_TYPE_BF16);
        auto wide=ggml_cast(c.get(),narrow,GGML_TYPE_F32);
        auto mul=(mode==10 || mode==12)?ggml_mul(c.get(),wide,gate):ggml_mul(c.get(),gate,wide);
        auto narrow_out=ggml_cast(c.get(),mul,GGML_TYPE_BF16);
        auto out=mode>=15?ggml_cpy(c.get(),narrow_out,mul):ggml_cast(c.get(),narrow_out,GGML_TYPE_F32);
        // Marking a view OUTPUT recursively marks its destination OUTPUT too.
        // A downstream copy keeps the explicit-destination positive unobserved,
        // as in the trained FFN. Mode 16 retains that conservative fallback.
        if(mode==15)out=ggml_cont(c.get(),out);
        std::vector<ggml_tensor *> observations{out};
        if(mode==2)observations.push_back(silu);
        if(mode==3)observations.push_back(narrow);
        if(mode==4 || mode==16)observations.push_back(wide);
        if(mode==5)observations.push_back(ggml_scale(c.get(),silu,2.f));
        if(mode==6)observations.push_back(ggml_scale(c.get(),wide,2.f));
        if(mode==7)observations.push_back(mul);
        if(mode==8)observations.push_back(narrow_out);
        if(mode==9)observations.push_back(ggml_scale(c.get(),mul,2.f));
        if(mode==11)observations.push_back(ggml_cast(c.get(),narrow,GGML_TYPE_F32));
        auto graph=ggml_new_graph_custom(c.get(),64,false);
        // Materialize the gate view before the chain in either multiply order;
        // otherwise DFS inserts that VIEW between the casts and multiply.
        ggml_build_forward_expand(graph,gate);
        for(auto t:observations){ggml_set_output(t);ggml_build_forward_expand(graph,t);}
        sam3d::neural_session::scratch_buffer allocation;
        std::unique_ptr<ggml_gallocr,decltype(&ggml_gallocr_free)> planner(nullptr,ggml_gallocr_free);
        if(reuse){
            planner.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(session.backend())));
            if(!planner || !ggml_gallocr_alloc_graph(planner.get(),graph))throw std::bad_alloc();
        }else{allocation=session.allocate(c.get());if(!allocation)throw std::bad_alloc();}
        std::vector<float> input(ggml_nelements(storage)),weights(ggml_nelements(gate_storage)),actual(n*rows);
        for(int repeat=0;repeat<2;++repeat){
            for(size_t i=0;i<input.size();++i)input[i]=float(int((i*197+repeat*13)%1481)-740)/37.f;
            constexpr float special[]={0.f,-0.f,1e-38f,-1e-38f,88.f,-100.f,1.00390625f,-1.01171875f};
            for(size_t i=0;i<std::size(special) && i<input.size();++i)input[i]=special[i];
            for(size_t i=0;i<weights.size();++i)weights[i]=float(int((i*31+repeat*23)%149)-74)/19.f;
            std::vector<std::vector<unsigned char>> baseline;
            for(bool enabled:{false,true}){
                restore::set(true,enabled?"1":"0");
                // Graph allocation may reuse input storage after its last use.
                ggml_backend_tensor_set(storage,input.data(),0,input.size()*4);
                ggml_backend_tensor_set(gate_storage,weights.data(),0,weights.size()*4);
                if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("SiLU gate graph failed");
                for(size_t j=0;j<observations.size();++j){
                    std::vector<unsigned char> bytes(ggml_nbytes(observations[j]));
                    ggml_backend_tensor_get(observations[j],bytes.data(),0,bytes.size());
                    if(!enabled)baseline.push_back(bytes);
                    else if(bytes!=baseline[j])throw std::runtime_error("SiLU gate changed bits: n="+std::to_string(n)+" mode="+std::to_string(mode)+" tap="+std::to_string(j));
                }
                ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
                for(int row=0;row<rows;++row)for(int col=0;col<n;++col){
                    const float v=input[input_offset+row*stride+col];
                    const float first=ggml_bf16_to_fp32(ggml_fp32_to_bf16(v/(1.f+std::exp(-v))));
                    const float expected=ggml_bf16_to_fp32(ggml_fp32_to_bf16(first*weights[gate_offset+(mode==12?0:row)*n+col]));
                    const float got=actual[row*n+col];
                    // exp/divide may vary between host and GPU near BF16 ties;
                    // exact off/on checks above are the optimization assertion.
                    if(!std::isfinite(got) || std::abs(got-expected)>.01f*(1+std::abs(expected)))throw std::runtime_error("SiLU gate independent arithmetic oracle failed");
                }
            }
            ++cases;
        }
    }
    std::cout<<cases<<" SiLU/BF16 gate cases passed off/on exactly (tails, padded, observed/shared intermediates, broadcasts, descriptor offsets, both multiply orders, graph storage reuse)"<<std::endl;
    std::cout<<"SILU_GATE_TEST_END"<<std::endl;
}
