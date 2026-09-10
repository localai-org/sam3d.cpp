#include "neural.hpp"
#include "bf16.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <iostream>
#include <memory>

// Isolated trained attention oracle, not image-to-mesh acceptance. The format
// carries original post-RoPE Q/K and V in [batch,heads,tokens,head_dim] order.
int main(int argc,char **argv){try{
    if(argc!=5)throw std::invalid_argument("expected MODULE DESCRIPTION INPUT OUTPUT");
    static_assert(std::endian::native==std::endian::little);
    std::ifstream in(argv[3],std::ios::binary);
    auto read=[&](void *data,size_t bytes){if(!in.read(static_cast<char *>(data),bytes))throw std::invalid_argument("truncated attention input");};
    std::array<char,8> magic;read(magic.data(),8);
    if(std::string_view(magic.data(),8)!="S3DATT01")throw std::invalid_argument("invalid attention input");
    std::array<uint32_t,4> dims;read(dims.data(),sizeof(dims));auto [b,h,n,d]=dims;
    if(b<1 || b>2 || h<1 || h>20 || n<1 || n>1029 || d<1 || d>128 || d%8)throw std::invalid_argument("attention dimensions out of bounds");
    std::array<std::vector<float>,3> values;
    for(auto &v:values){v.resize(uint64_t(b)*h*n*d);read(v.data(),v.size()*4);
        if(sam3d::bf16_values(v)!=v)throw std::invalid_argument("attention input must contain finite BF16 values");}
    if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing attention input");
    sam3d::neural_session session(argv[1],"Vulkan",0,1,argv[2],true);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
    if(!ctx)throw std::bad_alloc();auto c=ctx.get();
    auto q=ggml_new_tensor_4d(c,GGML_TYPE_F32,d,n,h,b);
    auto k=ggml_new_tensor_4d(c,GGML_TYPE_BF16,d,n,h,b),v=ggml_new_tensor_4d(c,GGML_TYPE_BF16,d,n,h,b);
    auto out=ggml_flash_attn_ext(c,q,k,v,nullptr,float(1./std::sqrt(double(d))),0,0);
    ggml_flash_attn_ext_set_prec(out,GGML_PREC_F32);
    auto rounded=sam3d::bf16_round(c,out);
    auto graph=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(graph,rounded);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)if(!ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)))throw std::invalid_argument("unsupported attention operation");
    auto buffer=session.allocate(c);if(!buffer)throw std::bad_alloc();
    ggml_backend_tensor_set(q,values[0].data(),0,values[0].size()*4);
    sam3d::set_bf16_tensor(k,values[1]);sam3d::set_bf16_tensor(v,values[2]);
    if(ggml_backend_graph_compute(session.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("attention graph failed");
    std::vector<float> result(uint64_t(b)*h*n*d);
    ggml_backend_tensor_get(rounded,result.data(),0,result.size()*4);
    for(float x:result)if(!std::isfinite(x))throw std::runtime_error("nonfinite attention result");
    std::ofstream output(argv[4],std::ios::binary);
    output.write(reinterpret_cast<const char *>(result.data()),result.size()*4);output.close();
    if(!output)throw std::runtime_error("attention output write failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
