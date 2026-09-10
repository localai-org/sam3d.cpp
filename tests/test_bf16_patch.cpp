#include "neural.hpp"
#include "bf16.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <bit>

int main(int argc,char **argv){try{
    if(argc!=3 && argc!=4)throw std::invalid_argument("expected module, fixture, optional backend");
    std::ifstream f(argv[2]);std::string magic;sam3d::patch_shape s;
    if(!(f>>magic) || magic!="S3D_BF16_PATCH_V1" || !(f>>s.batch>>s.channels>>s.height>>s.width>>s.outputs>>s.patch))throw std::invalid_argument("invalid fixture");
    sam3d::validate_patch_shape(s);
    auto read=[&](std::string_view key,uint64_t size){
        std::string name;uint64_t count;if(!(f>>name>>count) || name!=key || count!=size)throw std::invalid_argument("fixture shape/order mismatch");
        std::vector<float> v(size);for(auto &x:v)if(!(f>>x) || !std::isfinite(x))throw std::invalid_argument("invalid fixture value");return v;};
    auto x=read("image",uint64_t(s.batch)*s.channels*s.height*s.width);
    auto w=read("weight",uint64_t(s.outputs)*s.channels*s.patch*s.patch),b=read("bias",s.outputs);
    auto expected=read("output",uint64_t(s.batch)*s.outputs*(s.height/s.patch)*(s.width/s.patch));
    f>>std::ws;if(!f.eof())throw std::invalid_argument("trailing fixture data");
    sam3d::neural_session session(argv[1],argc==4?argv[3]:"CPU",0);
    for(bool capture:{true,false})if(sam3d::patch_embed(session,s,x,w,b,capture,true).tokens!=expected)
        throw std::runtime_error("original BF16 convolution/bias rounding mismatch");
    for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::max()}){
        bool rejected=false;try{sam3d::bf16_values(std::span(&bad,1));}catch(const std::invalid_argument &){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid BF16 conversion accepted");
    }
    std::vector<float> values;
    for(uint32_t sign:{0u,0x80000000u})for(uint32_t exponent=0;exponent<255;++exponent)
        for(uint32_t mantissa:{0u,1u,0x7fffu,0x8000u,0x8001u,0x18000u,0x7effffu}){
            const float value=std::bit_cast<float>(sign|(exponent<<23)|mantissa);
            if(std::isfinite(ggml_bf16_to_fp32(ggml_fp32_to_bf16(value))))values.push_back(value);
        }
    const auto rounded=sam3d::bf16_values(values);
    for(size_t i=0;i<values.size();++i)
        if(std::bit_cast<uint32_t>(rounded[i])!=std::bit_cast<uint32_t>(ggml_bf16_to_fp32(ggml_fp32_to_bf16(values[i]))))
            throw std::runtime_error("bulk BF16 conversion changed scalar ties, signed zeros or subnormals");
    std::cout<<"BF16 original convolution, bias rounding and conversion rejection passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
