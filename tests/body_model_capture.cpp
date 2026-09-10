#include "body_model.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
int main(int argc,char **argv){try{
    const bool bf16=argc>1 && std::string_view(argv[argc-1])=="--bf16";if(bf16)--argc;
    if(argc!=12)throw std::invalid_argument("usage: body-model-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- BACKBONE_GGUF BRANCH_GGUF MHR_GGUF IMAGE_INPUT OUTPUT THREADS OPERATIONS_0_OR_1");
    auto number=[](const char *p){uint32_t n;auto [end,e]=std::from_chars(p,p+std::strlen(p),n);if(e!=std::errc() || *end)throw std::invalid_argument("invalid integer");return n;};
    const uint32_t operations=number(argv[11]);if(operations>1)throw std::invalid_argument("invalid operations flag");
    static_assert(std::endian::native==std::endian::little);
    std::ifstream in(argv[8],std::ios::binary);auto read=[&](void *p,size_t n){if(!in.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated image input");};
    std::array<char,8> magic;read(magic.data(),magic.size());if(std::string(magic.data(),magic.size())!="S3DIMG01")throw std::invalid_argument("invalid image input");
    std::array<uint32_t,3> dims;read(dims.data(),sizeof(dims));auto [width,height,stride]=dims;
    if(!width || !height || width>32766 || height>32766 || uint64_t(width)*height>16000000 || stride!=uint64_t(width)*3)throw std::invalid_argument("invalid image dimensions");
    std::array<float,4> box,intrinsics;read(box.data(),sizeof(box));read(intrinsics.data(),sizeof(intrinsics));
    std::vector<uint8_t> rgb(uint64_t(height)*stride);read(rgb.data(),rgb.size());if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing image input");
    sam3d::body_model model(argv[5],argv[6],argv[7],argv[1],argv[2],number(argv[3]),number(argv[10]),std::string(argv[4])=="-"?"":argv[4],bf16);
    std::cerr<<"Loaded GGUF-only Body pose branch: "<<model.description()<<'\n';
    auto result=model.infer_rgb(rgb,width,height,stride,box,intrinsics,operations);
    std::ofstream out(argv[9],std::ios::binary);
    for(auto &[name,value]:result)if(!out.write(reinterpret_cast<const char *>(value.data()),value.size()*4))throw std::runtime_error("result write failed");
    out.close();if(!out)throw std::runtime_error("result close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
