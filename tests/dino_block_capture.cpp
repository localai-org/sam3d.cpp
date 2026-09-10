#include "dino_block.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        const bool bf16=argc>1 && std::string_view(argv[argc-1])=="--bf16";
        if(bf16)--argc;
        if (argc!=7 && argc!=8) throw std::invalid_argument("usage: block-capture MODULE CPU|Vulkan DEVICE EXPECTED_DESCRIPTION|- INPUT OUTPUT [THREADS]");
        uint32_t index;
        auto [end,ec]=std::from_chars(argv[3],argv[3]+std::strlen(argv[3]),index);
        if (ec!=std::errc() || *end) throw std::invalid_argument("invalid device index");
        uint32_t threads=1;
        if (argc==8) {
            auto [last,error]=std::from_chars(argv[7],argv[7]+std::strlen(argv[7]),threads);
            if (error!=std::errc() || *last) throw std::invalid_argument("invalid threads");
        }
        static_assert(std::endian::native==std::endian::little);
        std::ifstream in(argv[5],std::ios::binary);
        auto read=[&](void *p,size_t bytes) {
            if (!in.read(static_cast<char *>(p),bytes)) throw std::runtime_error("truncated block input");
        };
        std::array<char,8> magic; read(magic.data(),8);
        if (std::string(magic.data(),8)!="S3DBLK01") throw std::invalid_argument("invalid block input");
        std::array<uint32_t,7> dims; read(dims.data(),sizeof(dims));
        auto [b,h,w,d,heads,prefix,hidden]=dims;
        sam3d::dino_shape s{b,h,w,d,heads,prefix,hidden};
        auto sizes=sam3d::dino_parameter_sizes(s);
        std::vector<float> x(uint64_t(b)*(h*w+prefix)*d);
        read(x.data(),x.size()*4);
        sam3d::named_floats parameters;
        for (auto &[name,size]:sizes) {
            auto &v=parameters[name]; v.resize(size); read(v.data(),v.size()*4);
        }
        if (in.peek()!=std::ifstream::traits_type::eof()) throw std::invalid_argument("trailing block input");
        sam3d::neural_session session(argv[1],argv[2],index,threads,std::string(argv[4])=="-"?"":argv[4]);
        std::cerr<<"backend="<<session.description()<<'\n';
        sam3d::weight_map checked=parameters;
        auto taps=bf16?sam3d::dino_block_bf16(session,s,x,checked):sam3d::dino_block(session,s,x,parameters);
        std::ofstream output(argv[6],std::ios::binary);
        for (auto &[name,v]:taps) {
            if (!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4))
                throw std::runtime_error("block output write failed");
        }
        output.close(); if (!output) throw std::runtime_error("block output close failed");
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
    return 0;
}
