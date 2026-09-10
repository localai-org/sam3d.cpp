#include "camera_encoder.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if (argc!=7 && argc!=8) throw std::invalid_argument("usage: camera-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT [THREADS]");
        auto number=[](const char *text) {
            uint32_t value; auto [end,error]=std::from_chars(text,text+std::strlen(text),value);
            if (error!=std::errc() || *end) throw std::invalid_argument("invalid numeric argument");
            return value;
        };
        static_assert(std::endian::native==std::endian::little);
        std::ifstream input(argv[5],std::ios::binary);
        auto read=[&](void *p,size_t bytes) {
            if (!input.read(static_cast<char *>(p),bytes)) throw std::runtime_error("truncated camera input");
        };
        std::array<char,8> magic; read(magic.data(),8);
        if (std::string(magic.data(),8)!="S3DCAM01") throw std::invalid_argument("invalid camera input");
        std::array<uint32_t,5> dimensions; read(dimensions.data(),sizeof(dimensions));
        auto [b,h,w,p,d]=dimensions; sam3d::camera_shape shape{b,h,w,p,d};
        sam3d::validate_camera_shape(shape);
        auto tensor=[&](uint64_t count) {std::vector<float> v(count); read(v.data(),v.size()*4); return v;};
        auto features=tensor(uint64_t(b)*d*(h/p)*(w/p)), rays=tensor(uint64_t(b)*2*h*w);
        sam3d::named_floats parameters;
        parameters["conv.weight"]=tensor(uint64_t(d)*(d+99));
        parameters["norm.weight"]=tensor(d); parameters["norm.bias"]=tensor(d);
        if (input.peek()!=std::ifstream::traits_type::eof()) throw std::invalid_argument("trailing camera input");
        sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==8?number(argv[7]):1,
            std::string(argv[4])=="-"?"":argv[4]);
        std::cerr<<"backend="<<session.description()<<'\n';
        auto taps=sam3d::camera_encode(session,shape,features,rays,parameters);
        std::ofstream output(argv[6],std::ios::binary);
        for (auto &[name,v]:taps)
            if (!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4))
                throw std::runtime_error("camera output write failed");
        output.close(); if (!output) throw std::runtime_error("camera output close failed");
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
