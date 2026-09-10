#include "body_prompt.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if(argc!=7 && argc!=8) throw std::invalid_argument("usage: prompt-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT [THREADS]");
        auto number=[](const char *t){uint32_t v;auto [end,error]=std::from_chars(t,t+std::strlen(t),v);
            if(error!=std::errc() || *end) throw std::invalid_argument("invalid number");return v;};
        static_assert(std::endian::native==std::endian::little);
        std::ifstream input(argv[5],std::ios::binary);
        auto read=[&](void *p,size_t bytes){if(!input.read(static_cast<char *>(p),bytes)) throw std::runtime_error("truncated prompt fixture");};
        std::array<char,8> magic;read(magic.data(),8);
        if(std::string(magic.data(),8)!="S3DPRM01") throw std::invalid_argument("invalid prompt fixture");
        std::array<uint32_t,8> dims;read(dims.data(),sizeof(dims));auto [b,n,d,j,h,w,ih,iw]=dims;
        sam3d::prompt_shape s{b,n,d,j,h,w};auto sizes=sam3d::prompt_parameter_sizes(s);std::sort(sizes.begin(),sizes.end());
        if(ih<1 || ih>32766 || iw<1 || iw>32766) throw std::invalid_argument("invalid pixel image size");
        auto tensor=[&](uint64_t count){std::vector<float> values(count);read(values.data(),count*4);return values;};
        auto points=tensor(uint64_t(b)*n*3),pixels=tensor(uint64_t(b)*n*2);sam3d::named_floats parameters;
        for(auto &[name,size]:sizes) parameters[name]=tensor(size);
        if(input.peek()!=std::ifstream::traits_type::eof()) throw std::invalid_argument("trailing prompt fixture");
        sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==8?number(argv[7]):1,std::string(argv[4])=="-"?"":argv[4]);
        std::cerr<<"backend="<<session.description()<<'\n';
        const auto division=std::string(argv[2])=="Vulkan"?sam3d::scalar_division::reciprocal_multiply:sam3d::scalar_division::direct;
        std::cerr<<"scalar_division="<<(division==sam3d::scalar_division::direct?"direct":"reciprocal_multiply")<<'\n';
        auto taps=sam3d::body_prompt_encode(session,s,points,parameters,division);
        auto pixel_taps=sam3d::body_position_pixels(session,b,n,d,ih,iw,pixels,parameters.at("pe_layer.positional_encoding_gaussian_matrix"),division);
        for(auto &[name,v]:pixel_taps) taps["30.pixel."+name]=std::move(v);
        std::ofstream output(argv[6],std::ios::binary);
        for(auto &[name,v]:taps) if(!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4)) throw std::runtime_error("prompt output write failed");
        output.close();if(!output) throw std::runtime_error("prompt output close failed");
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
