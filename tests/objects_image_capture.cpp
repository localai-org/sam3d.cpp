#include "objects_image.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("usage: objects-image-capture INPUT OUTPUT");
    static_assert(std::endian::native==std::endian::little);std::ifstream f(argv[1],std::ios::binary);
    auto read=[&](void *p,size_t n){if(!f.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated image capture");};
    std::array<char,8>tag;std::array<uint32_t,4>d;std::array<double,2>options;
    read(tag.data(),8);read(d.data(),sizeof(d));read(options.data(),sizeof(options));
    if(std::string(tag.data(),8)!="S3DOIM01" || !d[0] || !d[1] || d[0]>4096 || d[1]>4096 || d[2]<uint64_t(d[0])*4 || d[2]>uint64_t(d[0])*4+4096)throw std::invalid_argument("invalid image header");
    std::vector<uint8_t> rgb(uint64_t(d[1])*d[2]);read(rgb.data(),rgb.size());if(f.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing image capture");
    auto taps=sam3d::objects_prepare_rgba(rgb,d[0],d[1],d[2],{d[3],options[0],options[1]});std::ofstream out(argv[2],std::ios::binary);
    for(auto &[key,v]:taps)if(!out.write(reinterpret_cast<const char *>(v.data()),v.size()*4))throw std::runtime_error("cannot write image capture");
    out.close();if(!out)throw std::runtime_error("cannot close image capture");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
