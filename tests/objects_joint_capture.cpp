#include "objects_image.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("usage: objects-joint-capture INPUT OUTPUT");
    static_assert(std::endian::native==std::endian::little);std::ifstream in(argv[1],std::ios::binary);
    auto read=[&](void *v,size_t n){if(!in.read(static_cast<char *>(v),n))throw std::invalid_argument("truncated joint input");};
    std::array<char,8>tag;std::array<uint32_t,5>a;std::array<double,2>options;read(tag.data(),8);read(a.data(),sizeof(a));read(options.data(),sizeof(options));
    if(std::string(tag.data(),8)!="S3DPJM01" || !a[0] || !a[1] || !a[2] || !a[3] || a[0]>4096 || a[1]>4096 || a[2]>2048 || a[3]>2048 || a[4]<4*a[0] || a[4]>4*a[0]+4096)throw std::invalid_argument("invalid joint input header");
    std::vector<uint8_t> rgba(uint64_t(a[1])*a[4]);std::vector<float> xyz(uint64_t(3)*a[2]*a[3]);read(rgba.data(),rgba.size());read(xyz.data(),xyz.size()*4);
    if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing joint input");
    auto result=sam3d::objects_prepare_pointmap_joint(rgba,a[0],a[1],a[4],xyz,a[2],a[3],options[0],options[1],true);
    result.taps.emplace("90.result_rgb",std::move(result.rgb));
    result.taps.emplace("91.result_mask",std::move(result.mask));
    result.taps.emplace("92.result_pointmap",std::move(result.pointmap));
    std::ofstream out(argv[2],std::ios::binary);auto write=[&](const void *p,size_t n){if(!out.write(static_cast<const char *>(p),n))throw std::runtime_error("joint capture write failed");};
    write("S3DST001",8);uint32_t count=result.taps.size();write(&count,4);
    for(auto &[key,v]:result.taps){uint32_t length=key.size();uint64_t n=v.size();write(&length,4);write(key.data(),length);write(&n,8);write(v.data(),n*4);}
    out.close();if(!out)throw std::runtime_error("joint capture close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
