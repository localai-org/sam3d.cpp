#include "body_hand_unmirror.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <iostream>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("usage: hand-unmirror-capture INPUT OUTPUT");
    static_assert(std::endian::native==std::endian::little);
    std::ifstream in(argv[1],std::ios::binary);auto read=[&](void *p,size_t n){if(!in.read(static_cast<char*>(p),n))throw std::invalid_argument("truncated unmirror input");};
    std::array<char,8> magic;read(magic.data(),8);uint32_t b,width;read(&b,4);read(&width,4);
    if(std::string(magic.data(),8)!="S3DHUF01" || b<1 || b>64 || width<1 || width>32766)throw std::invalid_argument("invalid unmirror header");
    auto tensor=[&](size_t n){std::vector<float> v(n);read(v.data(),n*4);return v;};
    auto mean=tensor(68),components=tensor(28*68),scale=tensor(b*28),rotations=tensor(b*127*9),hand=tensor(b*108),center=tensor(b*2);
    if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing unmirror input");
    auto result=sam3d::body_unmirror_left(b,width,scale,rotations,hand,center,mean,components);
    std::ofstream out(argv[2],std::ios::binary);for(auto &[_,v]:result)if(!out.write(reinterpret_cast<const char*>(v.data()),v.size()*4))throw std::runtime_error("unmirror write failed");
    out.close();if(!out)throw std::runtime_error("unmirror close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
