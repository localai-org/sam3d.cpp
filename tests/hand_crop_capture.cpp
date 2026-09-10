#include "body_hand_crop.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("usage: hand-crop-capture INPUT OUTPUT");
    static_assert(std::endian::native==std::endian::little);
    std::ifstream in(argv[1],std::ios::binary);
    auto read=[&](void *p,size_t n){if(!in.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated hand crop input");};
    std::array<char,8> magic;read(magic.data(),8);if(std::string(magic.data(),8)!="S3DHCP01")throw std::invalid_argument("invalid hand crop magic");
    std::array<uint32_t,4> dims;read(dims.data(),sizeof(dims));auto [w,h,stride,crop]=dims;
    if(!w || !h || w>32766 || h>32766 || uint64_t(w)*h>16000000 || stride<uint64_t(w)*3 || uint64_t(stride)*h>64*1024*1024 || !crop || crop>512)throw std::invalid_argument("invalid hand crop dimensions");
    std::array<float,6> affine;std::array<float,8> boxes;std::array<float,4> camera;
    read(affine.data(),sizeof(affine));read(boxes.data(),sizeof(boxes));read(camera.data(),sizeof(camera));
    std::vector<uint8_t> rgb(uint64_t(stride)*h);read(rgb.data(),rgb.size());if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing hand crop data");
    auto result=sam3d::body_prepare_hands(rgb,w,h,stride,boxes,affine,camera,crop);
    // Rays/CLIFF already have separate original-operation regression coverage.
    for(auto name:{"left.rays","right.rays","left.cliff","right.cliff"})result.erase(name);
    std::ofstream out(argv[2],std::ios::binary);
    for(auto &[_,value]:result)if(!out.write(reinterpret_cast<const char*>(value.data()),value.size()*4))throw std::runtime_error("hand output write failed");
    out.close();if(!out)throw std::runtime_error("hand output close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
