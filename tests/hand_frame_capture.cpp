#include "body_hand_frame.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <iostream>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("usage: hand-frame-capture INPUT OUTPUT");static_assert(std::endian::native==std::endian::little);
    std::ifstream in(argv[1],std::ios::binary);auto read=[&](void *p,size_t n){if(!in.read(static_cast<char*>(p),n))throw std::invalid_argument("truncated hand frame input");};
    std::array<char,8> magic;read(magic.data(),8);uint32_t b;read(&b,4);if(std::string(magic.data(),8)!="S3DHFR01" || !b || b>64)throw std::invalid_argument("invalid hand frame input");
    auto tensor=[&](size_t n){std::vector<float> v(n);read(v.data(),n*4);return v;};
    auto rotation=tensor(b*3),translation=tensor(b*3),world=tensor(9),wrist=tensor(3),root=tensor(3),parameters=tensor(b*204),points=tensor(b*308*3);
    std::array<int32_t,145> indices;read(indices.data(),indices.size()*4);if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing hand frame input");
    auto result=sam3d::body_hand_frame(b,rotation,translation,world,wrist,root);
    result["05.masked_parameters"]=sam3d::body_hand_mask_parameters(b,parameters,indices);result["06.masked_keypoints"]=sam3d::body_hand_mask_keypoints(b,points);
    std::ofstream out(argv[2],std::ios::binary);for(auto &[_,v]:result)if(!out.write(reinterpret_cast<const char*>(v.data()),v.size()*4))throw std::runtime_error("hand frame output failed");out.close();if(!out)throw std::runtime_error("hand frame close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
