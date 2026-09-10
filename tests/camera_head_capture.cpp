#include "body_camera_head.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv){
    try{
        if(argc!=7 && argc!=8)throw std::invalid_argument("usage: camera-head-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT [THREADS]");
        auto number=[](const char *t){uint32_t n;auto [end,error]=std::from_chars(t,t+std::strlen(t),n);
            if(error!=std::errc() || *end)throw std::invalid_argument("invalid number");return n;};
        static_assert(std::endian::native==std::endian::little);std::ifstream file(argv[5],std::ios::binary);
        auto read=[&](void *p,size_t n){if(!file.read(static_cast<char *>(p),n))throw std::runtime_error("truncated camera-head fixture");};
        std::array<char,8> magic;read(magic.data(),8);if(std::string(magic.data(),8)!="S3DCHD01")throw std::invalid_argument("invalid camera-head fixture");
        std::array<uint32_t,7> dims;read(dims.data(),sizeof(dims));float scale;read(&scale,4);auto [b,d,f,depth,n,center,initial]=dims;
        if(center>1 || initial>1)throw std::invalid_argument("invalid camera-head flags");
        sam3d::camera_head_shape s{b,d,f,depth,n,scale,bool(center)};auto sizes=sam3d::camera_head_parameter_sizes(s);std::sort(sizes.begin(),sizes.end());
        auto tensor=[&](uint64_t n){std::vector<float> v(n);read(v.data(),n*4);return v;};
        auto token=tensor(uint64_t(b)*d),estimate=initial?tensor(uint64_t(b)*3):std::vector<float>{};
        auto points=tensor(uint64_t(b)*n*3),centers=tensor(uint64_t(b)*2),box=tensor(b),image=tensor(uint64_t(b)*2),k=tensor(uint64_t(b)*9);
        sam3d::named_floats parameters;for(auto &[name,size]:sizes)parameters[name]=tensor(size);
        if(file.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing camera-head fixture");
        sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==8?number(argv[7]):1,std::string(argv[4])=="-"?"":argv[4]);
        std::cerr<<"backend="<<session.description()<<'\n';
        auto taps=sam3d::body_camera_head(session,s,token,estimate,points,centers,box,image,k,parameters);std::ofstream output(argv[6],std::ios::binary);
        for(auto &[name,v]:taps)if(!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4))throw std::runtime_error("camera-head output write failed");
        output.close();if(!output)throw std::runtime_error("camera-head output close failed");
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
