#include "body_pose.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=7 && argc!=8)throw std::invalid_argument("usage: pose-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT [THREADS]");
    auto number=[](const char *t){uint32_t n;auto [end,error]=std::from_chars(t,t+std::strlen(t),n);if(error!=std::errc() || *end)throw std::invalid_argument("invalid number");return n;};
    static_assert(std::endian::native==std::endian::little);std::ifstream file(argv[5],std::ios::binary);
    auto read=[&](void *p,size_t n){if(!file.read(static_cast<char *>(p),n))throw std::runtime_error("truncated pose input");};
    std::array<char,8> magic;read(magic.data(),8);if(std::string(magic.data(),8)!="S3DPSE01")throw std::invalid_argument("invalid pose input");
    std::array<uint32_t,5> dims;read(dims.data(),sizeof(dims));auto [b,d,h,depth,has_initial]=dims;if(has_initial>1)throw std::invalid_argument("invalid pose flag");
    sam3d::pose_shape s{b,d,h,depth};auto sizes=sam3d::pose_parameter_sizes(s);std::sort(sizes.begin(),sizes.end());std::array<int32_t,54> indices;read(indices.data(),sizeof(indices));
    auto tensor=[&](uint64_t n){std::vector<float> v(n);read(v.data(),n*4);return v;};auto token=tensor(uint64_t(b)*d),initial=tensor(has_initial?uint64_t(b)*519:0);
    sam3d::named_floats parameters;for(auto &[name,size]:sizes)parameters[name]=tensor(size);if(file.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing pose input");
    sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==8?number(argv[7]):1,std::string(argv[4])=="-"?"":argv[4]);std::cerr<<"backend="<<session.description()<<'\n';
    auto result=sam3d::body_pose(session,s,token,initial,indices,parameters);std::ofstream output(argv[6],std::ios::binary);
    for(auto &[name,v]:result)if(!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4))throw std::runtime_error("pose output write failed");output.close();if(!output)throw std::runtime_error("pose output close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
