#include "feedback_case.hpp"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=7 && argc!=8)throw std::invalid_argument("usage: feedback-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT [THREADS]");
    auto number=[](const char *t){uint32_t n;auto [end,error]=std::from_chars(t,t+std::strlen(t),n);if(error!=std::errc() || *end)throw std::invalid_argument("invalid number");return n;};
    static_assert(std::endian::native==std::endian::little);std::ifstream file(argv[5],std::ios::binary);
    auto read=[&](void *p,size_t bytes){if(!file.read(static_cast<char *>(p),bytes))throw std::runtime_error("truncated feedback fixture");};
    std::array<char,8> magic;read(magic.data(),8);if(std::string(magic.data(),8)!="S3DFBK01")throw std::invalid_argument("invalid feedback fixture");
    std::array<uint32_t,15> dims;read(dims.data(),sizeof(dims));auto s=feedback_dimensions(dims);
    std::vector<int32_t> idx(s.keypoints),idx3(s.keypoints3d);read(idx.data(),idx.size()*4);read(idx3.data(),idx3.size()*4);
    auto tensor=[&](uint64_t count){std::vector<float> v(count);read(v.data(),count*4);return v;};sam3d::named_floats input,parameters;
    for(auto &[name,size]:feedback_inputs(s))input[name]=tensor(size);
    auto sizes=sam3d::feedback_parameter_sizes(s);std::sort(sizes.begin(),sizes.end());for(auto &[name,size]:sizes)parameters[name]=tensor(size);
    if(file.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing feedback fixture");
    sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==8?number(argv[7]):1,std::string(argv[4])=="-"?"":argv[4]);std::cerr<<"backend="<<session.description()<<'\n';
    auto taps=run_feedback(session,s,input,parameters,idx,idx3);std::ofstream output(argv[6],std::ios::binary);
    for(auto &[name,v]:taps)if(!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4))throw std::runtime_error("feedback output write failed");
    output.close();if(!output)throw std::runtime_error("feedback output close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
