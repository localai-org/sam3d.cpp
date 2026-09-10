#include "condition_case.hpp"
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>

int main(int argc,char **argv) {
    try {
        if(argc!=7 && argc!=8)throw std::invalid_argument("usage: condition-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT [THREADS]");
        auto number=[](const char *t){uint32_t n;auto [end,error]=std::from_chars(t,t+std::strlen(t),n);
            if(error!=std::errc() || *end)throw std::invalid_argument("invalid numeric argument");return n;};
        static_assert(std::endian::native==std::endian::little);
        std::ifstream file(argv[5],std::ios::binary);
        auto read=[&](void *p,size_t n){if(!file.read(static_cast<char *>(p),n))throw std::runtime_error("truncated conditioning fixture");};
        std::array<char,8> magic;read(magic.data(),8);
        if(std::string(magic.data(),8)!="S3DCND01")throw std::invalid_argument("invalid conditioning fixture");
        std::array<uint32_t,16> dimensions;read(dimensions.data(),sizeof(dimensions));condition_case value(dimensions);
        auto tensor=[&](uint64_t count){std::vector<float> v(count);read(v.data(),count*4);return v;};
        auto &s=value.shape;const uint64_t b=s.batch,grid=(s.height/s.patch)*(s.width/s.patch);
        sam3d::named_floats input,parameters;
        input["features"]=tensor(b*s.context_dim*grid);input["rays"]=tensor(b*2*s.height*s.width);
        input["cliff"]=tensor(b*3);input["keypoints"]=tensor(b*s.points*3);
        if(value.previous)input["previous"]=tensor(b*(s.pose_dim+3));
        for(auto &[name,size]:value.sizes())parameters[name]=tensor(size);
        if(file.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing conditioning fixture");
        sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==8?number(argv[7]):1,std::string(argv[4])=="-"?"":argv[4]);
        const auto division=std::string(argv[2])=="Vulkan"?sam3d::scalar_division::reciprocal_multiply:sam3d::scalar_division::direct;
        std::cerr<<"backend="<<session.description()<<" scalar_division="<<(division==sam3d::scalar_division::direct?"direct":"reciprocal_multiply")<<'\n';
        auto taps=value.run(session,input,parameters,division);std::ofstream output(argv[6],std::ios::binary);
        for(auto &[name,v]:taps)if(!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4))throw std::runtime_error("conditioning output write failed");
        output.close();if(!output)throw std::runtime_error("conditioning output close failed");
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
