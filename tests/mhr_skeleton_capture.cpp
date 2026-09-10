#include "mhr_skeleton.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=8 && argc!=9)throw std::invalid_argument("usage: mhr-skeleton-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- GGUF INPUT OUTPUT [THREADS]");
    auto number=[](const char *p){uint32_t n;auto [end,e]=std::from_chars(p,p+std::strlen(p),n);if(e!=std::errc() || *end)throw std::invalid_argument("invalid number");return n;};
    static_assert(std::endian::native==std::endian::little);std::ifstream input(argv[6],std::ios::binary);
    auto read=[&](void *p,size_t n){if(!input.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated MHR skeleton input");};
    std::array<char,8> magic;read(magic.data(),8);if(std::string(magic.data(),8)!="S3DMHS01")throw std::invalid_argument("wrong MHR skeleton input");uint32_t batch;read(&batch,4);if(batch<1 || batch>2)throw std::invalid_argument("invalid MHR batch");
    std::vector<float> parameters(size_t(batch)*204);read(parameters.data(),parameters.size()*4);if(input.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing MHR input");
    sam3d::tensor_archive archive(argv[5],"sam3d.mhr.lod1",sam3d::mhr_lod1_shapes());
    sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==9?number(argv[8]):1,std::string(argv[4])=="-"?"":argv[4]);std::cerr<<"backend="<<session.description()<<'\n';
    auto result=sam3d::mhr_skeleton(session,archive,batch,parameters);std::ofstream output(argv[7],std::ios::binary);
    auto write=[&](const void *p,size_t n){if(!output.write(static_cast<const char *>(p),n))throw std::runtime_error("MHR capture write failed");};
    write("S3DMHO01",8);uint32_t count=result.f32.size()+result.f64.size();write(&count,4);
    auto tensor=[&](const std::string &name,const auto &v,uint32_t dtype){uint32_t size=name.size();uint64_t count=v.size();write(&size,4);write(name.data(),size);write(&dtype,4);write(&count,8);write(v.data(),v.size()*sizeof(v[0]));};
    for(auto &[name,v]:result.f32)tensor(name,v,4);for(auto &[name,v]:result.f64)tensor(name,v,8);output.close();if(!output)throw std::runtime_error("MHR capture close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
