#include "objects_fuser.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=8)throw std::invalid_argument("usage: fuser-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT THREADS");
    auto number=[](const char *p){uint32_t n;auto [e,ec]=std::from_chars(p,p+std::strlen(p),n);if(ec!=std::errc() || *e)throw std::invalid_argument("invalid numeric argument");return n;};
    static_assert(std::endian::native==std::endian::little);std::ifstream in(argv[5],std::ios::binary);
    auto read=[&](void *p,size_t n){if(!in.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated fuser input");};
    std::array<char,8>tag;std::array<uint32_t,5>a;std::array<double,2>m;read(tag.data(),8);read(a.data(),sizeof(a));read(m.data(),sizeof(m));
    if(std::string(tag.data(),8)!="S3DFUS01" || !a[1] || a[1]>8 || !a[2] || a[2]>16 || a[3]>1 || a[4]>1)throw std::invalid_argument("invalid fuser header");
    sam3d::fuser_shape s;s.batch=a[0];s.pre_norm=a[3];s.random_position=a[4];s.projection_multiplier=m[0];s.compression_multiplier=m[1];s.embed_dims.resize(a[1]);read(s.embed_dims.data(),a[1]*4);
    for(uint32_t i=0;i<a[2];++i){sam3d::fuser_input x;uint32_t drop;read(&x.embedder,4);read(&x.tokens,4);read(&x.position,4);read(&drop,4);if(drop>1)throw std::invalid_argument("invalid drop flag");x.forced_drop=drop;s.inputs.push_back(x);}
    auto sizes=sam3d::fuser_parameter_sizes(s);
    auto tensor=[&](size_t n){std::vector<float> v(n);read(v.data(),n*4);return v;};
    std::vector<std::vector<float>> embeddings;for(auto x:s.inputs)embeddings.push_back(tensor(uint64_t(s.batch)*x.tokens*s.embed_dims[x.embedder]));
    sam3d::named_floats params;for(auto &[key,n]:sizes)params[key]=tensor(n);
    if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing fuser input");
    sam3d::neural_session session(argv[1],argv[2],number(argv[3]),number(argv[7]),std::string(argv[4])=="-"?"":argv[4]);std::cerr<<session.description()<<'\n';
    auto taps=sam3d::objects_fuse(session,s,embeddings,params),compact=sam3d::objects_fuse(session,s,embeddings,params,false);
    if(compact.size()!=1 || compact.at("90.output")!=taps.at("90.output"))throw std::runtime_error("fusion observer changed result");
    taps.emplace("91.independent_output",std::move(compact.at("90.output")));
    std::ofstream out(argv[6],std::ios::binary);auto write=[&](const void *p,size_t n){if(!out.write(static_cast<const char *>(p),n))throw std::runtime_error("fuser output write failed");};
    write("S3DST001",8);uint32_t count=taps.size();write(&count,4);
    for(auto &[key,v]:taps){uint32_t len=key.size();uint64_t n=v.size();write(&len,4);write(key.data(),len);write(&n,8);write(v.data(),n*4);}
    out.close();if(!out)throw std::runtime_error("fuser output close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
