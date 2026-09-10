#include "body_output.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=8 && argc!=9)throw std::invalid_argument("usage: body-output-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- GGUF INPUT OUTPUT [THREADS]");
    auto number=[](const char *p){uint32_t n;auto [end,e]=std::from_chars(p,p+std::strlen(p),n);if(e!=std::errc() || *end)throw std::invalid_argument("invalid number");return n;};
    static_assert(std::endian::native==std::endian::little);std::ifstream input(argv[6],std::ios::binary);auto read=[&](void *p,size_t n){if(!input.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated Body output input");};
    std::array<char,8> magic;read(magic.data(),8);const bool hand=std::string(magic.data(),8)=="S3DPGH01";if(!hand && std::string(magic.data(),8)!="S3DPGO01")throw std::invalid_argument("wrong Body input");std::array<uint32_t,5> dims;read(dims.data(),sizeof(dims));auto [b,d,h,depth,initial_flag]=dims;
    sam3d::pose_shape shape{b,d,h,depth};auto sizes=sam3d::pose_parameter_sizes(shape);std::sort(sizes.begin(),sizes.end());if(initial_flag>1)throw std::invalid_argument("invalid initial flag");
    std::array<int32_t,54> indices;read(indices.data(),sizeof(indices));auto tensor=[&](size_t n){std::vector<float> v(n);read(v.data(),n*4);return v;};auto token=tensor(size_t(b)*d),initial=tensor(initial_flag?b*519:0);
    sam3d::named_floats parameters;for(auto &[name,n]:sizes)parameters[name]=tensor(n);auto mapping=tensor(308*(18439+127));
    std::vector<float> world,wrist,root;std::array<int32_t,145> nonhand{};
    if(hand){world=tensor(9);wrist=tensor(3);root=tensor(3);read(nonhand.data(),sizeof(nonhand));}
    sam3d::hand_pose_config hand_config{world,wrist,root,nonhand};
    if(input.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing Body input");
    sam3d::tensor_archive archive(argv[5],"sam3d.mhr.lod1",sam3d::mhr_lod1_shapes());sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==9?number(argv[8]):1,std::string(argv[4])=="-"?"":argv[4]);std::cerr<<"backend="<<session.description()<<'\n';
    auto arithmetic=std::string(argv[2])=="Vulkan"?sam3d::scalar_division::reciprocal_multiply:sam3d::scalar_division::direct;
    auto result=sam3d::body_pose_geometry(session,archive,shape,token,initial,indices,parameters,mapping,arithmetic,hand?&hand_config:nullptr);std::ofstream output(argv[7],std::ios::binary);
    for(auto &[name,v]:result)if(!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4))throw std::runtime_error("Body output write failed");output.close();if(!output)throw std::runtime_error("Body output close failed");
    if(hand){auto faces=archive.read_i32("mesh.faces",36874*3*4);std::ofstream topology(std::string(argv[7])+".faces",std::ios::binary);
        if(!topology.write(reinterpret_cast<const char*>(faces.data()),faces.size()*4))throw std::runtime_error("hand topology write failed");topology.close();if(!topology)throw std::runtime_error("hand topology close failed");}
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
