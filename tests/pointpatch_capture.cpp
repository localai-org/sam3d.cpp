#include "objects_pointpatch.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
int main(int argc,char **argv){try{
    if(argc!=9)throw std::invalid_argument("usage: pointpatch-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT_DIR THREADS CHUNK");
    auto number=[](const char *p){uint32_t n;auto [end,e]=std::from_chars(p,p+std::strlen(p),n);if(e!=std::errc() || *end)throw std::invalid_argument("invalid number");return n;};
    static_assert(std::endian::native==std::endian::little);std::ifstream f(argv[5],std::ios::binary);
    auto read=[&](void *p,size_t n){if(!f.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated PointPatch input");};
    std::array<char,8>tag;std::array<uint32_t,10>a;read(tag.data(),8);read(a.data(),sizeof(a));
    if(std::string(tag.data(),8)!="S3DPPT01" || a[7]>1 || a[8]>1 || a[9]>1)throw std::invalid_argument("invalid PointPatch header");
    sam3d::pointpatch_shape s{a[0],a[1],a[2],a[3],a[4],a[5],sam3d::point_remapping(a[6]),bool(a[8]),bool(a[9])};auto sizes=sam3d::pointpatch_parameter_sizes(s);
    std::vector<float> xyz(uint64_t(s.batch)*3*s.height*s.width);read(xyz.data(),xyz.size()*4);
    std::vector<uint8_t> mask(a[7]?uint64_t(s.batch)*s.side*s.side:0);read(mask.data(),mask.size());
    sam3d::named_floats params;for(auto &[key,n]:sizes){auto &v=params[key];v.resize(n);read(v.data(),n*4);}if(f.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing PointPatch input");
    sam3d::neural_session session(argv[1],argv[2],number(argv[3]),number(argv[7]),std::string(argv[4])=="-"?"":argv[4]);std::cerr<<session.description()<<'\n';
    std::filesystem::path dir(argv[6]);std::filesystem::create_directories(dir);
    struct file{std::ofstream stream;uint64_t total=0,next=0;};std::map<std::string,file> files;
    auto result=sam3d::objects_pointpatch(session,s,xyz,mask,params,[&](const std::string &key,std::span<const float> v,uint64_t offset,uint64_t total){
        auto &out=files[key];if(!out.stream.is_open()){out.stream.open(dir/(key+".bin"),std::ios::binary);out.total=total;}
        if(offset!=out.next || total!=out.total || v.size()>total-offset)throw std::runtime_error("incomplete/overlapping diagnostic chunks");
        if(!out.stream.write(reinterpret_cast<const char *>(v.data()),v.size_bytes()))throw std::runtime_error("PointPatch capture write failed");out.next+=v.size();
        if(key=="90.output")std::cerr<<"windows output "<<out.next<<'/'<<total<<'\n';
    },number(argv[8]));
    for(auto &[key,out]:files){out.stream.close();if(out.next!=out.total || !out.stream)throw std::runtime_error("incomplete PointPatch capture");}
    std::ofstream out(dir/"result.bin",std::ios::binary);if(!out.write(reinterpret_cast<const char *>(result.data()),result.size()*4))throw std::runtime_error("result write failed");out.close();if(!out)throw std::runtime_error("result close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
