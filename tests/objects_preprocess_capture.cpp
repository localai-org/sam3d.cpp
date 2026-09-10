#include "objects_preprocess.hpp"
#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("usage: objects-preprocess-capture INPUT OUTPUT_DIRECTORY");
    static_assert(std::endian::native==std::endian::little);
    std::ifstream in(argv[1],std::ios::binary);
    auto read=[&](void *p,size_t n){if(!in.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated preprocess input");};
    std::array<char,8>tag;std::array<uint32_t,9>a;std::array<double,2>crop;
    read(tag.data(),8);read(a.data(),sizeof(a));read(crop.data(),sizeof(crop));
    if(std::string(tag.data(),8)!="S3DOPP01" || !a[0] || !a[1] || a[0]>4096 || a[1]>4096 || a[2]<a[0]*4 || a[2]>a[0]*4+4096 || !a[3] || !a[4] || a[3]>2048 || a[4]>2048 || a[7]>1 || a[8]>1)throw std::invalid_argument("invalid preprocess input header");
    sam3d::objects_preprocess_options opt;opt.image_side=a[5];opt.point_side=a[6];opt.normalize=a[7];opt.point_nan_padding=a[8];opt.box_factor=crop[0];opt.padding=crop[1];
    for(auto *o:{&opt.object_normalizer,&opt.full_normalizer}){
        std::array<uint32_t,3>b;std::array<double,4>d;read(b.data(),sizeof(b));read(d.data(),sizeof(d));
        if(b[1]>1 || b[2]>1)throw std::invalid_argument("invalid normalizer flags");
        o->mode=sam3d::objects_ssi_mode(b[0]);o->allow_override=b[1];o->raise_on_no_valid_points=b[2];o->quantile_drop=d[0];o->clip=d[1];o->scale_factor=d[2];o->log_disparity_shift=d[3];
    }
    sam3d::validate_objects_preprocess_options(opt);
    std::vector<uint8_t> rgba(uint64_t(a[1])*a[2]);std::vector<float> xyz(uint64_t(3)*a[3]*a[4]);read(rgba.data(),rgba.size());read(xyz.data(),xyz.size()*4);
    if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing preprocess input");
    std::filesystem::path directory(argv[2]);std::filesystem::create_directories(directory);
    auto write=[&](const std::string &key,std::span<const float> v){
        std::ofstream out(directory/(key+".bin"),std::ios::binary);uint32_t count=1,length=key.size();uint64_t n=v.size();
        out.write("S3DST001",8);out.write(reinterpret_cast<char *>(&count),4);out.write(reinterpret_cast<char *>(&length),4);out.write(key.data(),length);out.write(reinterpret_cast<char *>(&n),8);out.write(reinterpret_cast<const char *>(v.data()),n*4);out.close();
        if(!out)throw std::runtime_error("preprocess capture write failed");
    };
    auto result=sam3d::objects_preprocess_pointmap(rgba,a[0],a[1],a[2],xyz,a[3],a[4],opt,write);
    // Independently serialize actual returned buffers, not diagnostic aliases.
    for(auto &[key,v]:result)write("90.final."+key,v);
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
