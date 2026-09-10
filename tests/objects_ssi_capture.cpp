#include "objects_ssi.hpp"
#include <bit>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("usage: objects-ssi-capture INPUT OUTPUT");
    static_assert(std::endian::native==std::endian::little);std::ifstream in(argv[1],std::ios::binary);
    auto read=[&](void *v,size_t n){if(!in.read(static_cast<char *>(v),n))throw std::invalid_argument("truncated SSI capture input");};
    std::array<char,8>tag;std::array<uint32_t,9>a;std::array<double,4>v;std::array<float,3>so,to;
    read(tag.data(),8);read(a.data(),sizeof(a));read(v.data(),sizeof(v));read(so.data(),sizeof(so));read(to.data(),sizeof(to));
    if(std::string(tag.data(),8)!="S3DSSI01" || a[5]>1 || a[6]>1 || a[7]>1 || a[8]>1)throw std::invalid_argument("invalid SSI header");
    sam3d::objects_ssi_shape shape{a[0],a[1],a[2],a[3]};sam3d::objects_ssi_options o{sam3d::objects_ssi_mode(a[4]),v[0],v[1],v[2],v[3],bool(a[5]),bool(a[6])};
    sam3d::validate_objects_ssi(shape,o);std::vector<float> xyz(uint64_t(3)*a[0]*a[1]),mask(uint64_t(a[2])*a[3]);
    read(xyz.data(),xyz.size()*4);read(mask.data(),mask.size()*4);if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing SSI capture input");
    auto result=sam3d::objects_normalize_pointmap(shape,o,xyz,mask,a[7]?std::span<const float>(so):std::span<const float>(),a[8]?std::span<const float>(to):std::span<const float>(),true);
    sam3d::objects_denormalize_pointmap(shape.height,shape.width,o.mode,result.pointmap,result.scale,result.shift,&result.taps);
    std::ofstream out(argv[2],std::ios::binary);auto write=[&](const void *p,size_t n){if(!out.write(static_cast<const char *>(p),n))throw std::runtime_error("cannot write SSI capture");};
    write("S3DST001",8);uint32_t count=result.taps.size();write(&count,4);
    for(auto &[key,values]:result.taps){uint32_t length=key.size();uint64_t n=values.size();write(&length,4);write(key.data(),length);write(&n,8);write(values.data(),n*4);}
    out.close();if(!out)throw std::runtime_error("cannot close SSI capture");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
