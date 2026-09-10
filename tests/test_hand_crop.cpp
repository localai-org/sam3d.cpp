#include "body_hand_crop.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv){try{
    static_assert(std::endian::native==std::endian::little);
    if(argc!=2)throw std::invalid_argument("expected original regression file");std::ifstream in(argv[1]);std::string magic;size_t cases;
    if(!(in>>magic>>cases) || magic!="S3D_HAND_CROP_REGRESSION_V1" || cases!=2)throw std::runtime_error("invalid hand regression");
    auto require=[](bool b){if(!b)throw std::runtime_error("hand crop regression failed");};
    auto reject=[&](auto f){bool bad=false;try{f();}catch(const std::invalid_argument &){bad=true;}require(bad);};
    for(size_t i=0;i<cases;++i){
        uint32_t w,h,stride,crop;in>>w>>h>>stride>>crop;std::array<float,6> affine;std::array<float,8> boxes;std::array<float,4> camera;
        for(auto &v:affine)in>>v;for(auto &v:boxes)in>>v;for(auto &v:camera)in>>v;
        size_t bytes;std::string hex;in>>bytes>>hex;require(bytes==uint64_t(stride)*h && bytes<=64*1024 && hex.size()==bytes*2);
        std::vector<uint8_t> rgb(bytes);auto digit=[](char c)->uint8_t{if(c>='0' && c<='9')return c-'0';if(c>='a' && c<='f')return c-'a'+10;throw std::runtime_error("invalid hex");};
        for(size_t j=0;j<bytes;++j)rgb[j]=digit(hex[j*2])*16+digit(hex[j*2+1]);
        auto result=sam3d::body_prepare_hands(rgb,w,h,stride,boxes,affine,camera,crop);
        for(auto key:{"left.rays","right.rays","left.cliff","right.cliff"})result.erase(key);
        size_t n;in>>n;require(n==23 && result.size()==n);
        for(size_t j=0;j<n;++j){std::string name;uint64_t size,expected;in>>name>>size>>expected;require(result.contains(name) && result.at(name).size()==size);
            uint64_t actual=14695981039346656037ull;auto values=std::as_bytes(std::span(result.at(name)));
            for(auto value:values)actual=(actual^std::to_integer<uint8_t>(value))*1099511628211ull;
            if(expected!=actual)throw std::runtime_error("hand crop mismatch: "+name);
        }
        auto bad=affine;bad[1]=.1f;reject([&]{sam3d::body_hand_boxes(boxes,bad,w,crop);});
        bad=affine;bad[0]=0;reject([&]{sam3d::body_hand_boxes(boxes,bad,w,crop);});
        auto invalid=boxes;invalid[2]=0;reject([&]{sam3d::body_hand_boxes(invalid,affine,w,crop);});
        invalid=boxes;invalid[0]=std::numeric_limits<float>::quiet_NaN();reject([&]{sam3d::body_hand_boxes(invalid,affine,w,crop);});
        reject([&]{sam3d::body_prepare_hands(std::span(rgb).first(2),w,h,stride,boxes,affine,camera,crop);});
        reject([&]{sam3d::body_prepare_hands(rgb,w,h,UINT64_MAX,boxes,affine,camera,crop);});
    }
    require(bool(in));std::string trailing;require(!(in>>trailing));
    std::cout<<"Original hand crop pixels/geometry, mirroring, padding and rejection tests pass\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
