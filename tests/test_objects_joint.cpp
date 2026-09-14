#include "objects_image.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected original joint fixture");std::ifstream in(argv[1]);std::string magic;uint32_t count;
    if(!(in>>magic>>count) || magic!="S3D_JOINT_REGRESSION_V1" || count!=6)throw std::runtime_error("invalid joint fixture");
    auto values=[&](size_t n){std::vector<float> v(n);for(auto &x:v){std::string token;if(!(in>>token))throw std::runtime_error("truncated fixture");auto [end,e]=std::from_chars(token.data(),token.data()+token.size(),x);if(e!=std::errc() || end!=token.data()+token.size())throw std::runtime_error("bad fixture value");}return v;};
    const std::set<std::string> points={"04.aligned_pointmap","06.clean","07.resized","12.crop_pointmap","22.pointmap"};unsigned checked=0;bool soft=false;
    for(uint32_t i=0;i<count;++i){uint32_t w,h,ph,pw,stride;double factor,pad;
        if(!(in>>w>>h>>ph>>pw>>stride>>factor>>pad) || w>32 || h>32 || ph>32 || pw>32 || stride>w*4+32)throw std::runtime_error("bad fixture header");
        std::vector<uint8_t> rgba(size_t(h)*stride);for(auto &v:rgba){unsigned x;if(!(in>>x) || x>255)throw std::runtime_error("bad fixture byte");v=uint8_t(x);}
        auto xyz=values(size_t(3)*ph*pw);auto out=sam3d::objects_prepare_pointmap_joint(rgba,w,h,stride,xyz,ph,pw,factor,pad,true);
        uint32_t nt;if(!(in>>nt) || nt!=out.taps.size())throw std::runtime_error("joint tap count mismatch");
        for(auto &[key,v]:out.taps){std::string name;size_t n;if(!(in>>name>>n) || name!=key || n!=v.size())throw std::runtime_error("joint tap shape/name mismatch");
            auto ref=values(n);double maximum=0,error2=0,ref2=0;
            for(size_t j=0;j<n;++j){float x=v[j],y=ref[j];
                if(std::isnan(x)!=std::isnan(y) || std::isinf(x)!=std::isinf(y) || (std::isinf(x) && std::signbit(x)!=std::signbit(y)))throw std::runtime_error("joint nonfinite mismatch "+key);
                if(!std::isfinite(y)){if(!points.contains(key))throw std::runtime_error("nonfinite non-point tensor");continue;}
                double delta=double(x)-y;maximum=std::max(maximum,std::abs(delta));error2+=delta*delta;ref2+=double(y)*y;
                if(key=="21.mask" && x>0 && x<1)soft=true;
            }
            if(maximum>1e-5 || std::sqrt(error2)/std::max(std::sqrt(ref2),1e-12)>2e-5 || (!points.contains(key) && maximum!=0))throw std::runtime_error("original joint mismatch "+key);
            ++checked;
        }
        if(out.rgb!=out.taps.at("20.rgb") || out.mask!=out.taps.at("21.mask") || out.rgb.size()!=size_t(3)*out.height*out.width)throw std::runtime_error("incorrect joint result");
        auto &expected_pointmap=out.taps.at("22.pointmap");
        if(out.pointmap.size()!=expected_pointmap.size())throw std::runtime_error("incorrect returned XYZ extent");
        for(size_t j=0;j<out.pointmap.size();++j)if(!(out.pointmap[j]==expected_pointmap[j] || (std::isnan(out.pointmap[j]) && std::isnan(expected_pointmap[j]))))throw std::runtime_error("returned XYZ differs from observed output");
        auto compact=sam3d::objects_prepare_pointmap_joint(rgba,w,h,stride,xyz,ph,pw,factor,pad);
        if(!compact.taps.empty() || compact.rgb!=out.rgb || compact.mask!=out.mask || compact.pointmap.size()!=out.pointmap.size())throw std::runtime_error("joint observer affected output");
        for(size_t j=0;j<out.pointmap.size();++j)if(!(compact.pointmap[j]==out.pointmap[j] || (std::isnan(compact.pointmap[j]) && std::isnan(out.pointmap[j]))))throw std::runtime_error("joint observer affected XYZ");
    }
    in>>std::ws;if(!in.eof() || !soft)throw std::runtime_error("invalid fixture coverage");
    auto reject=[](auto fn){bool caught=false;try{fn();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid joint input accepted");};
    uint32_t w=9,h=7,ph=5,pw=3;uint64_t stride=36;double factor=1,pad=.1;std::vector<uint8_t> rgba(h*stride,255);std::vector<float> xyz(3*ph*pw,1);
    auto run=[&]{sam3d::objects_prepare_pointmap_joint(rgba,w,h,stride,xyz,ph,pw,factor,pad);};run();
    w=0;reject(run);w=4097;reject(run);w=9;ph=0;reject(run);ph=4097;reject(run);ph=5;
    xyz.pop_back();reject(run);xyz.push_back(1);rgba.pop_back();reject(run);rgba.push_back(255);
    stride=35;reject(run);stride=UINT64_MAX;reject(run);stride=36;
    for(double v:{0.,.1,4.1,std::numeric_limits<double>::quiet_NaN()}){factor=v;reject(run);}factor=1;
    for(double v:{-.1,1.1,std::numeric_limits<double>::infinity()}){pad=v;reject(run);}pad=.1;
    std::fill(rgba.begin(),rgba.end(),0);reject(run);rgba[(3*w+4)*4+3]=255;reject(run);
    std::cout<<checked<<" original joint boundaries plus soft-mask/invalid-input checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
