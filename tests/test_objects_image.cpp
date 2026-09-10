#include "objects_image.hpp"
#include "sam3d.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected original Objects fixture");
    std::ifstream in(argv[1]);std::string magic;uint32_t count;
    if(!(in>>magic>>count) || magic!="S3D_OBJECTS_IMAGE_REGRESSION_V1" || count!=5)throw std::runtime_error("invalid regression header");
    uint32_t checked=0;bool overshoot=false;
    for(uint32_t i=0;i<count;++i){
        uint32_t w,h,stride;sam3d::objects_image_options s;
        if(!(in>>w>>h>>stride>>s.output_side>>s.box_size_factor>>s.padding_factor) || w>64 || h>64 || stride>w*4+32)throw std::runtime_error("invalid test shape");
        std::vector<uint8_t> rgba(uint64_t(h)*stride);
        for(auto &x:rgba){unsigned byte;if(!(in>>byte) || byte>255)throw std::runtime_error("invalid test pixel");x=uint8_t(byte);}
        auto out=sam3d::objects_prepare_rgba(rgba,w,h,stride,s);std::set<std::string> seen;
        if(out.size()!=15)throw std::runtime_error("unexpected native tap set");
        for(uint32_t k=0;k<15;++k){
            std::string key;uint64_t n;if(!(in>>key>>n) || !out.contains(key) || !seen.insert(key).second || n!=out.at(key).size())throw std::runtime_error("invalid original tap shape/name");
            double maximum=0,delta2=0,ref2=0;bool exact=true;
            for(auto x:out.at(key)){
                float y;if(!(in>>y) || !std::isfinite(x) || !std::isfinite(y))throw std::runtime_error("invalid test value");
                double delta=double(x)-y;maximum=std::max(maximum,std::abs(delta));delta2+=delta*delta;ref2+=double(y)*y;exact=exact && x==y;
                if(key=="08.image" && (y<0 || y>1))overshoot=true;
            }
            const bool interpolated=key=="08.image" || key=="08.rgb_image";
            bool pass=interpolated?(maximum<=1e-6 && std::sqrt(delta2)/std::max(std::sqrt(ref2),1e-12)<=1e-6):exact;
            if(!pass)throw std::runtime_error("original Objects mismatch at case "+std::to_string(i)+" "+key+" absolute "+std::to_string(maximum));
            ++checked;
        }
        s3d_objects_image_request *request=nullptr;s3d_objects_image_result *result=nullptr;
        if(s3d_objects_image_request_create(&request,nullptr,0)!=S3D_OK ||
           s3d_objects_image_request_set_output_side(request,s.output_side,nullptr,0)!=S3D_OK ||
           s3d_objects_image_request_set_box_factor(request,s.box_size_factor,nullptr,0)!=S3D_OK ||
           s3d_objects_image_request_set_padding(request,s.padding_factor,nullptr,0)!=S3D_OK ||
           s3d_objects_image_prepare(request,rgba.data(),rgba.size(),w,h,stride,&result,nullptr,0)!=S3D_OK)
            throw std::runtime_error("C API failed original Objects case");
        s3d_objects_image_request_free(request);
        const char *names[]={"08.image","08.mask","08.rgb_image","08.rgb_image_mask"};
        for(uint32_t field=0;field<4;++field){const float *v=nullptr;uint64_t n=0;
            if(s3d_objects_image_get_tensor(result,field,&v,&n,nullptr,0)!=S3D_OK || n!=out.at(names[field]).size() ||
               !std::equal(v,v+n,out.at(names[field]).begin()))throw std::runtime_error("C API fields differ from verified native outputs");
        }
        s3d_objects_image_result_free(result);
    }
    in>>std::ws;if(!in.eof() || !overshoot)throw std::runtime_error("invalid fixture coverage/trailing values");
    auto reject=[](auto fn){bool caught=false;try{fn();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid image input accepted");};
    std::vector<uint8_t> rgba(9*13*4);uint32_t w=13,h=9;uint64_t stride=52;sam3d::objects_image_options s;
    auto run=[&]{return sam3d::objects_prepare_rgba(rgba,w,h,stride,s);};
    reject(run);rgba[(4*13+5)*4+3]=255;reject(run);
    for(uint32_t x=2;x<10;++x)rgba[(4*13+x)*4+3]=255;reject(run);
    std::fill(rgba.begin(),rgba.end(),0);for(uint32_t y=2;y<8;++y)rgba[(y*13+4)*4+3]=255;reject(run);
    std::fill(rgba.begin(),rgba.end(),255);s.output_side=16;run();
    auto original=rgba;rgba.resize(4);reject(run);rgba=original;
    w=0;reject(run);w=4097;reject(run);w=13;h=0;reject(run);h=9;
    stride=51;reject(run);stride=std::numeric_limits<uint64_t>::max();reject(run);stride=52;
    s.output_side=0;reject(run);s.output_side=1025;reject(run);s.output_side=16;
    for(double factor:{0.,.1,4.1,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){s.box_size_factor=factor;reject(run);}s.box_size_factor=1;
    for(double padding:{-.1,1.1,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){s.padding_factor=padding;reject(run);}s.padding_factor=.1;
    std::cout<<checked<<" original Objects preprocessing boundaries plus malformed-input/degenerate-mask checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
