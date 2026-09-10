#include "objects_preprocess.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace {
bool equal(float a,float b){return a==b || (std::isnan(a) && std::isnan(b));}
bool equal(std::span<const float> a,std::span<const float> b){return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin(),[](float x,float y){return equal(x,y);});}
}
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected original preprocessing fixture");
    std::ifstream in(argv[1]);std::string magic;unsigned count;
    if(!(in>>magic>>count) || magic!="S3D_PREPROCESS_REGRESSION_V1" || count!=8)throw std::runtime_error("invalid preprocessing fixture");
    auto values=[&](size_t n){std::vector<float> v(n);for(auto &x:v){std::string t;if(!(in>>t))throw std::runtime_error("truncated fixture");auto [end,e]=std::from_chars(t.data(),t.data()+t.size(),x);if(e!=std::errc() || end!=t.data()+t.size())throw std::runtime_error("bad fixture value");}return v;};
    unsigned checks=0;
    for(unsigned i=0;i<count;++i){
        uint32_t w,h,stride,ph,pw,norm,nan;sam3d::objects_preprocess_options o;
        if(!(in>>w>>h>>stride>>ph>>pw>>o.image_side>>o.point_side>>norm>>nan>>o.box_factor>>o.padding) || !w || !h || w>32 || h>32 || !ph || !pw || ph>32 || pw>32 || stride>w*4+32 || norm>1 || nan>1)throw std::runtime_error("invalid fixture dimensions");
        o.normalize=norm;o.point_nan_padding=nan;
        for(auto *opt:{&o.object_normalizer,&o.full_normalizer}){
            unsigned mode,override,raise;
            if(!(in>>mode>>override>>raise>>opt->quantile_drop>>opt->clip>>opt->scale_factor>>opt->log_disparity_shift) || override>1 || raise>1)throw std::runtime_error("invalid fixture normalizer");
            opt->mode=sam3d::objects_ssi_mode(mode);opt->allow_override=override;opt->raise_on_no_valid_points=raise;
        }
        std::vector<uint8_t> rgba(size_t(h)*stride);for(auto &v:rgba){unsigned x;if(!(in>>x) || x>255)throw std::runtime_error("bad byte");v=uint8_t(x);}
        auto xyz=values(size_t(3)*ph*pw);sam3d::objects_image_taps taps;
        auto observed=sam3d::objects_preprocess_pointmap(rgba,w,h,stride,xyz,ph,pw,o,[&](const std::string &k,std::span<const float> v){if(!taps.emplace(k,std::vector<float>(v.begin(),v.end())).second)throw std::runtime_error("duplicate native tap");});
        auto compact=sam3d::objects_preprocess_pointmap(rgba,w,h,stride,xyz,ph,pw,o);
        if(observed.size()!=11 || compact.size()!=11)throw std::runtime_error("incomplete returned fields");
        for(auto &[key,v]:observed)if(!equal(v,compact.at(key)) || !equal(v,taps.at("06.return."+key)))throw std::runtime_error("observer or result mismatch");
        unsigned nt;if(!(in>>nt) || nt!=taps.size())throw std::runtime_error("tap count mismatch");
        for(auto &[key,v]:taps){std::string name;size_t n;if(!(in>>name>>n) || name!=key || n!=v.size())throw std::runtime_error("tap extent mismatch");auto ref=values(n);
            bool point=key.find("pointmap")!=std::string::npos || key.ends_with(".clean") || key.ends_with(".resized") || key.ends_with(".scale") || key.ends_with(".shift");
            for(auto index:{2,5,6})point|=key.starts_with("04.apply."+std::to_string(index)+".");
            bool exact=key.find("mask")!=std::string::npos || key.find("bbox")!=std::string::npos || key.starts_with("04.apply.1.") || key.starts_with("04.apply.4.");
            double mx=0,e2=0,r2=0;
            for(size_t j=0;j<n;++j){float x=v[j],y=ref[j];
                if(std::isnan(x)!=std::isnan(y) || std::isinf(x)!=std::isinf(y) || (std::isinf(x) && std::signbit(x)!=std::signbit(y)))throw std::runtime_error("nonfinite mismatch "+key);
                if(!std::isfinite(y)){if(!point || key.ends_with("scale") || key.ends_with("shift"))throw std::runtime_error("unexpected nonfinite "+key);continue;}
                double delta=double(x)-y;mx=std::max(mx,std::abs(delta));e2+=delta*delta;r2+=double(y)*y;
            }
            if(mx>(point?1e-4:exact?0:1e-6) || std::sqrt(e2)/std::max(std::sqrt(r2),1e-12)>(point?2e-5:exact?0:1e-6))throw std::runtime_error("original preprocessing mismatch "+key);
            ++checks;
        }
    }
    in>>std::ws;if(!in.eof())throw std::runtime_error("trailing fixture data");
    sam3d::objects_preprocess_options o;o.image_side=16;o.point_side=8;
    uint32_t w=9,h=7,ph=5,pw=3;uint64_t stride=36;std::vector<uint8_t> rgba(h*stride,255);std::vector<float> xyz(3*ph*pw,1);
    auto run=[&]{sam3d::objects_preprocess_pointmap(rgba,w,h,stride,xyz,ph,pw,o);};run();
    auto reject=[&]{bool caught=false;try{run();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid preprocessing input accepted");};
    for(uint32_t v:{0u,4097u,UINT32_MAX}){w=v;reject();}w=9;
    ph=0;reject();ph=2049;reject();ph=5;stride=35;reject();stride=UINT64_MAX;reject();stride=36;
    rgba.pop_back();reject();rgba.push_back(255);xyz.pop_back();reject();xyz.push_back(1);
    o.point_side=0;reject();o.point_side=1025;reject();o.point_side=8;
    o.image_side=0;reject();o.image_side=1025;reject();o.image_side=16;
    o.box_factor=std::numeric_limits<double>::quiet_NaN();reject();o.box_factor=1;
    o.padding=-1;reject();o.padding=.1;o.full_normalizer.mode=sam3d::objects_ssi_mode(8);reject();o.full_normalizer.mode=sam3d::objects_ssi_mode::basic;
    o.object_normalizer.scale_factor=0;reject();o.object_normalizer.scale_factor=1;
    std::fill(rgba.begin(),rgba.end(),0);reject();
    std::cout<<checks<<" original preprocessing boundaries plus own-result/observation/rejection checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
