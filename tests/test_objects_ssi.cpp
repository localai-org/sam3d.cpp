#include "objects_ssi.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected original SSI fixture");std::ifstream in(argv[1]);std::string magic;uint32_t count;
    if(!(in>>magic>>count) || magic!="S3D_SSI_REGRESSION_V1" || count!=15)throw std::runtime_error("bad SSI fixture header");
    auto values=[&](uint64_t n){std::vector<float> v(n);for(auto &x:v){std::string token;if(!(in>>token))throw std::runtime_error("truncated SSI fixture");
        auto [end,error]=std::from_chars(token.data(),token.data()+token.size(),x);
        if(error!=std::errc() || end!=token.data()+token.size())throw std::runtime_error("bad SSI fixture value");}return v;};
    uint32_t checked=0;
    for(uint32_t i=0;i<count;++i){
        sam3d::objects_ssi_shape s;sam3d::objects_ssi_options o;uint32_t mode,allow,raise,has_s,has_t;
        if(!(in>>s.height>>s.width>>s.mask_height>>s.mask_width>>mode>>allow>>raise>>has_s>>has_t>>o.quantile_drop>>o.clip>>o.scale_factor>>o.log_disparity_shift) ||
            s.height>32 || s.width>32 || s.mask_height>32 || s.mask_width>32)throw std::runtime_error("bad SSI fixture options");
        o.mode=sam3d::objects_ssi_mode(mode);o.allow_override=allow;o.raise_on_no_valid_points=raise;sam3d::validate_objects_ssi(s,o);
        auto so=values(3),to=values(3),xyz=values(uint64_t(3)*s.height*s.width),mask=values(uint64_t(s.mask_height)*s.mask_width);
        auto out=sam3d::objects_normalize_pointmap(s,o,xyz,mask,has_s?std::span<const float>(so):std::span<const float>(),has_t?std::span<const float>(to):std::span<const float>(),true);
        auto back=sam3d::objects_denormalize_pointmap(s.height,s.width,o.mode,out.pointmap,out.scale,out.shift,&out.taps);
        uint32_t tap_count;if(!(in>>tap_count) || tap_count!=out.taps.size())throw std::runtime_error("bad SSI fixture tap count");
        for(auto &[key,v]:out.taps){std::string name;uint64_t n;if(!(in>>name>>n) || name!=key || n!=v.size())throw std::runtime_error("SSI tap name/shape mismatch");
            auto ref=values(n);double maximum=0,error2=0,ref2=0;
            for(size_t j=0;j<n;++j){auto x=v[j],y=ref[j];
                if(std::isnan(x)!=std::isnan(y) || std::isinf(x)!=std::isinf(y) || (std::isinf(x) && std::signbit(x)!=std::signbit(y)))throw std::runtime_error("SSI nonfinite category mismatch "+key);
                if(!std::isfinite(y))continue;
                double delta=double(x)-y;maximum=std::max(maximum,std::abs(delta));error2+=delta*delta;ref2+=double(y)*y;
            }
            if(maximum>1e-4 || std::sqrt(error2)/std::max(std::sqrt(ref2),1e-12)>2e-5 || (key=="01.mask" && v!=ref))throw std::runtime_error("SSI original mismatch "+key);
            ++checked;
        }
        auto compact=sam3d::objects_normalize_pointmap(s,o,xyz,mask,has_s?std::span<const float>(so):std::span<const float>(),has_t?std::span<const float>(to):std::span<const float>());
        if(!compact.taps.empty() || compact.scale!=out.scale || compact.shift!=out.shift)throw std::runtime_error("SSI observation altered parameters");
        for(size_t j=0;j<out.pointmap.size();++j)if(!(compact.pointmap[j]==out.pointmap[j] || (std::isnan(compact.pointmap[j]) && std::isnan(out.pointmap[j]))))throw std::runtime_error("SSI observation altered output");
    }
    in>>std::ws;if(!in.eof())throw std::runtime_error("trailing SSI fixture values");
    auto reject=[](auto run){bool caught=false;try{run();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid SSI input accepted");};
    sam3d::objects_ssi_shape s{5,7,5,7};sam3d::objects_ssi_options o;std::vector<float> xyz(3*35,1),mask(35,1);std::array<float,3> scale{1,1,1},shift{0,0,0};
    auto run=[&]{sam3d::objects_normalize_pointmap(s,o,xyz,mask,scale,shift);};run();
    s.height=0;reject(run);s.height=2049;reject(run);s.height=5;xyz.pop_back();reject(run);xyz.push_back(1);
    mask.pop_back();reject(run);mask.push_back(1);mask[0]=std::numeric_limits<float>::quiet_NaN();reject(run);mask[0]=1.1f;reject(run);mask[0]=1;
    o.mode=sam3d::objects_ssi_mode(8);reject(run);o.mode=sam3d::objects_ssi_mode::basic;
    o.quantile_drop=.5;reject(run);o.quantile_drop=-.1;reject(run);o.quantile_drop=.1;
    o.clip=-1;reject(run);o.clip=0;o.scale_factor=0;reject(run);o.scale_factor=1;
    for(float invalid:{0.f,-1.f,std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){scale[0]=invalid;reject(run);}scale[0]=1;
    scale[0]=1e-30f;shift[0]=1e30f;reject(run);scale[0]=1;shift[0]=0;
    xyz[0]=std::numeric_limits<float>::quiet_NaN();auto invalid=sam3d::objects_normalize_pointmap(s,o,xyz,mask,scale,shift);
    if(!std::isnan(invalid.pointmap[0]) || !std::isnan(invalid.pointmap[35]) || !std::isnan(invalid.pointmap[70]))throw std::runtime_error("homogeneous invalidity was lost");
    o.mode=sam3d::objects_ssi_mode::object_scene;std::fill(mask.begin(),mask.end(),0);reject(run);std::fill(mask.begin(),mask.end(),1);
    std::fill(xyz.begin(),xyz.end(),std::numeric_limits<float>::quiet_NaN());o.raise_on_no_valid_points=true;reject(run);
    o.raise_on_no_valid_points=false;run();
    // The official RGB mask is 4096x2160, larger than its supplied pointmap.
    // Uniform foreground must resize to the same mask without shrinking RGB.
    std::fill(xyz.begin(),xyz.end(),1);s.mask_height=2160;s.mask_width=4096;
    mask.assign(size_t(s.mask_height)*s.mask_width,1);o.mode=sam3d::objects_ssi_mode::apparent_object;
    auto large_mask=sam3d::objects_normalize_pointmap(s,o,xyz,mask);
    if(large_mask.pointmap!=xyz || large_mask.scale!=scale || large_mask.shift!=shift)throw std::runtime_error("full-resolution mask normalization failed");
    s.mask_width=4097;reject(run);
    std::cout<<checked<<" original SSI boundaries plus invalid-input/homogeneous/observer checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
