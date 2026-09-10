// Copyright (c) Meta Platforms, Inc. and affiliates.
// Modified: checked C++ SSI normalizers and homogeneous transforms.
// SAM License and PyTorch3D BSD terms: see THIRD_PARTY_NOTICES.md.
#include "objects_ssi.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace sam3d {
namespace {
constexpr float nan=std::numeric_limits<float>::quiet_NaN();
void require(bool b,const char *m){if(!b)throw std::invalid_argument(m);}
void put(objects_image_taps *t,const std::string &key,std::span<const float> v){if(t)t->emplace(key,std::vector<float>(v.begin(),v.end()));}
void vector3(std::span<const float> v,bool positive){
    require(v.size()==3,"SSI scale/shift must have three elements");
    for(float x:v)require(std::isfinite(x) && (!positive || (x>0 && std::isfinite(1.f/x))),"invalid SSI scale/shift");
}
float median(std::vector<float> v){
    std::erase_if(v,[](float x){return std::isnan(x);});if(v.empty())return nan;
    auto at=(v.size()-1)/2;std::nth_element(v.begin(),v.begin()+at,v.end());return v[at];
}
float radius(float x,float y,float z){return std::sqrt((x*x+y*y)+z*z);}
std::vector<float> transformed(std::span<const float> xyz,std::span<const float> scale,
    std::span<const float> shift,bool inverse,objects_image_taps *t){
    vector3(scale,true);vector3(shift,false);const auto n=xyz.size()/3;
    std::array<float,16> matrix{};matrix[15]=1;
    for(size_t c=0;c<3;++c){matrix[c*4+c]=inverse?scale[c]:1.f/scale[c];matrix[12+c]=inverse?shift[c]:(-shift[c])*matrix[c*4+c];}
    for(float x:matrix)require(std::isfinite(x),"overflow in SSI transform matrix");
    const auto base=inverse?30:10;put(t,std::to_string(base)+".matrix",matrix);
    std::vector<float> homogeneous(n*4),raw(n*4),result(n*3);
    for(size_t p=0;p<n;++p){
        for(size_t c=0;c<3;++c)homogeneous[p*4+c]=xyz[c*n+p];
        homogeneous[p*4+3]=1;
        // Preserve 0*NaN/Inf, homogeneous denominator and sum operation order.
        // Simplifying this to independent (x-shift)/scale changes invalidity.
        for(size_t c=0;c<4;++c){float sum=0;for(size_t k=0;k<4;++k)sum+=homogeneous[p*4+k]*matrix[k*4+c];raw[p*4+c]=sum;}
        for(size_t c=0;c<3;++c)result[c*n+p]=raw[p*4+c]/raw[p*4+3];
    }
    put(t,std::to_string(base+1)+".homogeneous",homogeneous);put(t,std::to_string(base+2)+".transformed",raw);
    put(t,inverse?"33.metric_space":"13.unclipped",result);return result;
}
}
void validate_objects_ssi(objects_ssi_shape s,objects_ssi_options o){
    require(s.height>=1 && s.height<=2048 && s.width>=1 && s.width<=2048 &&
        s.mask_height>=1 && s.mask_height<=4096 && s.mask_width>=1 && s.mask_width<=4096,"invalid SSI shape");
    require(uint32_t(o.mode)<=7,"invalid SSI normalizer");
    require(std::isfinite(o.quantile_drop) && o.quantile_drop>=0 && o.quantile_drop<.5,"invalid SSI quantile range");
    require(std::isfinite(o.clip) && o.clip>=0 && o.clip<=1e6 &&
        std::isfinite(o.scale_factor) && o.scale_factor>0 && o.scale_factor<=1e6 &&
        std::isfinite(o.log_disparity_shift) && std::abs(o.log_disparity_shift)<=1e6,"invalid SSI options");
}
objects_ssi_result objects_normalize_pointmap(objects_ssi_shape s,objects_ssi_options o,
    std::span<const float> xyz,std::span<const float> mask,std::span<const float> so,std::span<const float> to,bool observe){
    validate_objects_ssi(s,o);uint64_t n=uint64_t(s.height)*s.width;
    require(xyz.size()==3*n && mask.size()==uint64_t(s.mask_height)*s.mask_width,"SSI input extent mismatch");
    for(float x:mask)require(std::isfinite(x) && x>=0 && x<=1,"invalid SSI mask");
    if(!so.empty())vector3(so,true);
    if(!to.empty())vector3(to,false);
    objects_ssi_result result;auto taps=observe?&result.taps:nullptr;auto &scale=result.scale;auto &shift=result.shift;
    const auto mode=uint32_t(o.mode);std::vector<float> points(xyz.begin(),xyz.end());
    if(mode>=6)for(size_t i=0;i<n;++i){points[i]=xyz[i]/xyz[2*n+i];points[n+i]=xyz[n+i]/xyz[2*n+i];points[2*n+i]=std::log(xyz[2*n+i]);}
    put(taps,"00.remapped",points);
    const bool compute=(mode>=1 && mode<=3) || so.empty() || to.empty();
    if(compute){
        std::vector<size_t> selected;std::vector<float> resized;
        const bool use_mask=(mode>=1 && mode<=3) || mode==5 || mode>=6;
        if(use_mask){
            resized.resize(n);
            for(uint32_t y=0;y<s.height;++y)for(uint32_t x=0;x<s.width;++x){
                auto my=std::min(uint32_t(std::floor(y*(float(s.mask_height)/s.height))),s.mask_height-1);
                auto mx=std::min(uint32_t(std::floor(x*(float(s.mask_width)/s.width))),s.mask_width-1);
                float v=mask[uint64_t(my)*s.mask_width+mx];resized[uint64_t(y)*s.width+x]=v;
                if(v>.5f)selected.push_back(uint64_t(y)*s.width+x);
            }
            put(taps,"01.mask",resized);
        }
        auto all_axis=[&](uint32_t c){return std::vector<float>(points.begin()+c*n,points.begin()+(c+1)*n);};
        auto mask_axis=[&](uint32_t c){std::vector<float> v;v.reserve(selected.size());for(auto i:selected)v.push_back(points[c*n+i]);return v;};
        if(mode==0){
            shift={0,0,median(all_axis(2))};std::vector<float> centered(points);double sum=0;uint64_t count=0;
            for(size_t c=0;c<3;++c)for(size_t i=0;i<n;++i){float v=points[c*n+i]-shift[c];centered[c*n+i]=v;if(!std::isnan(v)){sum+=std::abs(v);++count;}}
            scale.fill(count?float(sum/count):nan);put(taps,"02.centered",centered);
        }else if(mode<=3){
            require(!selected.empty(),"empty SSI object mask");bool any_finite=false;
            for(auto i:selected)for(size_t c=0;c<3;++c)any_finite=any_finite || std::isfinite(points[c*n+i]);
            if(!any_finite){require(!o.raise_on_no_valid_points,"no valid SSI object points");scale.fill(float(o.scale_factor));shift.fill(0);}
            else{
                for(size_t c=0;c<3;++c)shift[c]=median(mask_axis(c));
                const auto k=mode==1?n:selected.size();std::vector<float> centered(k*3),statistic(k);
                for(size_t c=0;c<3;++c)for(size_t i=0;i<k;++i)centered[c*k+i]=points[c*n+(mode==1?i:selected[i])]-shift[c];
                for(size_t i=0;i<k;++i){float x=centered[i],y=centered[k+i],z=centered[2*k+i];
                    statistic[i]=mode==1?((std::isnan(x)||std::isnan(y)||std::isnan(z))?nan:std::max({std::abs(x),std::abs(y),std::abs(z)})):radius(x,y,z);
                }
                float value;
                if(mode==2){
                    auto sorted=statistic;std::erase_if(sorted,[](float x){return std::isnan(x);});require(!sorted.empty(),"no valid SSI object radii");std::sort(sorted.begin(),sorted.end());
                    auto quantile=[&](float q){float at=q*float(sorted.size()-1);auto low=size_t(std::floor(at)),high=size_t(std::ceil(at));float weight=at-float(low),a=sorted[low],b=sorted[high];
                        return weight<.5f?a+weight*(b-a):b-(b-a)*(1.f-weight);};
                    std::array<float,2> quantiles{quantile(float(o.quantile_drop)),quantile(float(1.-o.quantile_drop))};put(taps,"04.quantiles",quantiles);value=(quantiles[1]-quantiles[0])*2.f;
                }else value=median(statistic);
                scale.fill(value*float(o.scale_factor));put(taps,"02.centered",centered);put(taps,"03.statistic",statistic);
            }
        }else if(mode<=5){float z=median(mode==4?all_axis(2):mask_axis(2));scale.fill(z*float(o.scale_factor));shift.fill(0);}
        else{scale.fill(1);if(mode==6)shift={0,0,median(all_axis(2))+float(o.log_disparity_shift)};
            else for(size_t c=0;c<3;++c)shift[c]=median(mask_axis(c));}
        put(taps,"05.computed_scale",scale);put(taps,"06.computed_shift",shift);
    }
    if((mode>=1 && mode<=3)?o.allow_override:(!so.empty() && !to.empty())){
        if(!so.empty())std::copy(so.begin(),so.end(),scale.begin());
        if(!to.empty())std::copy(to.begin(),to.end(),shift.begin());
    }
    vector3(scale,true);vector3(shift,false);result.pointmap=transformed(points,scale,shift,false,taps);
    if(mode!=0 && o.clip>0)for(size_t i=0;i<n;++i){float x=result.pointmap[i],y=result.pointmap[n+i],z=result.pointmap[2*n+i];
        float v=mode<=3?radius(x,y,z):mode<=5?z:std::abs(z);if(v>float(o.clip))for(size_t c=0;c<3;++c)result.pointmap[c*n+i]=nan;
    }
    for(float x:result.pointmap)require(!std::isinf(x),"overflow in SSI normalization");
    put(taps,"20.normalized",result.pointmap);put(taps,"21.scale",scale);put(taps,"22.shift",shift);return result;
}
std::vector<float> objects_denormalize_pointmap(uint32_t h,uint32_t w,objects_ssi_mode mode,
    std::span<const float> xyz,std::span<const float> scale,std::span<const float> shift,objects_image_taps *taps){
    validate_objects_ssi({h,w,1,1},{mode});require(xyz.size()==uint64_t(3)*h*w,"SSI denormalization extent mismatch");
    auto result=transformed(xyz,scale,shift,true,taps);const auto n=uint64_t(h)*w;
    if(uint32_t(mode)>=6)for(size_t i=0;i<n;++i){float z=std::exp(result[2*n+i]);result[i]*=z;result[n+i]*=z;result[2*n+i]=z;}
    for(float x:result)require(!std::isinf(x),"overflow in SSI denormalization");
    put(taps,"40.denormalized",result);return result;
}
}
