// Copyright (c) Meta Platforms, Inc. and affiliates.
// Objects image/mask transforms: SAM license; NOTICE.
// Antialiased bicubic conventions follow PyTorch v2.7.0: LICENSES/PyTorch.txt.
#include "objects_image.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace sam3d {
namespace {
void require(bool ok,const char *what){if(!ok)throw std::invalid_argument(what);}
constexpr uint64_t max_pixels=16*1024*1024;
struct tensor {
    uint32_t c,h,w;std::vector<float> v;
    tensor(uint32_t channels,uint32_t height,uint32_t width):c(channels),h(height),w(width){
        require(c>=1 && c<=4 && h>=1 && w>=1 && uint64_t(h)*w<=max_pixels,"Objects image allocation exceeds limit");v.resize(uint64_t(c)*h*w);
    }
    float &at(uint32_t z,uint32_t y,uint32_t x){return v[(uint64_t(z)*h+y)*w+x];}
    float at(uint32_t z,uint32_t y,uint32_t x) const{return v[(uint64_t(z)*h+y)*w+x];}
};
tensor pad(const tensor &in,uint32_t left,uint32_t right,uint32_t top,uint32_t bottom,float value=0){
    tensor out(in.c,in.h+top+bottom,in.w+left+right);
    if(value!=0)std::fill(out.v.begin(),out.v.end(),value);
    for(uint32_t z=0;z<in.c;++z)for(uint32_t y=0;y<in.h;++y)
        std::copy_n(in.v.begin()+(uint64_t(z)*in.h+y)*in.w,in.w,out.v.begin()+(uint64_t(z)*out.h+y+top)*out.w+left);
    return out;
}
tensor square(const tensor &in,float value=0){auto side=std::max(in.h,in.w);auto dy=side-in.h,dx=side-in.w;return pad(in,dx/2,dx-dx/2,dy/2,dy-dy/2,value);}
struct filter{uint32_t first;std::vector<float> weights;};
float cubic(float x){
    x=std::abs(x);constexpr float a=-0.5f;
    if(x<1.f)return ((a+2.f)*x-(a+3.f))*x*x+1.f;
    if(x<2.f)return ((a*x-5.f*a)*x+8.f*a)*x-4.f*a;
    return 0.f;
}
std::vector<filter> filters(uint32_t input,uint32_t output,bool linear=false){
    std::vector<filter> out(output);const float scale=float(input)/output;
    const float radius=linear?1.f:2.f;
    const float support=scale>=1.f?radius*scale:radius;
    const float invscale=scale>=1.f?float(1.0/scale):1.f;
    const auto max_size=int64_t(std::ceil(support))*2+1;
    for(uint32_t i=0;i<output;++i){
        auto &f=out[i];const float center=float(scale*(i+0.5));
        f.first=uint32_t(std::max(int64_t(center-support+0.5),int64_t(0)));
        auto count=std::clamp(std::min(int64_t(center+support+0.5),int64_t(input))-f.first,int64_t(0),max_size);
        require(count>0,"empty Objects resize filter");float sum=0;
        for(int64_t j=0;j<count;++j){float x=float((float(j+f.first)-center+0.5)*invscale);float w=linear?std::max(0.f,1.f-std::abs(x)):cubic(x);f.weights.push_back(w);sum+=w;}
        require(sum!=0 && std::isfinite(sum),"invalid Objects resize filter");for(auto &w:f.weights)w/=sum;
    }
    return out;
}
tensor bicubic(const tensor &in,uint32_t height,uint32_t width){
    if(in.h==height && in.w==width)return in;
    auto fx=filters(in.w,width),fy=filters(in.h,height);tensor tmp(in.c,in.h,width),out(in.c,height,width);
    for(uint32_t z=0;z<in.c;++z)for(uint32_t y=0;y<in.h;++y)for(uint32_t x=0;x<width;++x){
        const auto &f=fx[x];float v=in.at(z,y,f.first)*f.weights[0];
        for(uint32_t j=1;j<f.weights.size();++j)v+=in.at(z,y,f.first+j)*f.weights[j];
        tmp.at(z,y,x)=v;
    }
    for(uint32_t z=0;z<in.c;++z)for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x){
        const auto &f=fy[y];float v=tmp.at(z,f.first,x)*f.weights[0];
        for(uint32_t j=1;j<f.weights.size();++j)v+=tmp.at(z,f.first+j,x)*f.weights[j];
        out.at(z,y,x)=v;
    }
    return out; // Original float bicubic intentionally does NOT clamp overshoot.
}
tensor nearest(const tensor &in,uint32_t height,uint32_t width=0){
    if(!width)width=height;
    tensor out(in.c,height,width);const float sx=float(in.w)/width,sy=float(in.h)/height;
    for(uint32_t z=0;z<in.c;++z)for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x)
        out.at(z,y,x)=in.at(z,std::min(uint32_t(std::floor(y*sy)),in.h-1),std::min(uint32_t(std::floor(x*sx)),in.w-1));
    return out;
}
tensor bilinear(const tensor &in,uint32_t height,uint32_t width){
    if(in.h==height && in.w==width)return in;
    auto fx=filters(in.w,width,true),fy=filters(in.h,height,true);tensor tmp(in.c,in.h,width),out(in.c,height,width);
    for(uint32_t c=0;c<in.c;++c)for(uint32_t y=0;y<in.h;++y)for(uint32_t x=0;x<width;++x){
        if(in.w==width){tmp.at(c,y,x)=in.at(c,y,x);continue;}
        auto &f=fx[x];float v=in.at(c,y,f.first)*f.weights[0];
        for(uint32_t j=1;j<f.weights.size();++j)v+=in.at(c,y,f.first+j)*f.weights[j];
        tmp.at(c,y,x)=v;
    }
    for(uint32_t c=0;c<in.c;++c)for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x){
        if(in.h==height){out.at(c,y,x)=tmp.at(c,y,x);continue;}
        auto &f=fy[y];float v=tmp.at(c,f.first,x)*f.weights[0];
        for(uint32_t j=1;j<f.weights.size();++j)v+=tmp.at(c,f.first+j,x)*f.weights[j];
        out.at(c,y,x)=v;
    }
    return out;
}
}
void validate_objects_image_options(objects_image_options s){
    require(s.output_side>=1 && s.output_side<=1024 && std::isfinite(s.box_size_factor) && s.box_size_factor>=.25 && s.box_size_factor<=4 && std::isfinite(s.padding_factor) && s.padding_factor>=0 && s.padding_factor<=1,"invalid Objects preprocessing options");
}
objects_image_taps objects_prepare_rgba(std::span<const uint8_t> rgba,uint32_t width,uint32_t height,uint64_t stride,objects_image_options s){
    require(width>=1 && height>=1 && width<=4096 && height<=4096 && uint64_t(width)*height<=max_pixels,"invalid Objects image dimensions");
    require(stride>=uint64_t(width)*4 && stride<=uint64_t(width)*4+4096 && rgba.size()>=uint64_t(height-1)*stride+uint64_t(width)*4,"invalid Objects RGBA buffer/stride");
    validate_objects_image_options(s);
    tensor unit(4,height,width),rgb(3,height,width),mask(1,height,width);objects_image_taps out;
    int32_t minx=int32_t(width),miny=int32_t(height),maxx=-1,maxy=-1;
    for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x){
        for(uint32_t z=0;z<4;++z){auto value=float(double(rgba[y*stride+x*4+z])/255.0);unit.at(z,y,x)=value;if(z<3)rgb.at(z,y,x)=value;}
        if(rgba[y*stride+x*4+3]){mask.at(0,y,x)=1;minx=std::min(minx,int32_t(x));miny=std::min(miny,int32_t(y));maxx=std::max(maxx,int32_t(x));maxy=std::max(maxy,int32_t(y));}
    }
    // Upstream computes max-min, NOT max-min+1, and rejects spans below two.
    require(maxx>=0 && maxy>=0 && maxx-minx>=2 && maxy-miny>=2,"Objects mask is empty or spans fewer than three pixels on an axis");
    const double cx=(minx+maxx)/2.0,cy=(miny+maxy)/2.0;
    const int32_t size=int32_t(std::max({maxx-minx,maxy-miny,2})*s.box_size_factor),half=size/2;
    const int32_t x0=int32_t(cx-half),y0=int32_t(cy-half),x1=int32_t(cx+half),y1=int32_t(cy+half);
    require(x1>x0 && y1>y0,"Objects crop has no area");
    out["00.rgba"]=std::move(unit.v);out["01.binary_mask"]=mask.v;out["02.bbox"]={float(x0),float(y0),float(x1),float(y1)};
    tensor cr(3,uint32_t(y1-y0),uint32_t(x1-x0)),cm(1,cr.h,cr.w);
    for(uint32_t y=0;y<cr.h;++y)for(uint32_t x=0;x<cr.w;++x){
        int32_t iy=y0+int32_t(y),ix=x0+int32_t(x);
        if(ix>=0 && iy>=0 && uint32_t(ix)<width && uint32_t(iy)<height){
            for(uint32_t z=0;z<3;++z)cr.at(z,y,x)=rgb.at(z,iy,ix);
            cm.at(0,y,x)=mask.at(0,iy,ix);
        }
    }
    cr=square(cr);cm=square(cm);auto extend=uint32_t(cr.h*s.padding_factor);
    cr=pad(cr,extend,extend,extend,extend);cm=pad(cm,extend,extend,extend,extend);
    out["03.crop_rgb"]=cr.v;out["04.crop_mask"]=cm.v;
    for(uint32_t z=0;z<3;++z)for(uint32_t y=0;y<cr.h;++y)for(uint32_t x=0;x<cr.w;++x)cr.at(z,y,x)*=cm.at(0,y,x);
    out["05.masked_rgb"]=cr.v;out["06.masked_mask"]=cm.v;
    auto pr=square(cr),pm=square(cm),fr=square(rgb),fm=square(mask);
    out["07.pad.0"]=pr.v;out["07.pad.1"]=pm.v;out["07.pad.2"]=fr.v;out["07.pad.3"]=fm.v;
    out["08.image"]=bicubic(pr,s.output_side,s.output_side).v;out["08.mask"]=nearest(pm,s.output_side).v;
    out["08.rgb_image"]=bicubic(fr,s.output_side,s.output_side).v;out["08.rgb_image_mask"]=nearest(fm,s.output_side).v;
    return out;
}
objects_joint_result objects_prepare_pointmap_joint(std::span<const uint8_t> rgba,
    uint32_t width,uint32_t height,uint64_t stride,std::span<const float> xyz,
    uint32_t ph,uint32_t pw,double factor,double padding,bool observe){
    require(width>=1 && height>=1 && width<=4096 && height<=4096 && uint64_t(width)*height<=max_pixels &&
        ph>=1 && pw>=1 && ph<=4096 && pw<=4096 && uint64_t(ph)*pw<=max_pixels,"invalid joint image/pointmap dimensions");
    require(stride>=uint64_t(width)*4 && stride<=uint64_t(width)*4+4096 && rgba.size()>=uint64_t(height-1)*stride+width*4 && xyz.size()==uint64_t(3)*ph*pw,"invalid joint image/pointmap extents");
    validate_objects_image_options({1,factor,padding});objects_joint_result result{};
    auto keep=[&](const std::string &key,std::span<const float> v){if(observe)result.taps.emplace(key,std::vector<float>(v.begin(),v.end()));};
    tensor unit(4,height,width),rgb(3,height,width),mask(1,height,width),point(3,ph,pw);
    point.v.assign(xyz.begin(),xyz.end());int32_t minx=width,miny=height,maxx=-1,maxy=-1;
    for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x){
        for(uint32_t c=0;c<4;++c){float v=float(double(rgba[y*stride+x*4+c])/255.);unit.at(c,y,x)=v;if(c<3)rgb.at(c,y,x)=v;else mask.at(0,y,x)=v;}
        if(mask.at(0,y,x)>0){minx=std::min(minx,int32_t(x));miny=std::min(miny,int32_t(y));maxx=std::max(maxx,int32_t(x));maxy=std::max(maxy,int32_t(y));}
    }
    keep("00.rgba",unit.v);keep("01.alpha",mask.v);
    if(ph!=height || pw!=width){
        tensor invalid(1,ph,pw),clean=point;
        for(uint32_t y=0;y<ph;++y)for(uint32_t x=0;x<pw;++x)for(uint32_t c=0;c<3;++c)if(std::isnan(point.at(c,y,x))){invalid.at(0,y,x)=1;clean.at(c,y,x)=0;}
        keep("05.nan_mask",invalid.v);keep("06.clean",clean.v);
        point=bilinear(clean,height,width);keep("07.resized",point.v);invalid=nearest(invalid,height,width);keep("08.resized_nan_mask",invalid.v);
        for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x)if(invalid.at(0,y,x)>.5f)for(uint32_t c=0;c<3;++c)point.at(c,y,x)=std::numeric_limits<float>::quiet_NaN();
    }
    keep("02.aligned_rgb",rgb.v);keep("03.aligned_mask",mask.v);keep("04.aligned_pointmap",point.v);
    require(maxx>=0 && maxx-minx>=2 && maxy-miny>=2,"joint mask is empty or degenerate");
    const double cx=(minx+maxx)/2.,cy=(miny+maxy)/2.;const int32_t size=int32_t(std::max({maxx-minx,maxy-miny,2})*factor),half=size/2;
    const int32_t x0=int32_t(cx-half),y0=int32_t(cy-half),x1=int32_t(cx+half),y1=int32_t(cy+half);
    require(x1>x0 && y1>y0,"joint crop has no area");std::array<float,4> box{float(x0),float(y0),float(x1),float(y1)};keep("09.bbox",box);
    auto crop=[&](const tensor &in){tensor out(in.c,y1-y0,x1-x0);
        for(uint32_t c=0;c<in.c;++c)for(uint32_t y=0;y<out.h;++y)for(uint32_t x=0;x<out.w;++x){int32_t iy=y0+int32_t(y),ix=x0+int32_t(x);if(iy>=0 && ix>=0 && uint32_t(iy)<in.h && uint32_t(ix)<in.w)out.at(c,y,x)=in.at(c,iy,ix);}
        return out;
    };
    const float nan=std::numeric_limits<float>::quiet_NaN();rgb=square(crop(rgb));mask=square(crop(mask));point=square(crop(point),nan);
    auto extend=uint32_t(rgb.h*padding);rgb=pad(rgb,extend,extend,extend,extend);mask=pad(mask,extend,extend,extend,extend);point=pad(point,extend,extend,extend,extend,nan);
    keep("10.crop_rgb",rgb.v);keep("11.crop_mask",mask.v);keep("12.crop_pointmap",point.v);
    for(uint32_t c=0;c<3;++c)for(uint32_t y=0;y<rgb.h;++y)for(uint32_t x=0;x<rgb.w;++x){float m=mask.at(0,y,x);rgb.at(c,y,x)*=m;if(!(m>0))point.at(c,y,x)=nan;}
    keep("20.rgb",rgb.v);keep("21.mask",mask.v);keep("22.pointmap",point.v);
    result.height=rgb.h;result.width=rgb.w;result.rgb=std::move(rgb.v);result.mask=std::move(mask.v);result.pointmap=std::move(point.v);return result;
}
std::vector<float> objects_square_resize(std::span<const float> data,uint32_t channels,
    uint32_t height,uint32_t width,uint32_t side,bool smooth,bool nan_padding,const objects_tensor_observer &observe){
    require(channels>=1 && channels<=4 && height>=1 && height<=4096 && width>=1 && width<=4096 && side>=1 && side<=1024 && data.size()==uint64_t(channels)*height*width,"invalid square resize shape");
    require(!smooth || !nan_padding,"bicubic NaN padding is not a supported image transform");
    tensor value(channels,height,width);value.v.assign(data.begin(),data.end());
    value=square(value,nan_padding?std::numeric_limits<float>::quiet_NaN():0);
    if(observe)observe("pad",value.v);
    auto result=smooth?bicubic(value,side,side):nearest(value,side);
    if(observe)observe("output",result.v);
    return std::move(result.v);
}
std::vector<float> objects_resize_chw(std::span<const float> data,uint32_t channels,
    uint32_t height,uint32_t width,uint32_t output_height,uint32_t output_width,
    bool smooth){
    require(channels>=1 && channels<=4 && height>=1 && height<=4096 && width>=1 && width<=4096 &&
        output_height>=1 && output_height<=4096 && output_width>=1 && output_width<=4096 &&
        uint64_t(output_height)*output_width<=max_pixels &&
        data.size()==uint64_t(channels)*height*width,"invalid direct resize shape");
    tensor value(channels,height,width);value.v.assign(data.begin(),data.end());
    auto result=smooth?bicubic(value,output_height,output_width):bilinear(value,output_height,output_width);
    return std::move(result.v);
}
}
