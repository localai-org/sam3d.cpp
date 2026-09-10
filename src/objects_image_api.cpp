#include "error.hpp"
#include "objects_image.hpp"
#include <array>
#include <memory>
struct s3d_objects_image_request {sam3d::objects_image_options options;};
struct s3d_objects_image_result {uint32_t side;std::array<std::vector<float>,4> fields;};
namespace {
using s3d::require;using s3d::boundary;
template<class F> s3d_status change(s3d_objects_image_request *r,char *e,uint64_t n,F fn){
    return boundary(e,n,[&]{require(r,"request is required");auto s=r->options;fn(s);sam3d::validate_objects_image_options(s);r->options=s;});
}
}
s3d_status s3d_objects_image_request_create(s3d_objects_image_request **out,char *e,uint64_t n){
    if(out)*out=nullptr;
    return boundary(e,n,[&]{require(out,"output is required");*out=new s3d_objects_image_request;});
}
void s3d_objects_image_request_free(s3d_objects_image_request *r){delete r;}
s3d_status s3d_objects_image_request_set_output_side(s3d_objects_image_request *r,uint32_t side,char *e,uint64_t n){return change(r,e,n,[&](auto &s){s.output_side=side;});}
s3d_status s3d_objects_image_request_set_box_factor(s3d_objects_image_request *r,double factor,char *e,uint64_t n){return change(r,e,n,[&](auto &s){s.box_size_factor=factor;});}
s3d_status s3d_objects_image_request_set_padding(s3d_objects_image_request *r,double padding,char *e,uint64_t n){return change(r,e,n,[&](auto &s){s.padding_factor=padding;});}
s3d_status s3d_objects_image_request_get_options(const s3d_objects_image_request *r,uint32_t *side,double *factor,double *padding,char *e,uint64_t n){
    if(side)*side=0;
    if(factor)*factor=0;
    if(padding)*padding=0;
    return boundary(e,n,[&]{require(r && side && factor && padding,"request and outputs are required");*side=r->options.output_side;*factor=r->options.box_size_factor;*padding=r->options.padding_factor;});
}
s3d_status s3d_objects_image_prepare(const s3d_objects_image_request *r,const uint8_t *rgba,uint64_t capacity,uint32_t w,uint32_t h,uint64_t stride,s3d_objects_image_result **out,char *e,uint64_t n){
    if(out)*out=nullptr;
    return boundary(e,n,[&]{
        require(r && rgba && out,"request, pixels and output are required");
        require(w>=1 && w<=4096 && h>=1 && h<=4096 && stride>=uint64_t(w)*4 && stride<=uint64_t(w)*4+4096,"invalid Objects image dimensions/stride");
        const auto bytes=uint64_t(h-1)*stride+uint64_t(w)*4;
        require(bytes<=capacity && bytes<=std::numeric_limits<size_t>::max(),"Objects RGBA buffer is too small");
        auto values=sam3d::objects_prepare_rgba({rgba,size_t(bytes)},w,h,stride,r->options);
        auto result=std::make_unique<s3d_objects_image_result>();result->side=r->options.output_side;
        const char *names[]={"08.image","08.mask","08.rgb_image","08.rgb_image_mask"};
        for(uint32_t i=0;i<4;++i)result->fields[i]=std::move(values.at(names[i]));
        *out=result.release();
    });
}
void s3d_objects_image_result_free(s3d_objects_image_result *r){delete r;}
s3d_status s3d_objects_image_get_tensor(const s3d_objects_image_result *r,uint32_t field,const float **data,uint64_t *count,char *e,uint64_t n){
    if(data)*data=nullptr;
    if(count)*count=0;
    return boundary(e,n,[&]{require(r && data && count && field<4,"result, valid field and outputs are required");*data=r->fields[field].data();*count=r->fields[field].size();});
}
s3d_status s3d_objects_image_get_side(const s3d_objects_image_result *r,uint32_t *side,char *e,uint64_t n){
    if(side)*side=0;
    return boundary(e,n,[&]{require(r && side,"result and output are required");*side=r->side;});
}
