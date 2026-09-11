#include "body_model.hpp"
#include "body_result.hpp"
#include "error.hpp"
#include <array>
#include <cmath>
#include <cstring>
struct s3d_runtime_options {sam3d::body_inference_options inference;std::array<std::string,3> files;std::string module,description;uint32_t backend=0,device=0,threads=1,precision=S3D_BACKBONE_F32;};
struct s3d_body_request {std::vector<uint8_t> rgb;uint32_t width=0,height=0;std::array<float,4> box{},intrinsics{};bool geometry=false;};
struct s3d_body_model {std::unique_ptr<sam3d::body_model> model;};
namespace {
using s3d::require;using s3d::boundary;
std::string copied_string(const char *p,uint64_t n,bool empty=false){
    require(n<=4096 && (n==0?empty:bool(p)),"string must contain 1..4096 bytes");
    require(!n || !std::memchr(p,0,size_t(n)),"embedded NUL in string");return n?std::string(p,size_t(n)):std::string{};
}
}
s3d_status s3d_runtime_options_create(s3d_runtime_options **out,char *e,uint64_t n){if(out)*out=nullptr;return boundary(e,n,[&]{require(out,"output required");*out=new s3d_runtime_options;});}
void s3d_runtime_options_free(s3d_runtime_options *o){delete o;}
s3d_status s3d_runtime_options_set_body_inference(s3d_runtime_options *o,uint32_t crop,uint32_t mask,uint32_t correctives,uint32_t slim,char *e,uint64_t n){
    return boundary(e,n,[&]{require(o && correctives<=1 && slim<=1,"valid options and boolean flags required");
        sam3d::body_inference_options v{crop,mask,bool(correctives),bool(slim)};
        sam3d::validate_body_inference_options(v);o->inference=v;});
}
namespace {
s3d_status get_inference(const sam3d::body_inference_options *o,uint32_t *crop,uint32_t *mask,uint32_t *correctives,uint32_t *slim,char *e,uint64_t n){
    if(crop)*crop=0;
    if(mask)*mask=0;
    if(correctives)*correctives=0;
    if(slim)*slim=0;
    return boundary(e,n,[&]{require(o && crop && mask && correctives && slim,"options/model and all outputs required");
        *crop=o->crop_size;*mask=o->intermediate_mask;*correctives=o->correctives;*slim=o->slim_intermediates;});
}
}
s3d_status s3d_runtime_options_get_body_inference(const s3d_runtime_options *o,uint32_t *crop,uint32_t *mask,uint32_t *correctives,uint32_t *slim,char *e,uint64_t n){
    return get_inference(o?&o->inference:nullptr,crop,mask,correctives,slim,e,n);
}
s3d_status s3d_body_model_get_body_inference(const s3d_body_model *m,uint32_t *crop,uint32_t *mask,uint32_t *correctives,uint32_t *slim,char *e,uint64_t n){
    auto o=m?m->model->inference_options():sam3d::body_inference_options{};
    return get_inference(m?&o:nullptr,crop,mask,correctives,slim,e,n);
}
s3d_status s3d_runtime_options_set_backbone_precision(s3d_runtime_options *o,uint32_t precision,char *e,uint64_t n){
    return boundary(e,n,[&]{require(o && (precision==S3D_BACKBONE_F32 || precision==S3D_BACKBONE_BF16),"valid options and backbone precision required");o->precision=precision;});
}
s3d_status s3d_runtime_options_get_backbone_precision(const s3d_runtime_options *o,uint32_t *precision,char *e,uint64_t n){
    if(precision)*precision=0;
    return boundary(e,n,[&]{require(o && precision,"options and precision output required");*precision=o->precision;});
}
s3d_status s3d_body_model_get_backbone_precision(const s3d_body_model *m,uint32_t *precision,char *e,uint64_t n){
    if(precision)*precision=0;
    return boundary(e,n,[&]{require(m && precision,"model and precision output required");*precision=m->model->bf16()?S3D_BACKBONE_BF16:S3D_BACKBONE_F32;});
}
s3d_status s3d_runtime_options_set_backend(s3d_runtime_options *o,uint32_t backend,const char *module,uint64_t bytes,uint32_t device,uint32_t threads,const char *description,uint64_t dbytes,char *e,uint64_t n){
    return boundary(e,n,[&]{require(o && (backend==S3D_BACKEND_CPU || backend==S3D_BACKEND_VULKAN) && threads>=1 && threads<=1024,"valid options, backend and 1..1024 threads required");
        auto m=copied_string(module,bytes),d=copied_string(description,dbytes,true);
        o->module=std::move(m);o->description=std::move(d);o->backend=backend;o->device=device;o->threads=threads;});
}
s3d_status s3d_runtime_options_set_body_file(s3d_runtime_options *o,uint32_t c,const char *path,uint64_t bytes,char *e,uint64_t n){
    return boundary(e,n,[&]{require(o && c<3,"valid options and component required");auto p=copied_string(path,bytes);o->files[c]=std::move(p);});
}
s3d_status s3d_runtime_options_get_body_file(const s3d_runtime_options *o,uint32_t c,const char **path,uint64_t *bytes,char *e,uint64_t n){
    if(path)*path=nullptr;
    if(bytes)*bytes=0;
    return boundary(e,n,[&]{require(o && c<3 && path && bytes,"valid options, component and outputs required");*path=o->files[c].c_str();*bytes=o->files[c].size();});
}
s3d_status s3d_body_model_load(const s3d_runtime_options *o,s3d_body_model **out,char *e,uint64_t n){
    if(out)*out=nullptr;
    return boundary(e,n,[&]{require(o && out && o->backend && !o->module.empty() && std::all_of(o->files.begin(),o->files.end(),[](auto &p){return !p.empty();}),"set backend and all three Body files first");
        auto result=std::make_unique<s3d_body_model>();result->model=std::make_unique<sam3d::body_model>(o->files[0],o->files[1],o->files[2],o->module,o->backend==S3D_BACKEND_CPU?"CPU":"Vulkan",o->device,o->threads,o->description,o->precision==S3D_BACKBONE_BF16,o->inference);*out=result.release();});
}
void s3d_body_model_free(s3d_body_model *m){delete m;}
s3d_status s3d_body_model_get_info(const s3d_body_model *m,const char **description,uint64_t *capabilities,char *e,uint64_t n){
    if(description)*description=nullptr;
    if(capabilities)*capabilities=0;
    return boundary(e,n,[&]{require(m && description && capabilities,"model and outputs required");*description=m->model->description().c_str();*capabilities=S3D_CAP_BODY_POSE_BRANCH;});
}
s3d_status s3d_body_request_create(s3d_body_request **out,char *e,uint64_t n){if(out)*out=nullptr;return boundary(e,n,[&]{require(out,"output required");*out=new s3d_body_request;});}
void s3d_body_request_free(s3d_body_request *r){delete r;}
s3d_status s3d_body_request_set_rgb(s3d_body_request *r,const uint8_t *rgb,uint64_t bytes,uint32_t width,uint32_t height,uint64_t stride,char *e,uint64_t n){
    return boundary(e,n,[&]{require(r && rgb && width && height && width<=32766 && height<=32766 && uint64_t(width)*height<=16000000,"valid request, RGB and bounded dimensions required");
        const uint64_t row=uint64_t(width)*3;
        require(bytes<=SIZE_MAX && stride>=row && bytes>=row && (height==1 || stride<=(bytes-row)/(height-1)),"RGB stride/capacity mismatch");
        std::vector<uint8_t> copy(row*height);
        for(uint32_t y=0;y<height;++y)std::copy_n(rgb+uint64_t(y)*stride,row,copy.begin()+uint64_t(y)*row);
        r->rgb=std::move(copy);r->width=width;r->height=height;});
}
s3d_status s3d_body_request_get_rgb(const s3d_body_request *r,const uint8_t **rgb,uint64_t *bytes,uint32_t *width,uint32_t *height,uint64_t *stride,char *e,uint64_t n){
    if(rgb)*rgb=nullptr;
    if(bytes)*bytes=0;
    if(width)*width=0;
    if(height)*height=0;
    if(stride)*stride=0;
    return boundary(e,n,[&]{require(r && !r->rgb.empty() && rgb && bytes && width && height && stride,"populated request and outputs required");*rgb=r->rgb.data();*bytes=r->rgb.size();*width=r->width;*height=r->height;*stride=uint64_t(r->width)*3;});
}
s3d_status s3d_body_request_set_geometry(s3d_body_request *r,const float *box,uint64_t count,const float *camera,uint64_t ccount,char *e,uint64_t n){
    return boundary(e,n,[&]{require(r && box && camera && count==4 && ccount==4,"box and intrinsics require four floats each");
        require(std::all_of(box,box+4,[](float x){return std::isfinite(x);}) && std::all_of(camera,camera+4,[](float x){return std::isfinite(x);}) && camera[0]>0 && camera[1]>0,"finite geometry and positive focal lengths required");
        require(box[2]>box[0] && box[3]>box[1] && std::isfinite(box[2]-box[0]) && std::isfinite(box[3]-box[1]),"positive finite box dimensions required");
        std::copy_n(box,4,r->box.begin());std::copy_n(camera,4,r->intrinsics.begin());r->geometry=true;});
}
s3d_status s3d_body_model_infer(s3d_body_model *m,const s3d_body_request *r,s3d_body_result **out,char *e,uint64_t n){
    if(out)*out=nullptr;
    return boundary(e,n,[&]{require(m && r && out && !r->rgb.empty() && r->geometry,"model, populated RGB/geometry request and output required");
        auto values=m->model->infer_rgb(r->rgb,r->width,r->height,uint64_t(r->width)*3,r->box,r->intrinsics);
        *out=sam3d::make_body_result(std::move(values),m->model->faces()).release();});
}
