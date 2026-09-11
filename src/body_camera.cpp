// Copyright (c) Meta Platforms, Inc. and affiliates.
// SAM3DBody get_ray_condition / _get_decoder_condition adaptation, SAM license.
// See NOTICE; this does not implement a neural decoder.
#include "error.hpp"
#include <array>
#include <cmath>
#include <memory>
#include <vector>

struct s3d_body_camera_request {
    std::array<float,4> intrinsics{};
    uint32_t width=0,height=0;
    bool has_intrinsics=false, use_intrinsics_center=false;
};
struct s3d_body_camera_result {
    uint32_t width=0,height=0;
    std::vector<float> rays;
    std::array<float,3> condition{};
};
namespace {
using s3d::boundary;
using s3d::require;
}
s3d_status s3d_body_camera_request_create(s3d_body_camera_request **out,char *e,uint64_t n) {
    if (out) *out=nullptr;
    return boundary(e,n,[&] {require(out,"output pointer required"); *out=new s3d_body_camera_request;});
}
void s3d_body_camera_request_free(s3d_body_camera_request *request) {delete request;}
s3d_status s3d_body_camera_request_set_intrinsics(s3d_body_camera_request *request,
    const float *values,uint64_t count,char *e,uint64_t n) {
    return boundary(e,n,[&] {
        require(request && values && count==4,"four camera intrinsics required");
        for (int i=0;i<4;++i) require(std::isfinite(values[i]),"camera intrinsics must be finite");
        require(values[0]>0 && values[1]>0,"camera focal lengths must be positive");
        std::copy_n(values,4,request->intrinsics.begin()); request->has_intrinsics=true;
    });
}
s3d_status s3d_body_camera_request_set_image_size(s3d_body_camera_request *r,
    uint32_t width,uint32_t height,char *e,uint64_t n) {
    return boundary(e,n,[&] {
        require(r && width && height && width<32767 && height<32767,"camera image dimensions must be in [1,32766]");
        r->width=width;r->height=height;
    });
}
s3d_status s3d_body_camera_request_set_use_intrinsics_center(s3d_body_camera_request *r,
    uint32_t enabled,char *e,uint64_t n) {
    return boundary(e,n,[&] {require(r && enabled<=1,"camera center flag must be 0 or 1"); r->use_intrinsics_center=enabled;});
}
s3d_status s3d_body_camera_compute(const s3d_body_camera_request *r,const s3d_body_crop_result *crop,
    s3d_body_camera_result **out,char *e,uint64_t n) {
    if (out) *out=nullptr;
    return boundary(e,n,[&] {
        require(r && crop && out,"camera request, crop and output required");
        require(r->has_intrinsics && r->width && r->height,"set camera intrinsics and original-image size first");
        auto result=std::make_unique<s3d_body_camera_result>();
        require(s3d_body_crop_get_size(crop,&result->width,&result->height,nullptr,0)==S3D_OK,"invalid crop size");
        require(result->width==result->height && result->width<=512,"camera rays require a square crop up to 512x512");
        const double *affine=nullptr; uint64_t count=0;
        require(s3d_body_crop_get_affine(crop,&affine,&count,nullptr,0)==S3D_OK && count==6,"invalid crop affine");
        // Original prepare_batch converts OpenCV F64 metadata to F32 first.
        std::array<float,6> a;
        for (int i=0;i<6;++i) {a[i]=float(affine[i]); require(std::isfinite(a[i]),"camera affine overflow");}
        require(a[0]>0 && a[4]>0,"camera crop scale must be positive");
        const float scale=std::max(std::abs(a[0]),std::abs(a[4]));
        require(std::abs(a[1])<=scale*1e-6f && std::abs(a[3])<=scale*1e-6f,
                "camera rays support axis-aligned crops only");
        const uint64_t pixels=uint64_t(result->width)*result->height;
        result->rays.resize(pixels*2);
        for (int axis=0;axis<2;++axis) {
            const float zoom=a[axis*4], translation=a[axis*3+2]/zoom;
            for (uint32_t y=0;y<result->height;++y) for (uint32_t x=0;x<result->width;++x) {
                float value=float(axis==0?x:y)/zoom;
                value=value-translation;
                value=value-r->intrinsics[axis+2];
                value=value/r->intrinsics[axis];
                require(std::isfinite(value),"camera ray overflow");
                result->rays[axis*pixels+uint64_t(y)*result->width+x]=value;
            }
        }
        const float *center=nullptr,*box_scale=nullptr;
        require(s3d_body_crop_get_center(crop,&center,&count,nullptr,0)==S3D_OK && count==2,"invalid crop center");
        require(s3d_body_crop_get_scale(crop,&box_scale,&count,nullptr,0)==S3D_OK && count==2,"invalid crop scale");
        const float ox=r->use_intrinsics_center?r->intrinsics[2]:float(r->width)/2.f;
        const float oy=r->use_intrinsics_center?r->intrinsics[3]:float(r->height)/2.f;
        result->condition={(center[0]-ox)/r->intrinsics[0],(center[1]-oy)/r->intrinsics[0],box_scale[0]/r->intrinsics[0]};
        for (auto value:result->condition) require(std::isfinite(value),"camera condition overflow");
        *out=result.release();
    });
}
void s3d_body_camera_result_free(s3d_body_camera_result *result) {delete result;}
s3d_status s3d_body_camera_get_rays(const s3d_body_camera_result *r,const float **data,
    uint64_t *count,char *e,uint64_t n) {
    if (data) *data=nullptr;
    if (count) *count=0;
    return boundary(e,n,[&] {require(r && data && count,"camera result and outputs required"); *data=r->rays.data();*count=r->rays.size();});
}
s3d_status s3d_body_camera_get_condition(const s3d_body_camera_result *r,const float **data,
    uint64_t *count,char *e,uint64_t n) {
    if (data) *data=nullptr;
    if (count) *count=0;
    return boundary(e,n,[&] {require(r && data && count,"camera result and outputs required"); *data=r->condition.data();*count=3;});
}
s3d_status s3d_body_camera_get_size(const s3d_body_camera_result *r,uint32_t *w,uint32_t *h,char *e,uint64_t n) {
    if (w) *w=0;
    if (h) *h=0;
    return boundary(e,n,[&] {require(r && w && h,"camera result and outputs required");*w=r->width;*h=r->height;});
}
