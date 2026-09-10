#include "sam3d.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *input,size_t size){
    if(size<32)return 0;
    double factor,padding;uint32_t width,height;
    std::memcpy(&factor,input,8);std::memcpy(&padding,input+8,8);std::memcpy(&width,input+16,4);std::memcpy(&height,input+20,4);
    char error[41];const uint64_t ecap=input[24]%sizeof(error);
    s3d_objects_image_request *r=nullptr;auto status=s3d_objects_image_request_create(&r,error,ecap);
    if(status==S3D_OUT_OF_MEMORY)return 0;if(status!=S3D_OK || !r)std::abort();
    s3d_objects_image_request_set_output_side(r,width,error,ecap);
    // Bound successful output allocations while still fuzzing invalid setters.
    if(s3d_objects_image_request_set_output_side(r,width%32+1,error,ecap)!=S3D_OK)std::abort();
    s3d_objects_image_request_set_box_factor(r,factor,error,ecap);s3d_objects_image_request_set_padding(r,padding,error,ecap);
    uint32_t side=0;double f=0,p=0;
    if(s3d_objects_image_request_get_options(r,&side,&f,&p,error,ecap)!=S3D_OK || side<1 || side>32 || !std::isfinite(f) || !std::isfinite(p))std::abort();
    std::array<uint8_t,256> pixels;for(size_t i=0;i<pixels.size();++i)pixels[i]=input[i%size];
    const bool bounded=(input[25]&1);uint32_t w=bounded?8:width,h=bounded?8:height;
    uint64_t stride=bounded?32:uint64_t(input[26])*4;
    const uint64_t capacity=input[27]&1?pixels.size():input[28];
    s3d_objects_image_result *result=nullptr;status=s3d_objects_image_prepare(r,pixels.data(),capacity,w,h,stride,&result,error,ecap);
    s3d_objects_image_request_free(r);
    if(status==S3D_OK){
        if(!result)std::abort();uint32_t actual=0;
        if(s3d_objects_image_get_side(result,&actual,error,ecap)!=S3D_OK || actual!=side)std::abort();
        for(uint32_t field=0;field<4;++field){
            const float *v=nullptr;uint64_t n=0;
            if(s3d_objects_image_get_tensor(result,field,&v,&n,error,ecap)!=S3D_OK || !v || n!=uint64_t(side)*side*(field%2?1:3))std::abort();
            for(uint64_t i=0;i<n;++i)if(!std::isfinite(v[i]) || (field%2 && v[i]!=0 && v[i]!=1))std::abort();
        }
        const float *v=nullptr;uint64_t n=0;
        if(s3d_objects_image_get_tensor(result,UINT32_MAX,&v,&n,error,ecap)!=S3D_INVALID_ARGUMENT || v || n)std::abort();
    }else if(result || (status!=S3D_INVALID_ARGUMENT && status!=S3D_OUT_OF_MEMORY))std::abort();
    s3d_objects_image_result_free(result);return 0;
}
