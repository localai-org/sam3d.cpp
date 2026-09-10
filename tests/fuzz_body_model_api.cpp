#include "sam3d_model.h"
#include <array>
#include <cstdlib>
#include <cstring>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *input,size_t size){
    if(size<64)return 0;
    char error[17];const auto capacity=input[0]%sizeof(error);
    s3d_runtime_options *o=nullptr;s3d_body_request *r=nullptr;
    if(s3d_runtime_options_create(&o,error,capacity)!=S3D_OK || s3d_body_request_create(&r,error,capacity)!=S3D_OK)std::abort();
    std::array<uint8_t,256> pixels;for(size_t i=0;i<pixels.size();++i)pixels[i]=input[i%size];
    uint32_t width,height,backend,threads;uint64_t stride;
    std::memcpy(&width,input+4,4);std::memcpy(&height,input+8,4);std::memcpy(&stride,input+12,8);
    std::memcpy(&backend,input+20,4);std::memcpy(&threads,input+24,4);
    const char *path=reinterpret_cast<const char*>(pixels.data());
    s3d_runtime_options_set_backend(o,backend,path,input[1],input[2],threads,path,input[3],error,capacity);
    s3d_runtime_options_set_body_file(o,input[2]%5,path,input[3],error,capacity);
    uint32_t precision=99;
    auto precision_status=s3d_runtime_options_set_backbone_precision(o,input[3],error,capacity);
    if(s3d_runtime_options_get_backbone_precision(o,&precision,error,capacity)!=S3D_OK ||
       precision!=(precision_status==S3D_OK?input[3]:S3D_BACKBONE_F32))std::abort();
    const char *got=nullptr;uint64_t bytes=0;
    auto status=s3d_runtime_options_get_body_file(o,input[2]%5,&got,&bytes,error,capacity);
    if(status==S3D_OK && (!got || bytes>255))std::abort();
    // No model load: GGUF loading is explicitly excluded from fuzzing.
    s3d_body_result *result=nullptr;
    if(s3d_body_model_infer(nullptr,r,&result,error,capacity)!=S3D_INVALID_ARGUMENT || result)std::abort();
    if(input[28]&1){width=input[29]%8+1;height=input[30]%8+1;stride=width*3+input[31]%4;}
    status=s3d_body_request_set_rgb(r,pixels.data(),input[32],width,height,stride,error,capacity);
    if(status==S3D_OK){
        const uint8_t *rgb=nullptr;uint32_t w=0,h=0;uint64_t s=0;
        if(s3d_body_request_get_rgb(r,&rgb,&bytes,&w,&h,&s,error,capacity)!=S3D_OK || !rgb || w!=width || h!=height || s!=uint64_t(w)*3 || bytes!=s*h)std::abort();
        for(uint32_t y=0;y<h;++y)if(std::memcmp(rgb+y*s,pixels.data()+y*stride,size_t(s)))std::abort();
    }else if(status!=S3D_INVALID_ARGUMENT && status!=S3D_OUT_OF_MEMORY)std::abort();
    float geometry[8];std::memcpy(geometry,input+32,sizeof geometry);
    s3d_body_request_set_geometry(r,geometry,input[1]%6,geometry+4,input[2]%6,error,capacity);
    uint32_t count=99;uint64_t dim=99;
    if(s3d_body_result_get_count(nullptr,&count,error,capacity)!=S3D_INVALID_ARGUMENT || count)std::abort();
    if(s3d_body_result_get_dimension(nullptr,width,height,&dim,error,capacity)!=S3D_INVALID_ARGUMENT || dim)std::abort();
    s3d_body_request_free(r);s3d_runtime_options_free(o);return 0;
}
