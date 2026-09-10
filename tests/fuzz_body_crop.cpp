#include "sam3d.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 40) return 0;
    float values[8];
    uint32_t width, height;
    std::memcpy(values, data, sizeof(values));
    std::memcpy(&width, data + 32, 4); std::memcpy(&height, data + 36, 4);
    char error[73];
    const uint64_t capacity = size > 40 ? data[40] % sizeof(error) : sizeof(error);
    s3d_body_crop_request *request = nullptr;
    s3d_body_crop_result *result = nullptr;
    auto status = s3d_body_crop_request_create(&request, error, capacity);
    if (status == S3D_OUT_OF_MEMORY) return 0;
    if (status != S3D_OK || !request) std::abort();
    s3d_body_crop_request_set_box(request, values, 4, error, capacity);
    s3d_body_crop_request_set_padding(request, values[4], error, capacity);
    s3d_body_crop_request_set_prior_aspect(request, values[5], error, capacity);
    s3d_body_crop_request_set_rotation(request, values[6], error, capacity);
    s3d_body_crop_request_set_output_size(request, width, height, error, capacity);
    // Exercise image preparation with a bounded output allocation on each iteration.
    s3d_body_crop_request_set_output_size(request,width%32+1,height%32+1,error,capacity);
    if (size>42 && (data[42]&1)) {
        // Reach successful camera-ray paths as well as unsupported crop cases.
        s3d_body_crop_request_set_rotation(request,0,error,capacity);
        s3d_body_crop_request_set_output_size(request,width%32+1,width%32+1,error,capacity);
    }
    status = s3d_body_crop_compute(request, &result, error, capacity);
    s3d_body_crop_request_free(request);
    if (status == S3D_OK) {
        if (!result) std::abort();
        const double *affine = nullptr; uint64_t count = 0;
        if (s3d_body_crop_get_affine(result, &affine, &count, error, capacity) != S3D_OK || count != 6)
            std::abort();
        for (uint64_t i = 0; i < count; ++i) if (!std::isfinite(affine[i])) std::abort();
        s3d_body_image_result *image=nullptr;
        auto image_status=s3d_body_image_prepare(result,data,size,2,2,6,&image,error,capacity);
        if(image_status==S3D_OK) {
            const float *normalized=nullptr;
            if(s3d_body_image_get_normalized(image,&normalized,&count,error,capacity)!=S3D_OK) std::abort();
            for(uint64_t i=0;i<count;++i) if(!std::isfinite(normalized[i])) std::abort();
        } else if(image || (image_status!=S3D_INVALID_ARGUMENT && image_status!=S3D_OUT_OF_MEMORY)) {
            std::abort();
        }
        s3d_body_image_result_free(image);
        s3d_body_camera_request *camera=nullptr;
        auto camera_status=s3d_body_camera_request_create(&camera,error,capacity);
        if (camera_status==S3D_OK) {
            const float defaults[4]={3,4,1,1};
            if (s3d_body_camera_request_set_intrinsics(camera,defaults,4,error,capacity)!=S3D_OK) std::abort();
            if (s3d_body_camera_request_set_image_size(camera,2,2,error,capacity)!=S3D_OK) std::abort();
            s3d_body_camera_request_set_intrinsics(camera,values+4,4,error,capacity);
            s3d_body_camera_request_set_image_size(camera,width,height,error,capacity);
            s3d_body_camera_request_set_use_intrinsics_center(camera,size>41?data[41]:0,error,capacity);
            s3d_body_camera_result *condition=nullptr;
            camera_status=s3d_body_camera_compute(camera,result,&condition,error,capacity);
            s3d_body_camera_request_free(camera);
            if (camera_status==S3D_OK) {
                const float *data=nullptr;uint64_t count=0;
                if (s3d_body_camera_get_rays(condition,&data,&count,error,capacity)!=S3D_OK) std::abort();
                for (uint64_t i=0;i<count;++i) if (!std::isfinite(data[i])) std::abort();
                if (s3d_body_camera_get_condition(condition,&data,&count,error,capacity)!=S3D_OK || count!=3) std::abort();
                for (uint64_t i=0;i<count;++i) if (!std::isfinite(data[i])) std::abort();
            } else if (condition || (camera_status!=S3D_INVALID_ARGUMENT && camera_status!=S3D_OUT_OF_MEMORY)) std::abort();
            s3d_body_camera_result_free(condition);
        } else if (camera || camera_status!=S3D_OUT_OF_MEMORY) std::abort();
    } else if (result || (status != S3D_INVALID_ARGUMENT && status != S3D_OUT_OF_MEMORY)) {
        std::abort();
    }
    s3d_body_crop_result_free(result);
    return 0;
}
