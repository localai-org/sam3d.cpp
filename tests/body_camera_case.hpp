#pragma once
#include "sam3d.h"
#include <array>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct camera_case {
    uint32_t width,height,side,use_center;
    float padding;
    std::array<float,4> box,intrinsics;
};
inline std::map<std::string,std::vector<float>> run_camera_case(const camera_case &s) {
    char error[256]{};
    auto check=[&](s3d_status status) {if (status!=S3D_OK) throw std::runtime_error(error);};
    s3d_body_crop_request *crop_raw=nullptr;check(s3d_body_crop_request_create(&crop_raw,error,sizeof(error)));
    std::unique_ptr<s3d_body_crop_request,decltype(&s3d_body_crop_request_free)> request(crop_raw,s3d_body_crop_request_free);
    check(s3d_body_crop_request_set_box(request.get(),s.box.data(),4,error,sizeof(error)));
    check(s3d_body_crop_request_set_padding(request.get(),s.padding,error,sizeof(error)));
    check(s3d_body_crop_request_set_output_size(request.get(),s.side,s.side,error,sizeof(error)));
    s3d_body_crop_result *geometry_raw=nullptr;check(s3d_body_crop_compute(request.get(),&geometry_raw,error,sizeof(error)));
    std::unique_ptr<s3d_body_crop_result,decltype(&s3d_body_crop_result_free)> crop(geometry_raw,s3d_body_crop_result_free);
    s3d_body_camera_request *camera_raw=nullptr;check(s3d_body_camera_request_create(&camera_raw,error,sizeof(error)));
    std::unique_ptr<s3d_body_camera_request,decltype(&s3d_body_camera_request_free)> camera(camera_raw,s3d_body_camera_request_free);
    check(s3d_body_camera_request_set_intrinsics(camera.get(),s.intrinsics.data(),4,error,sizeof(error)));
    check(s3d_body_camera_request_set_image_size(camera.get(),s.width,s.height,error,sizeof(error)));
    check(s3d_body_camera_request_set_use_intrinsics_center(camera.get(),s.use_center,error,sizeof(error)));
    s3d_body_camera_result *result_raw=nullptr;check(s3d_body_camera_compute(camera.get(),crop.get(),&result_raw,error,sizeof(error)));
    std::unique_ptr<s3d_body_camera_result,decltype(&s3d_body_camera_result_free)> result(result_raw,s3d_body_camera_result_free);
    std::map<std::string,std::vector<float>> output;
    const float *values=nullptr;uint64_t count=0;
    auto copy=[&](const std::string &name){output[name]={values,values+count};};
    check(s3d_body_crop_get_center(crop.get(),&values,&count,error,sizeof(error)));copy("00.center");
    check(s3d_body_crop_get_scale(crop.get(),&values,&count,error,sizeof(error)));copy("01.scale");
    const double *affine=nullptr;check(s3d_body_crop_get_affine(crop.get(),&affine,&count,error,sizeof(error)));
    auto &f32=output["02.affine"];for (uint64_t i=0;i<count;++i) f32.push_back(float(affine[i]));
    check(s3d_body_camera_get_rays(result.get(),&values,&count,error,sizeof(error)));copy("03.rays");
    check(s3d_body_camera_get_condition(result.get(),&values,&count,error,sizeof(error)));copy("04.cliff");
    return output;
}
