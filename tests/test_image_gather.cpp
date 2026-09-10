#include "sam3d.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void flag(const char *value) {
#ifdef _WIN32
    _putenv_s("SAM3D_IMAGE_GATHER",value?value:"");
#else
    if(value)setenv("SAM3D_IMAGE_GATHER",value,1);else unsetenv("SAM3D_IMAGE_GATHER");
#endif
}
void check(s3d_status status){if(status!=S3D_OK)throw std::runtime_error("image gather C API failure");}
}
int main()try {
    const char *old=std::getenv("SAM3D_IMAGE_GATHER");
    struct restore {bool present;std::string value;~restore(){flag(present?value.c_str():nullptr);}} guard{old!=nullptr,old?old:""};
    constexpr std::array<unsigned,8> widths{1,2,7,63,127,511,512,513};
    constexpr std::array<float,4> rotations{0,17.5f,90,-45};
    for(unsigned trial=0;trial<80;++trial) {
        unsigned w=widths[trial%8],h=widths[(trial/8)%8],stride=w*3+trial%11;
        const size_t span=size_t(stride)*(h-1)+w*3;
        // Allocate the exact valid span: sanitizer redzones catch any reads
        // outside even when the final row lacks stride padding.
        std::vector<uint8_t> pixels(span);
        for(size_t i=0;i<span;++i)pixels[i]=uint8_t((i*37+trial*19)%256);
        s3d_body_crop_request *raw_request=nullptr;
        check(s3d_body_crop_request_create(&raw_request,nullptr,0));
        std::unique_ptr<s3d_body_crop_request,decltype(&s3d_body_crop_request_free)> request(raw_request,s3d_body_crop_request_free);
        const float shift=float(int(trial%5)-2)*.4f;
        const float box[4]={shift*w,shift*h,(shift+1)*w,(shift+1)*h};
        check(s3d_body_crop_request_set_box(request.get(),box,4,nullptr,0));
        check(s3d_body_crop_request_set_rotation(request.get(),rotations[trial%4],nullptr,0));
        check(s3d_body_crop_request_set_output_size(request.get(),widths[(trial+3)%8],widths[(trial+5)%8],nullptr,0));
        s3d_body_crop_result *raw_crop=nullptr;check(s3d_body_crop_compute(request.get(),&raw_crop,nullptr,0));
        std::unique_ptr<s3d_body_crop_result,decltype(&s3d_body_crop_result_free)> crop(raw_crop,s3d_body_crop_result_free);
        std::vector<uint8_t> expected_rgb;std::vector<float> expected_normalized;
        for(const char *mode: {"0","1","1"}) {
            flag(mode);s3d_body_image_result *raw_image=nullptr;
            check(s3d_body_image_prepare(crop.get(),pixels.data(),pixels.size(),w,h,stride,&raw_image,nullptr,0));
            std::unique_ptr<s3d_body_image_result,decltype(&s3d_body_image_result_free)> image(raw_image,s3d_body_image_result_free);
            const uint8_t *rgb=nullptr;const float *normalized=nullptr;uint64_t nr=0,nf=0;
            check(s3d_body_image_get_rgb(image.get(),&rgb,&nr,nullptr,0));
            check(s3d_body_image_get_normalized(image.get(),&normalized,&nf,nullptr,0));
            if(*mode=='0'){expected_rgb.assign(rgb,rgb+nr);expected_normalized.assign(normalized,normalized+nf);}
            else if(nr!=expected_rgb.size() || nf!=expected_normalized.size() ||
                std::memcmp(rgb,expected_rgb.data(),nr) || std::memcmp(normalized,expected_normalized.data(),nf*sizeof(float)))
                throw std::runtime_error("image gather changed bytes in trial "+std::to_string(trial));
        }
    }
    std::cout<<"80 image gather cases passed exactly, including repeated calls, rotation, borders, strides and singleton dimensions\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
