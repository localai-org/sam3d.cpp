#include "sam3d.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

int main(int argc,char **argv) {
    if(argc!=3) { std::cerr<<"usage: sam3d-image-capture FIXTURE_DIR OUTPUT_DIR\n"; return 2; }
    namespace fs=std::filesystem;
    const fs::path dir(argv[1]),dest(argv[2]);
    fs::create_directories(dest);
    std::ifstream cases(dir/"cases.txt");
    std::ofstream rows(dest/"results.txt");
    std::string header; cases>>header;
    if(header!="SAM3D_IMAGE_CASES_V1" || !rows) return 2;
    rows<<"SAM3D_IMAGE_RESULTS_V1\n";
    s3d_body_crop_request *request=nullptr;
    s3d_body_crop_result *crop=nullptr;
    s3d_body_image_result *result=nullptr;
    char error[256]{};
    auto check=[&](s3d_status s) { if(s!=S3D_OK) throw std::runtime_error(error); };
    try {
        for(unsigned count=0;;++count) {
            cases>>std::ws; if(cases.eof()) break;
            if(count>=1000) throw std::runtime_error("too many cases");
            uint32_t id,w,h,ow,oh; uint64_t stride; float padding,rotation,box[4];
            if(!(cases>>id>>w>>h>>stride>>ow>>oh>>padding>>rotation>>box[0]>>box[1]>>box[2]>>box[3]))
                throw std::runtime_error("bad case row");
            if(!h || stride>UINT64_C(100000000)/h) throw std::runtime_error("oversized input fixture");
            std::ostringstream prefix; prefix<<"case."<<std::setw(4)<<std::setfill('0')<<id;
            std::vector<uint8_t> input(stride*h);
            std::ifstream raw(dir/(prefix.str()+".input.rgb"),std::ios::binary);
            if(!raw.read(reinterpret_cast<char *>(input.data()),input.size())) throw std::runtime_error("short input");
            check(s3d_body_crop_request_create(&request,error,sizeof(error)));
            check(s3d_body_crop_request_set_box(request,box,4,error,sizeof(error)));
            check(s3d_body_crop_request_set_padding(request,padding,error,sizeof(error)));
            check(s3d_body_crop_request_set_rotation(request,rotation,error,sizeof(error)));
            check(s3d_body_crop_request_set_output_size(request,ow,oh,error,sizeof(error)));
            check(s3d_body_crop_compute(request,&crop,error,sizeof(error)));
            check(s3d_body_image_prepare(crop,input.data(),input.size(),w,h,stride,&result,error,sizeof(error)));
            const uint8_t *rgb=nullptr; const float *norm=nullptr; uint64_t nrgb=0,nnorm=0;
            check(s3d_body_image_get_rgb(result,&rgb,&nrgb,error,sizeof(error)));
            check(s3d_body_image_get_normalized(result,&norm,&nnorm,error,sizeof(error)));
            std::ofstream out_rgb(dest/(prefix.str()+".rgb"),std::ios::binary);
            std::ofstream out_norm(dest/(prefix.str()+".normalized.f32"),std::ios::binary);
            out_rgb.write(reinterpret_cast<const char *>(rgb),nrgb);
            out_norm.write(reinterpret_cast<const char *>(norm),nnorm*sizeof(float));
            if(!out_rgb || !out_norm) throw std::runtime_error("write failed");
            uint32_t actual_w=0,actual_h=0;
            check(s3d_body_image_get_size(result,&actual_w,&actual_h,error,sizeof(error)));
            rows<<id<<' '<<actual_w<<' '<<actual_h<<'\n';
            s3d_body_image_result_free(result); result=nullptr;
            s3d_body_crop_result_free(crop); crop=nullptr;
            s3d_body_crop_request_free(request); request=nullptr;
        }
    } catch(const std::exception &e) {
        std::cerr<<e.what()<<'\n';
        s3d_body_image_result_free(result); s3d_body_crop_result_free(crop); s3d_body_crop_request_free(request);
        return 1;
    }
    return rows ? 0 : 1;
}
