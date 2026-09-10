#include "sam3d.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"failed line %d: %s\n",__LINE__,#x); return 1; } } while(0)
int main(void) {
    const uint8_t rgb[18]={0,1,255, 2,3,4, 193,193,193, 128,64,32, 255,255,255, 193,193,193};
    const float box[4]={0,0,2,2};
    s3d_body_crop_request *request=NULL;
    s3d_body_crop_result *crop=NULL;
    s3d_body_image_result *image=NULL;
    const uint8_t *pixels=NULL; const float *normalized=NULL;
    uint64_t count=0; uint32_t w=0,h=0;
    char error[128];
    CHECK(s3d_body_crop_request_create(&request,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_box(request,box,4,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_padding(request,1,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_prior_aspect(request,1,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_output_size(request,2,2,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_compute(request,&crop,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_image_prepare(crop,rgb,14,2,2,9,&image,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(image==NULL);
    CHECK(s3d_body_image_prepare(crop,rgb,18,2,2,5,&image,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_image_prepare(crop,rgb,18,2,2,UINT64_MAX,&image,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_image_prepare(crop,rgb,18,0,2,9,&image,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_image_prepare(crop,NULL,18,2,2,9,&image,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_image_prepare(crop,rgb,15,2,2,9,&image,error,sizeof(error))==S3D_OK);
    s3d_body_crop_result_free(crop);
    CHECK(s3d_body_image_get_size(image,&w,&h,error,sizeof(error))==S3D_OK && w==2 && h==2);
    CHECK(s3d_body_image_get_rgb(image,&pixels,&count,error,sizeof(error))==S3D_OK && count==12);
    for(unsigned i=0;i<12;++i) CHECK(pixels[i]==rgb[(i/6)*9+i%6]);
    CHECK(s3d_body_image_get_normalized(image,&normalized,&count,error,sizeof(error))==S3D_OK && count==12);
    CHECK(fabsf(normalized[0]-(0-.485f)/.229f)<1e-6f);
    CHECK(fabsf(normalized[4]-(1/255.0f-.456f)/.224f)<1e-6f);
    CHECK(fabsf(normalized[8]-(1-.406f)/.225f)<1e-6f);
    s3d_body_image_result_free(image); image=NULL;
    CHECK(s3d_body_image_get_rgb(NULL,&pixels,&count,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(!pixels && !count);
    CHECK(s3d_body_crop_request_set_output_size(request,4097,4097,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_compute(request,&crop,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_image_prepare(crop,rgb,18,2,2,9,&image,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(image==NULL);
    s3d_body_crop_result_free(crop); s3d_body_crop_request_free(request);
    s3d_body_image_result_free(NULL);
    /* Full rotated-pattern RGB regression captured from actual upstream.
     * Provenance: tests/fixtures/body-image-pattern.json. This catches affine
     * rounding changes that the old floating-matrix tolerance could not. */
    uint8_t pattern[81*127*3];
    for(unsigned y=0;y<81;++y) for(unsigned x=0;x<127;++x) {
        pattern[(y*127+x)*3]=(uint8_t)((x*31+y*7)%256);
        pattern[(y*127+x)*3+1]=(uint8_t)((x*3+y*53)%256);
        pattern[(y*127+x)*3+2]=(uint8_t)((x^y)%256);
    }
    const float rotated_box[4]={-44.45f,-8.1f,142.24f,78.57f};
    CHECK(s3d_body_crop_request_create(&request,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_box(request,rotated_box,4,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_rotation(request,17.5f,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_compute(request,&crop,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_image_prepare(crop,pattern,sizeof(pattern),127,81,127*3,&image,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_image_get_rgb(image,&pixels,&count,error,sizeof(error))==S3D_OK && count==512*512*3);
    uint64_t checksum=UINT64_C(14695981039346656037);
    for(uint64_t i=0;i<count;++i) checksum=(checksum^pixels[i])*UINT64_C(1099511628211);
    CHECK(checksum==UINT64_C(0xb5f5723212f21032));
    s3d_body_image_result_free(image); s3d_body_crop_result_free(crop); s3d_body_crop_request_free(request);
    /* Every byte value in every channel: normalization must keep exact F32
     * bits, not just an approximate float tolerance. Identity 16x16 crop. */
    uint8_t ramp[256*3];
    for(unsigned i=0;i<256;++i){ramp[i*3]=(uint8_t)i;ramp[i*3+1]=(uint8_t)(i*17);ramp[i*3+2]=(uint8_t)(255-i);}
    const float ramp_box[4]={0,0,16,16},means[3]={.485f,.456f,.406f},stddevs[3]={.229f,.224f,.225f};
    CHECK(s3d_body_crop_request_create(&request,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_box(request,ramp_box,4,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_padding(request,1,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_prior_aspect(request,1,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_output_size(request,16,16,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_compute(request,&crop,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_image_prepare(crop,ramp,sizeof(ramp),16,16,48,&image,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_image_get_rgb(image,&pixels,&count,error,sizeof(error))==S3D_OK && count==sizeof(ramp));
    CHECK(memcmp(pixels,ramp,sizeof(ramp))==0);
    CHECK(s3d_body_image_get_normalized(image,&normalized,&count,error,sizeof(error))==S3D_OK && count==768);
    for(unsigned c=0;c<3;++c)for(unsigned i=0;i<256;++i){
        const float unit=(float)ramp[i*3+c]/255.0f,expected=(unit-means[c])/stddevs[c];
        CHECK(memcmp(&normalized[c*256+i],&expected,sizeof(float))==0);
    }
    s3d_body_image_result_free(image);s3d_body_crop_result_free(crop);s3d_body_crop_request_free(request);
    puts("C image API: stride, bounds, layout, exact all-byte normalization and ownership passed");
    return 0;
}
