#include "sam3d.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do {if (!(x)) {fprintf(stderr,"camera C API check failed at %d: %s\n",__LINE__,#x);return 1;}} while (0)

int main(void) {
    char error[8];s3d_body_camera_request *request=NULL;s3d_body_crop_request *crop_request=NULL;
    s3d_body_crop_result *crop=NULL;s3d_body_camera_result *result=NULL;
    CHECK(s3d_body_camera_request_create(NULL,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(error[7]==0);
    CHECK(s3d_body_camera_request_create(&request,NULL,8)==S3D_INVALID_ARGUMENT && request==NULL);
    CHECK(s3d_body_camera_request_create(&request,error,sizeof(error))==S3D_OK && error[0]==0);
    CHECK(s3d_body_crop_request_create(&crop_request,error,sizeof(error))==S3D_OK);
    float box[4]={0,0,8,8};float intrinsics[4]={4,8,4,4};
    CHECK(s3d_body_crop_request_set_box(crop_request,box,4,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_padding(crop_request,1,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_prior_aspect(crop_request,1,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_output_size(crop_request,8,8,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_compute(crop_request,&crop,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_camera_compute(request,crop,&result,error,sizeof(error))==S3D_INVALID_ARGUMENT && result==NULL);
    CHECK(s3d_body_camera_request_set_intrinsics(request,intrinsics,4,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_camera_request_set_image_size(request,8,8,error,sizeof(error))==S3D_OK);
    intrinsics[0]=NAN; /* setters copy data; failed update preserves last good value */
    CHECK(s3d_body_camera_request_set_intrinsics(request,intrinsics,4,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_camera_request_set_intrinsics(request,NULL,4,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_camera_request_set_image_size(request,0,8,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_camera_request_set_use_intrinsics_center(request,2,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_camera_compute(request,crop,&result,error,sizeof(error))==S3D_OK);
    const float *data=NULL;uint64_t count=0;uint32_t w=0,h=0;
    CHECK(s3d_body_camera_get_rays(result,&data,&count,error,sizeof(error))==S3D_OK && count==128);
    for (unsigned y=0;y<8;++y) for (unsigned x=0;x<8;++x) {
        CHECK(data[y*8+x]==((float)x-4)/4);
        CHECK(data[64+y*8+x]==((float)y-4)/8);
    }
    CHECK(s3d_body_camera_get_condition(result,&data,&count,error,sizeof(error))==S3D_OK && count==3);
    CHECK(data[0]==0 && data[1]==0 && data[2]==2);
    s3d_body_camera_result_free(result);result=NULL;
    CHECK(s3d_body_camera_request_set_image_size(request,12,10,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_camera_compute(request,crop,&result,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_camera_get_condition(result,&data,&count,error,sizeof(error))==S3D_OK);
    CHECK(data[0]==-.5f && data[1]==-.25f && data[2]==2);
    s3d_body_camera_result_free(result);result=NULL;
    CHECK(s3d_body_camera_request_set_use_intrinsics_center(request,1,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_camera_compute(request,crop,&result,error,sizeof(error))==S3D_OK);
    s3d_body_crop_result_free(crop);crop=NULL;
    /* Results are owned independently of request/crop lifetimes. */
    CHECK(s3d_body_crop_request_set_output_size(crop_request,8,4,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_compute(crop_request,&crop,error,sizeof(error))==S3D_OK);
    s3d_body_camera_result *bad=NULL;
    CHECK(s3d_body_camera_compute(request,crop,&bad,error,sizeof(error))==S3D_INVALID_ARGUMENT && bad==NULL);
    s3d_body_crop_result_free(crop);
    CHECK(s3d_body_crop_request_set_output_size(crop_request,8,8,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_request_set_rotation(crop_request,30,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_crop_compute(crop_request,&crop,error,sizeof(error))==S3D_OK);
    CHECK(s3d_body_camera_compute(request,crop,&bad,error,sizeof(error))==S3D_INVALID_ARGUMENT && bad==NULL);
    s3d_body_crop_result_free(crop);s3d_body_crop_request_free(crop_request);s3d_body_camera_request_free(request);
    CHECK(s3d_body_camera_get_size(result,&w,&h,error,sizeof(error))==S3D_OK && w==8 && h==8);
    CHECK(s3d_body_camera_get_condition(result,&data,&count,error,sizeof(error))==S3D_OK);
    CHECK(data[0]==0 && data[1]==0 && data[2]==2);
    CHECK(s3d_body_camera_get_condition(NULL,&data,&count,error,sizeof(error))==S3D_INVALID_ARGUMENT && data==NULL && count==0);
    CHECK(s3d_body_camera_get_size(NULL,&w,&h,error,sizeof(error))==S3D_INVALID_ARGUMENT && w==0 && h==0);
    s3d_body_camera_result_free(result);s3d_body_camera_result_free(NULL);s3d_body_camera_request_free(NULL);
    puts("camera C API ownership, copying, input/error validation and geometry checks passed");return 0;
}
