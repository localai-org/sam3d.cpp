#include "sam3d.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(void){
    char error[80];s3d_objects_image_request *r=NULL;s3d_objects_image_result *out=NULL;
    uint32_t side=0;double factor=0,padding=0;uint64_t count=0;const float *data=NULL;
    CHECK(s3d_objects_image_request_create(NULL,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_objects_image_request_create(&r,NULL,1)==S3D_INVALID_ARGUMENT && !r);
    CHECK(s3d_objects_image_request_create(&r,error,sizeof(error))==S3D_OK && !error[0]);
    CHECK(s3d_objects_image_request_get_options(r,&side,&factor,&padding,error,sizeof(error))==S3D_OK);
    CHECK(side==518 && factor==1. && padding==.1);
    CHECK(s3d_objects_image_request_set_output_side(r,4,error,sizeof(error))==S3D_OK);
    CHECK(s3d_objects_image_request_set_padding(r,0,error,sizeof(error))==S3D_OK);
    CHECK(s3d_objects_image_request_set_output_side(r,0,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_objects_image_request_set_output_side(r,1025,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_objects_image_request_set_box_factor(r,NAN,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_objects_image_request_set_padding(r,INFINITY,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    CHECK(s3d_objects_image_request_get_options(r,&side,&factor,&padding,error,sizeof(error))==S3D_OK && side==4 && factor==1. && padding==0);
    uint8_t rgba[5*23];memset(rgba,255,sizeof(rgba));
    CHECK(s3d_objects_image_prepare(r,rgba,111,5,5,23,&out,error,sizeof(error))==S3D_INVALID_ARGUMENT && !out);
    CHECK(s3d_objects_image_prepare(r,rgba,sizeof(rgba),0,5,23,&out,error,sizeof(error))==S3D_INVALID_ARGUMENT && !out);
    CHECK(s3d_objects_image_prepare(r,rgba,sizeof(rgba),5,5,19,&out,error,sizeof(error))==S3D_INVALID_ARGUMENT && !out);
    CHECK(s3d_objects_image_prepare(r,rgba,sizeof(rgba),5,5,UINT64_MAX,&out,error,sizeof(error))==S3D_INVALID_ARGUMENT && !out);
    CHECK(s3d_objects_image_prepare(NULL,rgba,sizeof(rgba),5,5,23,&out,error,sizeof(error))==S3D_INVALID_ARGUMENT && !out);
    CHECK(s3d_objects_image_prepare(r,NULL,sizeof(rgba),5,5,23,&out,error,sizeof(error))==S3D_INVALID_ARGUMENT && !out);
    CHECK(s3d_objects_image_prepare(r,rgba,112,5,5,23,&out,error,sizeof(error))==S3D_OK && out && !error[0]);
    s3d_objects_image_request_free(r);r=NULL;memset(rgba,0,sizeof(rgba));
    CHECK(s3d_objects_image_get_side(out,&side,error,sizeof(error))==S3D_OK && side==4);
    for(uint32_t field=0;field<4;++field){
        CHECK(s3d_objects_image_get_tensor(out,field,&data,&count,error,sizeof(error))==S3D_OK && data && count==(field%2?16:48));
        for(uint64_t j=0;j<count;++j)CHECK(isfinite(data[j]) && fabsf(data[j]-1)<1e-6f);
    }
    CHECK(s3d_objects_image_get_tensor(out,4,&data,&count,error,sizeof(error))==S3D_INVALID_ARGUMENT && !data && !count);
    CHECK(s3d_objects_image_get_tensor(out,0,&data,&count,NULL,1)==S3D_INVALID_ARGUMENT && !data && !count);
    CHECK(s3d_objects_image_get_side(out,NULL,error,sizeof(error))==S3D_INVALID_ARGUMENT);
    s3d_objects_image_result_free(out);out=NULL;
    char one[1]={42};CHECK(s3d_objects_image_get_tensor(NULL,0,&data,&count,one,1)==S3D_INVALID_ARGUMENT && !data && !count && !one[0]);
    CHECK(s3d_objects_image_request_create(&r,NULL,0)==S3D_OK);
    CHECK(s3d_objects_image_prepare(r,rgba,sizeof(rgba),5,5,23,&out,error,sizeof(error))==S3D_INVALID_ARGUMENT && !out && error[0]);
    CHECK(s3d_objects_image_request_get_options(NULL,&side,&factor,&padding,error,sizeof(error))==S3D_INVALID_ARGUMENT && !side && !factor && !padding);
    s3d_objects_image_request_free(r);s3d_objects_image_request_free(NULL);s3d_objects_image_result_free(NULL);
    puts("Objects C API options, ownership, bounds, fields and error contract passed");return 0;
}
