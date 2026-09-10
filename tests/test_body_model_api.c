#include "sam3d_model.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(void){
    char e[16]="old";s3d_runtime_options *o=NULL;s3d_body_request *r=NULL;s3d_body_model *m=NULL;s3d_body_result *result=NULL;
    CHECK(s3d_runtime_options_create(&o,e,sizeof e)==S3D_OK && o && !e[0]);
    uint32_t precision=99;
    CHECK(s3d_runtime_options_get_backbone_precision(o,&precision,e,sizeof e)==S3D_OK && precision==S3D_BACKBONE_F32);
    CHECK(s3d_runtime_options_set_backbone_precision(o,S3D_BACKBONE_BF16,e,sizeof e)==S3D_OK);
    CHECK(s3d_runtime_options_set_backbone_precision(o,99,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_runtime_options_get_backbone_precision(o,&precision,e,sizeof e)==S3D_OK && precision==S3D_BACKBONE_BF16);
    CHECK(s3d_runtime_options_set_backbone_precision(o,S3D_BACKBONE_F32,NULL,1)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_runtime_options_get_backbone_precision(o,&precision,e,sizeof e)==S3D_OK && precision==S3D_BACKBONE_BF16);
    CHECK(s3d_runtime_options_get_backbone_precision(NULL,&precision,e,sizeof e)==S3D_INVALID_ARGUMENT && !precision);
    CHECK(s3d_runtime_options_get_backbone_precision(o,NULL,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_model_get_backbone_precision(NULL,&precision,e,sizeof e)==S3D_INVALID_ARGUMENT && !precision);
    CHECK(s3d_body_model_load(o,&m,e,sizeof e)==S3D_INVALID_ARGUMENT && !m && e[15]==0);
    CHECK(s3d_runtime_options_set_backend(o,99,"x",1,0,1,NULL,0,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_runtime_options_set_backend(o,S3D_BACKEND_CPU,"x",1,0,0,NULL,0,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_runtime_options_set_backend(o,S3D_BACKEND_CPU,"x",1,0,1,NULL,0,e,sizeof e)==S3D_OK);
    char path[]="original";
    CHECK(s3d_runtime_options_set_body_file(o,0,path,8,e,sizeof e)==S3D_OK);path[0]='X';
    const char *got=NULL;uint64_t bytes=0;
    CHECK(s3d_runtime_options_get_body_file(o,0,&got,&bytes,e,sizeof e)==S3D_OK && bytes==8 && !strcmp(got,"original"));
    CHECK(s3d_runtime_options_set_body_file(o,0,"a\0b",3,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_runtime_options_set_body_file(o,0,NULL,UINT64_MAX,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_runtime_options_get_body_file(o,0,&got,&bytes,e,sizeof e)==S3D_OK && !strcmp(got,"original"));
    CHECK(s3d_runtime_options_get_body_file(o,99,&got,&bytes,e,sizeof e)==S3D_INVALID_ARGUMENT && !got && !bytes);
    CHECK(s3d_body_request_create(&r,e,sizeof e)==S3D_OK);
    uint8_t pixels[]={1,2,3,4,5,6,99,99,7,8,9,10,11,12};
    CHECK(s3d_body_request_set_rgb(r,pixels,sizeof pixels,2,2,8,e,sizeof e)==S3D_OK);pixels[0]=42;
    const uint8_t *rgb=NULL;uint32_t w=0,h=0;uint64_t stride=0;
    CHECK(s3d_body_request_get_rgb(r,&rgb,&bytes,&w,&h,&stride,e,sizeof e)==S3D_OK && bytes==12 && w==2 && h==2 && stride==6 && rgb[0]==1 && rgb[6]==7);
    CHECK(s3d_body_request_set_rgb(r,pixels,13,2,2,8,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_request_set_rgb(r,pixels,sizeof pixels,2,2,UINT64_MAX,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_request_set_rgb(r,pixels,sizeof pixels,UINT32_MAX,2,8,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_request_set_rgb(r,pixels,sizeof pixels,2,2,8,NULL,1)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_request_get_rgb(r,&rgb,&bytes,&w,&h,&stride,e,sizeof e)==S3D_OK && rgb[0]==1);
    float box[]={0,0,2,2},camera[]={10,10,1,1};
    CHECK(s3d_body_request_set_geometry(r,box,4,camera,4,e,sizeof e)==S3D_OK);
    box[2]=NAN;CHECK(s3d_body_request_set_geometry(r,box,4,camera,4,e,sizeof e)==S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_model_infer(NULL,r,&result,e,sizeof e)==S3D_INVALID_ARGUMENT && !result);
    const char *name=NULL;const void *data=NULL;uint32_t dtype=0,rank=0,count=0;uint64_t elements=0,dim=0,caps=0;
    CHECK(s3d_body_result_get_count(NULL,&count,e,sizeof e)==S3D_INVALID_ARGUMENT && !count);
    CHECK(s3d_body_result_get_tensor(NULL,0,&name,&dtype,&rank,&elements,&data,e,sizeof e)==S3D_INVALID_ARGUMENT && !name && !data && !dtype && !rank && !elements);
    CHECK(s3d_body_result_get_dimension(NULL,0,0,&dim,e,sizeof e)==S3D_INVALID_ARGUMENT && !dim);
    CHECK(s3d_body_model_get_info(NULL,&name,&caps,e,sizeof e)==S3D_INVALID_ARGUMENT && !name && !caps);
    s3d_body_result_free(NULL);s3d_body_model_free(NULL);s3d_body_request_free(r);s3d_body_request_free(NULL);s3d_runtime_options_free(o);s3d_runtime_options_free(NULL);
    CHECK(s3d_body_request_create(&r,NULL,1)==S3D_INVALID_ARGUMENT && !r);
    puts("Body model C ABI: copied inputs, failure preservation, nulls, overflow and error buffers pass");return 0;
}
