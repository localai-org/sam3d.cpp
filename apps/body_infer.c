/* Native demo/CLI consumer of the public C API. No Python or injected state.
 * Wire formats are described in demo/README.md. The C API ownership regression
 * deliberately compiles this same consumer. */
#include "sam3d_model.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double monotonic_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1000.0+(double)t.tv_nsec/1000000.0;}
static char error[512];
static uint32_t backbone_precision=S3D_BACKBONE_F32;
static uint32_t body_crop=512,body_mask=31,body_correctives=1,body_slim=0;
#define INFERENCE_USAGE "[--bf16] [--body-crop-size=384|448|512] [--body-intermediates=0,1,2|none] [--no-body-correctives] [--slim-body-intermediates]"
static int number(const char *s,uint32_t *out){char *end;errno=0;unsigned long v=strtoul(s,&end,10);if(errno || !*s || *end || *s=='-' || v>UINT32_MAX)return 0;*out=(uint32_t)v;return 1;}
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s: %s\n",__LINE__,#x,error);goto done;}}while(0)
#define API(x) CHECK((x)==S3D_OK)
static int infer_main(int argc,char **argv,s3d_body_model **reused){
    int result=1;FILE *in=NULL,*out=NULL;uint8_t *rgb=NULL;
    s3d_runtime_options *options=NULL;s3d_body_request *request=NULL;s3d_body_model *model=reused?*reused:NULL;s3d_body_result *motion=NULL;
    if(argc!=11){fprintf(stderr,"Usage: %s MODULE CPU|Vulkan DEVICE DESCRIPTION|- BACKBONE.gguf BRANCH.gguf MHR.gguf IMAGE.input RESULT.bin THREADS " INFERENCE_USAGE "\n",argv[0]);return 2;}uint32_t device,threads;
    CHECK(number(argv[3],&device) && number(argv[10],&threads));
    CHECK(!strcmp(argv[2],"CPU") || !strcmp(argv[2],"Vulkan"));
    in=fopen(argv[8],"rb");CHECK(in);char magic[8];uint32_t dims[3];float box[4],camera[4];
    CHECK(fread(magic,1,8,in)==8 && !memcmp(magic,"S3DIMG01",8));
    CHECK(fread(dims,4,3,in)==3 && fread(box,4,4,in)==4 && fread(camera,4,4,in)==4);
    CHECK(dims[0]>0 && dims[1]>0 && dims[0]<=32766 && dims[1]<=32766 && (uint64_t)dims[0]*dims[1]<=16000000 && dims[2]==dims[0]*3);
    const size_t bytes=(size_t)dims[2]*dims[1];rgb=malloc(bytes);CHECK(rgb && fread(rgb,1,bytes,in)==bytes && fgetc(in)==EOF);fclose(in);in=NULL;
    API(s3d_runtime_options_create(&options,error,sizeof error));
    API(s3d_runtime_options_set_backbone_precision(options,backbone_precision,error,sizeof error));
    API(s3d_runtime_options_set_body_inference(options,body_crop,body_mask,body_correctives,body_slim,error,sizeof error));
    const char *description=strcmp(argv[4],"-")?argv[4]:"";
    API(s3d_runtime_options_set_backend(options,!strcmp(argv[2],"CPU")?S3D_BACKEND_CPU:S3D_BACKEND_VULKAN,argv[1],strlen(argv[1]),device,threads,description,strlen(description),error,sizeof error));
    for(uint32_t i=0;i<3;++i)API(s3d_runtime_options_set_body_file(options,i,argv[5+i],strlen(argv[5+i]),error,sizeof error));
    API(s3d_body_request_create(&request,error,sizeof error));
    API(s3d_body_request_set_rgb(request,rgb,bytes,dims[0],dims[1],dims[2],error,sizeof error));
    API(s3d_body_request_set_geometry(request,box,4,camera,4,error,sizeof error));
    memset(rgb,0,bytes);free(rgb);rgb=NULL;memset(box,0,sizeof box);memset(camera,0,sizeof camera);
    if(!model){
        fprintf(stderr,"STAGE Loading and validating GGUF models\n");fflush(stderr);
        API(s3d_body_model_load(options,&model,error,sizeof error));
    }
    s3d_runtime_options_free(options);options=NULL;
    uint64_t capabilities=0;API(s3d_body_model_get_info(model,&description,&capabilities,error,sizeof error));
    CHECK(capabilities==S3D_CAP_BODY_POSE_BRANCH);fprintf(stderr,"C API model: %s\n",description);
    fprintf(stderr,"STAGE Estimating body: image encoder, pose decoder and mesh\n");fflush(stderr);
    const double inference_start=monotonic_ms();
    API(s3d_body_model_infer(model,request,&motion,error,sizeof error));
    // Worker timing precedes DONE on the SAME pipe. Separate stdout/stderr
    // readers cannot reliably preserve cross-pipe message ordering.
    FILE *timing_stream=reused?stdout:stderr;
    fprintf(timing_stream,"TIMING body_infer_ms %.6f\n",monotonic_ms()-inference_start);fflush(timing_stream);
    fprintf(stderr,"STAGE Writing body result\n");fflush(stderr);
    s3d_body_request_free(request);request=NULL;
    // One-shot mode deliberately tests result ownership after model destruction.
    if(!reused){s3d_body_model_free(model);model=NULL;}
    uint32_t count=0;API(s3d_body_result_get_count(motion,&count,error,sizeof error));
    out=fopen(argv[9],"wb");CHECK(out && fwrite("S3DOUT01",1,8,out)==8 && fwrite(&count,4,1,out)==1);
    for(uint32_t i=0;i<count;++i){const char *name=NULL;const void *data=NULL;uint32_t type,rank;uint64_t elements;
        API(s3d_body_result_get_tensor(motion,i,&name,&type,&rank,&elements,&data,error,sizeof error));
        const uint32_t len=(uint32_t)strlen(name);
        CHECK(fwrite(&len,4,1,out)==1 && fwrite(name,1,len,out)==len && fwrite(&type,4,1,out)==1 && fwrite(&rank,4,1,out)==1 && fwrite(&elements,8,1,out)==1);
        for(uint32_t j=0;j<rank;++j){uint64_t dim;API(s3d_body_result_get_dimension(motion,i,j,&dim,error,sizeof error));CHECK(fwrite(&dim,8,1,out)==1);}
        CHECK(fwrite(data,4,(size_t)elements,out)==elements);
    }
    {int closed=fclose(out);out=NULL;CHECK(!closed);}result=0;
done:
    if(in)fclose(in);if(out)fclose(out);free(rgb);s3d_body_result_free(motion);
    if(reused)*reused=model;else s3d_body_model_free(model);
    s3d_body_request_free(request);s3d_runtime_options_free(options);return result;
}
// Private local worker protocol, not a network interface. Paths are framed as
// uint32 little-endian length + non-NUL bytes (1..4096). No shell interpretation.
// EOF at a request boundary shuts down cleanly; malformed/failed jobs terminate
// the worker so its owner can discard all state and restart on the next job.
static int read_path(char path[4097],int boundary){
    unsigned char size[4];size_t got=fread(size,1,4,stdin);
    if(!got && feof(stdin) && boundary)return 0;
    if(got!=4)return -1;
    uint32_t n=(uint32_t)size[0]|(uint32_t)size[1]<<8|(uint32_t)size[2]<<16|(uint32_t)size[3]<<24;
    if(!n || n>4096 || fread(path,1,n,stdin)!=n || memchr(path,0,n))return -1;
    path[n]=0;return 1;
}
int main(int argc,char **argv){
    /* Precision is fixed for the entire persistent worker lifetime. */
    while(argc>1 && !strncmp(argv[argc-1],"--",2)){
        const char *arg=argv[argc-1];
        if(!strcmp(arg,"--bf16"))backbone_precision=S3D_BACKBONE_BF16;
        else if(!strcmp(arg,"--no-body-correctives"))body_correctives=0;
        else if(!strcmp(arg,"--slim-body-intermediates"))body_slim=1;
        else if(!strncmp(arg,"--body-crop-size=",17)){
            if(!number(arg+17,&body_crop) || (body_crop!=384 && body_crop!=448 && body_crop!=512))return 2;
        }else if(!strncmp(arg,"--body-intermediates=",21)){
            const char *v=arg+21;body_mask=0;
            if(strcmp(v,"none")){
                if(!*v)return 2;
                for(;;){if(*v<'0' || *v>'4')return 2;uint32_t bit=1u<<(*v++-'0');if(body_mask&bit)return 2;body_mask|=bit;
                    if(!*v)break;if(*v++!=',' || !*v)return 2;}
            }
        }else{fprintf(stderr,"unknown option: %s\n",arg);return 2;}
        --argc;
    }
    fprintf(stderr,"Body inference: crop=%u intermediate_mask=%u correctives=%u slim=%u\n",body_crop,body_mask,body_correctives,body_slim);
    if(argc<2 || strcmp(argv[1],"--worker"))return infer_main(argc,argv,NULL);
    if(argc!=10){fprintf(stderr,"Usage: %s --worker MODULE CPU|Vulkan DEVICE DESCRIPTION|- BACKBONE.gguf BRANCH.gguf MHR.gguf THREADS " INFERENCE_USAGE "\n",argv[0]);return 2;}
    uint32_t device,threads;
    if(!number(argv[4],&device) || !number(argv[9],&threads) || !threads || threads>256 ||
       (strcmp(argv[3],"CPU") && strcmp(argv[3],"Vulkan")))return 2;
    s3d_body_model *model=NULL;int result=0;
    if(fputs("READY\n",stdout)==EOF || fflush(stdout))return 1;
    for(;;){
        char input[4097],output[4097];int got=read_path(input,1);
        if(!got)break;
        if(got<0 || read_path(output,0)!=1){fprintf(stderr,"invalid worker request frame\n");result=2;break;}
        char *args[]={argv[0],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8],input,output,argv[9],NULL};
        result=infer_main(11,args,&model);
        if(result || fputs("DONE\n",stdout)==EOF || fflush(stdout)){result=1;break;}
    }
    s3d_body_model_free(model);return result;
}
