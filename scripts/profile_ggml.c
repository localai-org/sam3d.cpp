/* Diagnostic LD_PRELOAD shim: host-call timings, NOT GPU timestamps.
 * Async entries time enqueue only; synchronization is recorded separately.
 * Calls can nest, so inclusive rows are not additive wall-clock stages.
 * Build separately with cc -O2 -shared -fPIC -Iggml/include ... -ldl.
 * Never linked into the inference library. Set SAM3D_GGML_TRACE to a new CSV.
 * Intended for the current serialized inference runner, not concurrent sessions.
 */
#define _GNU_SOURCE
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static FILE *trace;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
__attribute__((constructor)) static void begin(void) {
    const char *path=getenv("SAM3D_GGML_TRACE");
    if(path) { trace=fopen(path,"wx"); if(!trace) abort();
        fprintf(trace,"start,end,operation,bytes,caller,library,offset\n"); }
}
__attribute__((destructor)) static void end(void) { if(trace) fclose(trace); }
static void record(double start,const char *op,size_t bytes,void *caller) {
    const double finish=now(); Dl_info info={0}; dladdr(caller,&info);
    if(trace) fprintf(trace,"%.9f,%.9f,%s,%zu,%s,%s,0x%zx\n",start,finish,op,bytes,
        info.dli_sname?info.dli_sname:"unknown",info.dli_fname?info.dli_fname:"unknown",
        (size_t)((uintptr_t)caller-(uintptr_t)info.dli_fbase));
}
#define REAL(name) static __typeof__(&name) real; if(!real) { *(void **)(&real)=dlsym(RTLD_NEXT,#name); if(!real) abort(); } double start=now()
#define RECORD(op,bytes) record(start,op,bytes,__builtin_return_address(0))
void ggml_backend_tensor_set(struct ggml_tensor *t,const void *data,size_t offset,size_t size) {
    REAL(ggml_backend_tensor_set); real(t,data,offset,size); RECORD("upload",size);
}
void ggml_backend_tensor_get(const struct ggml_tensor *t,void *data,size_t offset,size_t size) {
    REAL(ggml_backend_tensor_get); real(t,data,offset,size); RECORD("download",size);
}
void ggml_backend_tensor_set_async(ggml_backend_t b,struct ggml_tensor *t,const void *data,size_t offset,size_t size) {
    REAL(ggml_backend_tensor_set_async); real(b,t,data,offset,size); RECORD("upload_async",size);
}
void ggml_backend_tensor_get_async(ggml_backend_t b,const struct ggml_tensor *t,void *data,size_t offset,size_t size) {
    REAL(ggml_backend_tensor_get_async); real(b,t,data,offset,size); RECORD("download_async",size);
}
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t b,struct ggml_cgraph *g) {
    REAL(ggml_backend_graph_compute_async); enum ggml_status result=real(b,g); RECORD("compute_async",0); return result;
}
void ggml_backend_synchronize(ggml_backend_t b) {
    REAL(ggml_backend_synchronize); real(b); RECORD("synchronize",0);
}
void ggml_backend_dev_get_props(ggml_backend_dev_t d,struct ggml_backend_dev_props *p) {
    REAL(ggml_backend_dev_get_props); real(d,p); RECORD("device_properties",0);
}
enum ggml_status ggml_backend_graph_compute(ggml_backend_t b,struct ggml_cgraph *g) {
    REAL(ggml_backend_graph_compute); enum ggml_status result=real(b,g); RECORD("compute",0); return result;
}
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context *c,ggml_backend_t b) {
    REAL(ggml_backend_alloc_ctx_tensors); ggml_backend_buffer_t result=real(c,b); RECORD("allocate",0); return result;
}
void ggml_backend_buffer_free(ggml_backend_buffer_t b) {
    REAL(ggml_backend_buffer_free); real(b); RECORD("free",0);
}
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t type,size_t bytes) {
    REAL(ggml_backend_buft_alloc_buffer); ggml_backend_buffer_t result=real(type,bytes); RECORD("allocate_raw",bytes); return result;
}
