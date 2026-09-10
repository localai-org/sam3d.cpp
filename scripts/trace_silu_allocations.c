/* Diagnostic only: inspect actual graph allocations before Vulkan fusion.
 * cc -shared -fPIC -Iggml/include scripts/trace_silu_allocations.c -ldl -o TRACE.so
 * LD_PRELOAD=TRACE.so <native inference command>. Never linked into the library.
 */
#define _GNU_SOURCE
#include "ggml-backend.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph *graph) {
    static __typeof__(&ggml_backend_graph_compute) real;
    static int done;
    if (!real) { *(void **)(&real)=dlsym(RTLD_NEXT,"ggml_backend_graph_compute"); if (!real) abort(); }
    if (!done) for (int i=0;i+5<ggml_graph_n_nodes(graph);++i) {
        struct ggml_tensor *t=ggml_graph_node(graph,i);
        if (t->op!=GGML_OP_UNARY || ggml_get_unary_op(t)!=GGML_UNARY_OP_SILU) continue;
        done=1;
        for (int k=0;k<6;++k) {
            t=ggml_graph_node(graph,i+k);
            fprintf(stderr,"SILU_ALLOC node=%d op=%s name=%s buffer=%p data=%p bytes=%zu flags=%d\n",
                k,ggml_op_name(t->op),t->name,(void *)t->buffer,t->data,ggml_nbytes(t),t->flags);
            for (int s=0;s<GGML_MAX_SRC;++s) if(t->src[s]) {
                struct ggml_tensor *a=t->src[s];
                fprintf(stderr,"SILU_ALLOC node=%d src=%d op=%s name=%s buffer=%p data=%p bytes=%zu self=%d\n",
                    k,s,ggml_op_name(a->op),a->name,(void *)a->buffer,a->data,ggml_nbytes(a),a==t);
            }
        }
        break;
    }
    return real(backend,graph);
}
