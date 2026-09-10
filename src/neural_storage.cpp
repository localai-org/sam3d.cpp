#include "neural.hpp"
#include "ggml-alloc.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sam3d {
struct neural_session::resident_weight {
    std::shared_ptr<const std::vector<float>> values;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context{nullptr,ggml_free};
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer{nullptr,ggml_backend_buffer_free};
    ggml_tensor *tensor=nullptr;
};

neural_session::~neural_session() {
    if(backend_)ggml_backend_synchronize(backend_);
    constants_.clear();
    if(transfer_buffer_)ggml_backend_buffer_free(transfer_buffer_);
    for(auto buffer:scratch_)ggml_backend_buffer_free(buffer);
    if(backend_)ggml_backend_free(backend_);
}
void neural_session::scratch_release::operator()(ggml_backend_buffer_t buffer) const noexcept {
    session->recycle(buffer);
}
void neural_session::recycle(ggml_backend_buffer_t buffer) noexcept {
    // All primitive graph execution/readback is synchronous. Retain a bounded
    // pool; live/nested leases are removed from it until their readbacks end.
    size_t bytes=ggml_backend_buffer_get_size(buffer);
    for(auto other:scratch_)bytes+=ggml_backend_buffer_get_size(other);
    if(bytes<=2ULL*1024*1024*1024){
        try{scratch_.push_back(buffer);return;}catch(...){/* noexcept deleter */}
    }
    ggml_backend_buffer_free(buffer);
}
neural_session::scratch_buffer neural_session::allocate(ggml_context *context) {
    if(!context || !ggml_get_no_alloc(context))throw std::invalid_argument("scratch requires a no-alloc context");
    auto type=ggml_backend_get_default_buffer_type(backend_);
    const auto bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(context,type);
    constexpr size_t limit=2ULL*1024*1024*1024;
    if(bytes>limit)throw std::invalid_argument("scratch graph exceeds 2 GiB memory budget");
    const auto maximum=ggml_backend_buft_get_max_size(type),alignment=ggml_backend_buft_get_alignment(type);
    struct chunk {size_t bytes=0;std::vector<ggml_tensor *> tensors;};
    std::vector<chunk> chunks(1);
    // Follow GGML's context allocation layout (ggml-alloc.c, MIT): partition
    // on backend buffer limits, allocate roots, then initialize their views.
    for(auto *t=ggml_get_first_tensor(context);t;t=ggml_get_next_tensor(context,t))if(!t->data && !t->view_src){
        auto size=ggml_backend_buft_get_alloc_size(type,t);
        if(size>limit-alignment)throw std::invalid_argument("scratch tensor too large");
        size=(size+alignment-1)/alignment*alignment;
        if(chunks.back().bytes && chunks.back().bytes+size>maximum)chunks.emplace_back();
        chunks.back().bytes+=size;chunks.back().tensors.push_back(t);
    }
    scratch_buffer result;
    for(auto &chunk:chunks){
        auto best=scratch_.end();
        for(auto it=scratch_.begin();it!=scratch_.end();++it)
            if(ggml_backend_buffer_get_size(*it)>=chunk.bytes &&
               (best==scratch_.end() || ggml_backend_buffer_get_size(*it)<ggml_backend_buffer_get_size(*best)))best=it;
        ggml_backend_buffer_t buffer=nullptr;
        if(best!=scratch_.end()){buffer=*best;scratch_.erase(best);}
        else {
            // Discard undersized unused buffers before growing the workspace.
            for(auto old:scratch_)ggml_backend_buffer_free(old);scratch_.clear();
            buffer=ggml_backend_buft_alloc_buffer(type,std::max(chunk.bytes,size_t(1)));
        }
        if(!buffer)throw std::bad_alloc();
        scratch_part lease(buffer,scratch_release{this});
        ggml_backend_buffer_reset(buffer);auto allocator=ggml_tallocr_new(buffer);
        for(auto *t:chunk.tensors)if(ggml_tallocr_alloc(&allocator,t)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("scratch tensor allocation failed");
        result.parts.push_back(std::move(lease));
    }
    for(auto *t=ggml_get_first_tensor(context);t;t=ggml_get_next_tensor(context,t)) {
        ggml_status status=GGML_STATUS_SUCCESS;
        if(t->view_src && !t->buffer)status=ggml_backend_view_init(t);
        if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("scratch tensor allocation failed");
    }
    return result;
}
ggml_tensor *neural_session::constant(ggml_context *context,const validated_weights &weights,uint64_t columns,uint64_t rows) {
    if(!context || !columns || !rows || columns>INT64_MAX || rows>INT64_MAX ||
       columns>weights.values().size()/rows || columns*rows!=weights.values().size())
        throw std::invalid_argument("immutable weight dimensions mismatch");
    for(auto &entry:constants_)if(entry->values==weights.data_){
        if(entry->tensor->ne[0]!=int64_t(columns) || entry->tensor->ne[1]!=int64_t(rows))
            throw std::invalid_argument("immutable weight reused with different layout");
        return ggml_view_tensor(context,entry->tensor);
    }
    const auto bytes=weights.values().size_bytes();
    constexpr uint64_t limit=1024ULL*1024*1024;
    if(bytes>limit-constant_bytes_)throw std::invalid_argument("resident weights exceed device cache budget");
    auto entry=std::make_shared<resident_weight>();entry->values=weights.data_;
    entry->context.reset(ggml_init({ggml_tensor_overhead()*2,nullptr,true}));
    if(!entry->context)throw std::bad_alloc();
    entry->tensor=ggml_new_tensor_2d(entry->context.get(),GGML_TYPE_F32,columns,rows);
    entry->buffer.reset(ggml_backend_alloc_ctx_tensors(entry->context.get(),backend_));
    if(!entry->buffer)throw std::bad_alloc();
    ggml_backend_buffer_set_usage(entry->buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(entry->tensor,weights.values().data(),0,bytes);
    auto *tensor=entry->tensor;constants_.push_back(std::move(entry));constant_bytes_+=bytes;
    return ggml_view_tensor(context,tensor);
}
ggml_tensor *neural_session::parameter(ggml_context *context,const validated_weights &weights,uint64_t columns,uint64_t rows){
    if(!context || !columns || !rows || columns>INT64_MAX || rows>INT64_MAX ||
       columns>weights.size()/rows || columns*rows!=weights.size())
        throw std::invalid_argument("parameter dimensions mismatch");
    auto flat=constant(context,weights,weights.size(),1);
    return ggml_reshape_2d(context,flat,columns,rows);
}

ggml_status neural_session::compute(ggml_cgraph *graph,std::span<const upload> inputs,std::span<const download> outputs){
    if(!graph)throw std::invalid_argument("null compute graph");
    // Check every range before submitting anything. Oversized valid transfers
    // use the original synchronous path instead of growing a pinned pool.
    constexpr size_t limit=32ULL*1024*1024,alignment=64;
    size_t total=0;
    auto check=[&](const ggml_tensor *t,const void *data,size_t bytes){
        if(!t || !t->data || !t->buffer || !ggml_is_contiguous(t) || bytes!=ggml_nbytes(t) || (bytes && !data))
            throw std::invalid_argument("invalid complete tensor transfer");
        if(bytes>limit || total>limit || bytes>limit-total)total=limit+1;
        else {const auto padded=(bytes+alignment-1)/alignment*alignment;
            if(padded>limit-total)total=limit+1;else total+=padded;}
    };
    for(const auto &t:inputs)check(t.tensor,t.data,t.bytes);
    for(const auto &t:outputs)check(t.tensor,t.data,t.bytes);
    const auto *flag=std::getenv("SAM3D_BATCHED_TRANSFERS");
    ggml_backend_buffer_type_t host=nullptr;
    if(flag && std::strcmp(flag,"1")==0 && total && total<=limit){
        if(!transfer_type_checked_){
            auto device=ggml_backend_get_device(backend_);ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(device,&props);
            if(props.caps.async && props.caps.host_buffer){
                transfer_type_=ggml_backend_dev_host_buffer_type(device);
                // GGML's Vulkan host type currently belongs to device zero. Do
                // not accidentally pin on another device or assume cross-device IO.
                if(transfer_type_ && (ggml_backend_buft_get_device(transfer_type_)!=device || !ggml_backend_buft_is_host(transfer_type_)))transfer_type_=nullptr;
            }
            transfer_type_checked_=true;
        }
        host=transfer_type_;
    }
    auto synchronous=[&]{
        for(const auto &t:inputs)ggml_backend_tensor_set(t.tensor,t.data,0,t.bytes);
        const auto status=ggml_backend_graph_compute(backend_,graph);
        if(status==GGML_STATUS_SUCCESS)for(const auto &t:outputs)ggml_backend_tensor_get(t.tensor,t.data,0,t.bytes);
        return status;
    };
    if(!host || total>ggml_backend_buft_get_max_size(host))return synchronous();
    if(!transfer_buffer_ || ggml_backend_buffer_get_size(transfer_buffer_)<total){
        if(transfer_buffer_){ggml_backend_buffer_free(transfer_buffer_);transfer_buffer_=nullptr;}
        transfer_buffer_=ggml_backend_buft_alloc_buffer(host,total);
        if(!transfer_buffer_)throw std::bad_alloc();
    }
    // A backend may explicitly fall back to ordinary CPU memory on pinned
    // allocation failure. Keep correctness without claiming asynchronous IO.
    if(ggml_backend_buffer_get_type(transfer_buffer_)!=host)return synchronous();
    auto *base=static_cast<std::byte *>(ggml_backend_buffer_get_base(transfer_buffer_));
    if(!base)throw std::runtime_error("pinned transfer buffer is not host accessible");
    struct completion {
        ggml_backend_t backend;
        ~completion(){if(backend)ggml_backend_synchronize(backend);}
        void finish(){ggml_backend_synchronize(backend);backend=nullptr;}
    } pending{backend_};
    size_t offset=0;
    for(const auto &t:inputs){
        if(t.bytes)std::memcpy(base+offset,t.data,t.bytes);
        ggml_backend_tensor_set_async(backend_,t.tensor,base+offset,0,t.bytes);
        offset+=(t.bytes+alignment-1)/alignment*alignment;
    }
    // Vulkan's timestamp logger resets its query pool at graph entry and
    // explicitly requires an empty compute context. Async uploads may have
    // opened one. Drain them only for this diagnostic mode; otherwise retain
    // the single submission/synchronization boundary used in production.
    if(std::getenv("GGML_VK_PERF_LOGGER"))ggml_backend_synchronize(backend_);
    const auto status=ggml_backend_graph_compute_async(backend_,graph);
    if(status!=GGML_STATUS_SUCCESS)return status;
    const auto output_start=offset;
    for(const auto &t:outputs){
        ggml_backend_tensor_get_async(backend_,t.tensor,base+offset,0,t.bytes);
        offset+=(t.bytes+alignment-1)/alignment*alignment;
    }
    pending.finish();
    offset=output_start;
    for(const auto &t:outputs){
        if(t.bytes)std::memcpy(t.data,base+offset,t.bytes);
        offset+=(t.bytes+alignment-1)/alignment*alignment;
    }
    ++transfer_batches_;
    return GGML_STATUS_SUCCESS;
}

std::map<std::string,std::vector<float>> neural_session::evaluate_f32(ggml_cgraph *graph,
    std::span<const std::pair<ggml_tensor *,std::span<const float>>> values,
    const std::map<std::string,ggml_tensor *> &taps){
    std::vector<upload> inputs;std::vector<download> outputs;
    std::map<std::string,std::vector<float>> result;
    for(auto &[t,v]:values){
        if(!t || t->type!=GGML_TYPE_F32)throw std::invalid_argument("non-F32 graph input");
        inputs.push_back({t,v.data(),v.size_bytes()});
    }
    for(auto &[name,t]:taps){
        if(!t || t->type!=GGML_TYPE_F32)throw std::invalid_argument("non-F32 graph result");
        auto &v=result[name];v.resize(ggml_nelements(t));outputs.push_back({t,v.data(),v.size()*sizeof(float)});
    }
    if(compute(graph,inputs,outputs)!=GGML_STATUS_SUCCESS)throw std::runtime_error("graph computation failed");
    return result;
}
}
