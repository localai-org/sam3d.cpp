#include "gguf_loader.hpp"

#include <cstring>

#include "common.hpp"

namespace sam3d {

GGUFModel::~GGUFModel() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (meta_ctx_) ggml_free(meta_ctx_);
    if (gguf_) gguf_free(gguf_);
}

bool GGUFModel::load(const std::string& path, ggml_backend_buffer_type_t buft) {
    path_ = path;
    struct gguf_init_params params = {
        /*.no_alloc   =*/ true,
        /*.ctx        =*/ &meta_ctx_,
    };
    gguf_ = gguf_init_from_file(path.c_str(), params);
    if (!gguf_) {
        LOGE("failed to load GGUF: %s", path.c_str());
        return false;
    }

    const size_t n_tensors = gguf_get_n_tensors(gguf_);
    for (size_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(meta_ctx_, name);
        if (!t) {
            LOGE("tensor '%s' missing from context", name);
            return false;
        }
        tensors_[name] = t;
    }

    // Allocate one device buffer for every weight, then copy from the file.
    // alloc_ctx_tensors wires each tensor's data pointer into the buffer.
    const size_t total = ggml_get_mem_size(meta_ctx_);
    buf_ = ggml_backend_alloc_ctx_tensors_from_buft(meta_ctx_, buft);
    if (!buf_) {
        LOGE("failed to allocate %zu bytes for weights of %s", total, path.c_str());
        return false;
    }

    // Read tensor data straight from the file into the device buffer.
    // gguf_get_tensor_offset is relative to the data section, which itself
    // starts at gguf_get_data_offset in the file.
    const size_t data_base = gguf_get_data_offset(gguf_);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> chunk;
    for (size_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = tensors_[name];
        const size_t offset = gguf_get_tensor_offset(gguf_, i);
        const size_t nbytes = ggml_nbytes(t);
        chunk.resize(nbytes);
        if (fseek(f, (long)(data_base + offset), SEEK_SET) != 0 ||
            fread(chunk.data(), 1, nbytes, f) != nbytes) {
            LOGE("failed to read tensor '%s' from %s", name, path.c_str());
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(t, chunk.data(), 0, nbytes);
    }
    fclose(f);
    ggml_backend_buffer_set_usage(buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    LOGD("loaded %s: %zu tensors, %.1f MB", path.c_str(), n_tensors, total / 1.0e6);
    return true;
}

ggml_tensor* GGUFModel::get(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

std::string GGUFModel::str(const std::string& key, const std::string& def) const {
    const int iid = gguf_find_key(gguf_, key.c_str());
    if (iid < 0) return def;
    const enum gguf_type type = gguf_get_kv_type(gguf_, iid);
    if (type != GGUF_TYPE_STRING) return def;
    const char* v = gguf_get_val_str(gguf_, iid);
    return v ? std::string(v) : def;
}

uint32_t GGUFModel::u32(const std::string& key, uint32_t def) const {
    const int iid = gguf_find_key(gguf_, key.c_str());
    if (iid < 0) return def;
    switch (gguf_get_kv_type(gguf_, iid)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf_, iid);
        case GGUF_TYPE_INT32: return (uint32_t)gguf_get_val_i32(gguf_, iid);
        case GGUF_TYPE_UINT64: return (uint32_t)gguf_get_val_u64(gguf_, iid);
        case GGUF_TYPE_INT64: return (uint32_t)gguf_get_val_i64(gguf_, iid);
        default: return def;
    }
}

int32_t GGUFModel::i32(const std::string& key, int32_t def) const {
    return (int32_t)u32(key, (uint32_t)def);
}

float GGUFModel::f32(const std::string& key, float def) const {
    const int iid = gguf_find_key(gguf_, key.c_str());
    if (iid < 0) return def;
    if (gguf_get_kv_type(gguf_, iid) == GGUF_TYPE_FLOAT32) {
        return gguf_get_val_f32(gguf_, iid);
    }
    return def;
}

std::vector<std::string> GGUFModel::str_array(const std::string& key) const {
    std::vector<std::string> out;
    const int iid = gguf_find_key(gguf_, key.c_str());
    if (iid < 0) return out;
    if (gguf_get_kv_type(gguf_, iid) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf_, iid);
    for (size_t i = 0; i < n; i++) {
        const char* v = gguf_get_arr_str(gguf_, iid, i);
        out.emplace_back(v ? v : "");
    }
    return out;
}

std::vector<int32_t> GGUFModel::i32_array(const std::string& key) const {
    std::vector<int32_t> out;
    const int iid = gguf_find_key(gguf_, key.c_str());
    if (iid < 0) return out;
    if (gguf_get_kv_type(gguf_, iid) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf_, iid);
    const enum gguf_type et = gguf_get_arr_type(gguf_, iid);
    if (et == GGUF_TYPE_UINT32 || et == GGUF_TYPE_INT32) {
        const void* data = gguf_get_arr_data(gguf_, iid);
        const int32_t* vals = (const int32_t*)data;
        for (size_t i = 0; i < n; i++) out.push_back(vals[i]);
    }
    return out;
}

size_t GGUFModel::n_tensors() const { return tensors_.size(); }

}  // namespace sam3d
