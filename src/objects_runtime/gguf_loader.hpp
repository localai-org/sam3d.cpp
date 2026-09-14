// GGUF model loading with optional GPU preloading of weights.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

namespace sam3d {

class GGUFModel {
  public:
    GGUFModel() = default;
    ~GGUFModel();

    GGUFModel(const GGUFModel&) = delete;
    GGUFModel& operator=(const GGUFModel&) = delete;

    // Load GGUF and copy all tensors into `buft` (pass the GPU buffer type to
    // preload weights on the device, or CPU buffer type to keep them local).
    bool load(const std::string& path, ggml_backend_buffer_type_t buft);

    ggml_tensor* get(const std::string& name) const;
    bool has(const std::string& name) const { return get(name) != nullptr; }

    // metadata accessors (return defaults when missing)
    std::string str(const std::string& key, const std::string& def = "") const;
    uint32_t u32(const std::string& key, uint32_t def = 0) const;
    int32_t i32(const std::string& key, int32_t def = 0) const;
    float f32(const std::string& key, float def = 0.f) const;
    std::vector<std::string> str_array(const std::string& key) const;
    std::vector<int32_t> i32_array(const std::string& key) const;

    size_t n_tensors() const;
    const std::string& path() const { return path_; }

  private:
    std::string path_;
    gguf_context* gguf_ = nullptr;
    ggml_context* meta_ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};

}  // namespace sam3d
