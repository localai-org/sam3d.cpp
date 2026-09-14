// Backend handling: one CPU backend plus at most one GPU backend, wired
// through a ggml_backend_sched for transparent op-level fallback.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

namespace sam3d {

class Backend {
  public:
    // Load one explicit GGML backend module. This repository builds GGML
    // backends as runtime modules, so inference never discovers or silently
    // substitutes a backend from the process environment.
    static std::unique_ptr<Backend> create(const std::string& module_path,
                                           const std::string& backend_name,
                                           uint32_t device_index,
                                           int n_threads,
                                           const std::string& expected_description = {});

    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Highest-priority (GPU if present) buffer type for weight preloading.
    ggml_backend_buffer_type_t weights_buffer_type() const;

    bool has_gpu() const { return gpu_backend_ != nullptr; }
    const char* device_name() const;
    const char* backend_name() const { return backend_name_.c_str(); }
    int n_threads() const { return n_threads_; }

    // The optional JSONL trace distinguishes host transfer, graph submission,
    // and the mandatory completion boundary for one graph-owning stage.
    void set_profile_label(std::string label);

    // Build helpers -----------------------------------------------------------
    // alloc(): reset + allocate the graph on the scheduler (inputs can then
    // be filled with set_input_*). run(): compute + synchronize.
    bool alloc(struct ggml_cgraph* graph, std::vector<struct ggml_tensor*> inputs = {});
    bool run(ggml_cgraph* graph);
    bool compute(struct ggml_cgraph* graph, std::vector<struct ggml_tensor*> inputs = {});

    // Copy host data into an input tensor (must be called after alloc()).
    bool set_input_f32(struct ggml_tensor* t, const float* data, size_t n_floats);
    bool set_input_i32(struct ggml_tensor* t, const int32_t* data, size_t n_ints);

    // Read a tensor back to host (handles CPU fallback transparently).
    bool get_tensor_f32(struct ggml_tensor* t, std::vector<float>& out);
    bool get_tensor_i32(struct ggml_tensor* t, std::vector<int32_t>& out);

  private:
    Backend() = default;
    bool cpu_backend_only_ = false;
    bool init(const std::string& module_path, const std::string& backend_name,
              uint32_t device_index, int n_threads,
              const std::string& expected_description);
    void write_profile_jsonl() const;

    struct ProfileStats {
        uint64_t h2d_calls = 0;
        uint64_t h2d_bytes = 0;
        uint64_t h2d_submit_ns = 0;
        uint64_t d2h_calls = 0;
        uint64_t d2h_bytes = 0;
        uint64_t d2h_ns = 0;
        uint64_t graph_runs = 0;
        uint64_t graph_submit_ns = 0;
        uint64_t graph_sync_ns = 0;
        uint64_t graph_total_ns = 0;
    };

    ggml_backend_dev_t gpu_dev_ = nullptr;
    ggml_backend_t gpu_backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    ggml_backend_t compute_backend_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    std::string backend_name_ = "cpu";
    int n_threads_ = 8;
    bool profiling_enabled_ = false;
    std::string profile_path_;
    std::string profile_label_ = "unlabeled";
    ProfileStats profile_;
};

}  // namespace sam3d
