#include "backend.hpp"

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <map>
#include <mutex>

#include "common.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

namespace sam3d {

namespace {

using ProfileClock = std::chrono::steady_clock;

uint64_t elapsed_ns(ProfileClock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        ProfileClock::now() - start).count());
}

std::mutex registry_mutex;
std::map<std::string, ggml_backend_reg_t> registries;

}  // namespace

std::unique_ptr<Backend> Backend::create(const std::string& module_path,
                                         const std::string& backend_name,
                                         uint32_t device_index,
                                         int n_threads,
                                         const std::string& expected_description) {
    auto b = std::unique_ptr<Backend>(new Backend());
    if (!b->init(module_path, backend_name, device_index, n_threads,
                 expected_description)) return nullptr;
    return b;
}

bool Backend::init(const std::string& module_path, const std::string& backend_name,
                   uint32_t device_index, int n_threads,
                   const std::string& expected_description) {
    n_threads_ = n_threads;
    backend_name_ = backend_name;
    if (const char* profile_path = getenv("SAM3D_E2E_PROFILE_JSONL")) {
        profile_path_ = profile_path;
        profiling_enabled_ = !profile_path_.empty();
    }

    if (backend_name != "CPU" && backend_name != "Vulkan") {
        LOGE("backend must be CPU or Vulkan");
        return false;
    }
    if (n_threads < 1 || n_threads > 1024 || module_path.empty() ||
        !std::filesystem::is_regular_file(module_path)) {
        LOGE("invalid backend module or thread count");
        return false;
    }
    const std::string canonical = std::filesystem::canonical(module_path).string();
    ggml_backend_reg_t reg = nullptr;
    {
        std::lock_guard lock(registry_mutex);
        auto found = registries.find(canonical);
        reg = found == registries.end() ? ggml_backend_load(canonical.c_str()) : found->second;
        if (reg) registries.emplace(canonical, reg);
    }
    if (!reg || backend_name != ggml_backend_reg_name(reg)) {
        LOGE("backend module does not provide requested %s registry", backend_name.c_str());
        return false;
    }
    if (device_index >= ggml_backend_reg_dev_count(reg)) {
        LOGE("requested backend device %u is unavailable", device_index);
        return false;
    }
    ggml_backend_dev_t selected = ggml_backend_reg_dev_get(reg, device_index);
    if (!expected_description.empty() &&
        expected_description != ggml_backend_dev_description(selected)) {
        LOGE("selected device does not match required description");
        return false;
    }
    compute_backend_ = ggml_backend_dev_init(selected, nullptr);
    if (!compute_backend_) {
        LOGE("failed to initialize requested backend");
        return false;
    }
    if (backend_name == "Vulkan") {
        gpu_dev_ = selected;
        gpu_backend_ = compute_backend_;
    } else {
        cpu_backend_ = compute_backend_;
    }
    auto setter = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (backend_name == "CPU" && !setter) {
        LOGE("CPU backend lacks thread control");
        return false;
    }
    if (setter) setter(compute_backend_, n_threads);

    // Primary path: a single compute backend (GPU when present) with a
    // gallocr, since every weight lives in that same buffer type. The sched
    // below is kept as an opt-in experiment (SAM3D_USE_SCHED=1); the current
    // ggml release fails its auto-realloc path on the very large conv3d
    // graphs of the SS decoder.
    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(compute_backend_));
    if (!gallocr_) {
        LOGE("failed to create graph allocator");
        return false;
    }
    // The supported path is single-backend execution. Every graph operation
    // is checked by the selected backend; there is no implicit CPU fallback.
    LOGI("backend: %s (%s)", backend_name_.c_str(), device_name());
    return true;
}

Backend::~Backend() {
    write_profile_jsonl();
    if (sched_) ggml_backend_sched_free(sched_);
    if (gallocr_) ggml_gallocr_free(gallocr_);
    if (compute_backend_) ggml_backend_free(compute_backend_);
}

void Backend::set_profile_label(std::string label) {
    if (!label.empty()) profile_label_ = std::move(label);
}

void Backend::write_profile_jsonl() const {
    if (!profiling_enabled_) return;

    std::ofstream output(profile_path_, std::ios::app);
    if (!output) {
        LOGE("failed to open backend profile output: %s", profile_path_.c_str());
        return;
    }
    output << "{\"schema\":\"sam3d.e2e.backend_profile.v1\","
           << "\"label\":\"" << json_escape(profile_label_) << "\","
           << "\"backend\":\"" << json_escape(backend_name_) << "\","
           << "\"device\":\"" << json_escape(std::string(device_name())) << "\","
           << "\"h2d\":{\"calls\":" << profile_.h2d_calls
           << ",\"bytes\":" << profile_.h2d_bytes
           << ",\"submit_ns\":" << profile_.h2d_submit_ns << "},"
           << "\"d2h\":{\"calls\":" << profile_.d2h_calls
           << ",\"bytes\":" << profile_.d2h_bytes
           << ",\"ns\":" << profile_.d2h_ns << "},"
           << "\"graph\":{\"runs\":" << profile_.graph_runs
           << ",\"submit_ns\":" << profile_.graph_submit_ns
           << ",\"sync_ns\":" << profile_.graph_sync_ns
           << ",\"total_ns\":" << profile_.graph_total_ns << "}}\n";
}

ggml_backend_buffer_type_t Backend::weights_buffer_type() const {
    return ggml_backend_get_default_buffer_type(compute_backend_);
}

const char* Backend::device_name() const {
    if (gpu_dev_) return ggml_backend_dev_description(gpu_dev_);
    return "CPU";
}

bool Backend::alloc(ggml_cgraph* graph, std::vector<ggml_tensor*> inputs) {
    (void)inputs;
    if (sched_) {
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_reserve(sched_, graph)) {
            LOGE("sched reserve failed");
            return false;
        }
        if (ggml_backend_sched_alloc_graph(sched_, graph) != GGML_STATUS_SUCCESS) {
            LOGE("sched alloc failed (out of memory?)");
            return false;
        }
        return true;
    }
    if (ggml_gallocr_reserve(gallocr_, graph) == false) {
        LOGE("graph reserve failed (graph too large for memory?)");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(gallocr_, graph)) {
        LOGE("graph alloc failed");
        return false;
    }
    return true;
}

bool Backend::run(ggml_cgraph* graph) {
    const auto run_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_t be = sched_ ? nullptr : compute_backend_;
    if (be) {
        const auto submit_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
        const ggml_status status = ggml_backend_graph_compute_async(be, graph);
        if (profiling_enabled_) profile_.graph_submit_ns += elapsed_ns(submit_started);
        if (status != GGML_STATUS_SUCCESS) {
            LOGE("graph compute failed");
            return false;
        }
        const auto sync_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
        ggml_backend_synchronize(be);
        if (profiling_enabled_) {
            profile_.graph_runs++;
            profile_.graph_sync_ns += elapsed_ns(sync_started);
            profile_.graph_total_ns += elapsed_ns(run_started);
        }
        return true;
    }
    const auto submit_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    const bool submitted = ggml_backend_sched_graph_compute_async(sched_, graph);
    if (profiling_enabled_) profile_.graph_submit_ns += elapsed_ns(submit_started);
    if (!submitted) {
        LOGE("graph compute failed");
        return false;
    }
    const auto sync_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_sched_synchronize(sched_);
    if (profiling_enabled_) {
        profile_.graph_runs++;
        profile_.graph_sync_ns += elapsed_ns(sync_started);
        profile_.graph_total_ns += elapsed_ns(run_started);
    }
    return true;
}

bool Backend::compute(ggml_cgraph* graph, std::vector<ggml_tensor*> inputs) {
    if (!alloc(graph, inputs)) return false;
    return run(graph);
}

bool Backend::set_input_f32(ggml_tensor* t, const float* data, size_t n_floats) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    if ((size_t)ggml_nelements(t) != n_floats) {
        LOGE("set_input_f32: expected %lld floats, got %zu",
             (long long)ggml_nelements(t), n_floats);
        return false;
    }
    const size_t bytes = n_floats * sizeof(float);
    const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_tensor_set(t, data, 0, bytes);
    if (profiling_enabled_) {
        profile_.h2d_calls++;
        profile_.h2d_bytes += bytes;
        profile_.h2d_submit_ns += elapsed_ns(started);
    }
    return true;
}

bool Backend::set_input_i32(ggml_tensor* t, const int32_t* data, size_t n_ints) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    if ((size_t)ggml_nelements(t) != n_ints) {
        LOGE("set_input_i32: expected %lld ints, got %zu",
             (long long)ggml_nelements(t), n_ints);
        return false;
    }
    const size_t bytes = n_ints * sizeof(int32_t);
    const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_tensor_set(t, data, 0, bytes);
    if (profiling_enabled_) {
        profile_.h2d_calls++;
        profile_.h2d_bytes += bytes;
        profile_.h2d_submit_ns += elapsed_ns(started);
    }
    return true;
}

bool Backend::get_tensor_f32(ggml_tensor* t, std::vector<float>& out) {
    const size_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        if (profiling_enabled_) {
            profile_.d2h_calls++;
            profile_.d2h_bytes += n * sizeof(float);
            profile_.d2h_ns += elapsed_ns(started);
        }
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(n);
        const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
        ggml_backend_tensor_get(t, f16.data(), 0, n * sizeof(ggml_fp16_t));
        if (profiling_enabled_) {
            profile_.d2h_calls++;
            profile_.d2h_bytes += n * sizeof(ggml_fp16_t);
            profile_.d2h_ns += elapsed_ns(started);
        }
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(f16[i]);
        return true;
    }
    LOGE("get_tensor_f32: unsupported tensor type %s", ggml_type_name(t->type));
    out.clear();
    return false;
}

bool Backend::get_tensor_i32(ggml_tensor* t, std::vector<int32_t>& out) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    const size_t n = ggml_nelements(t);
    out.resize(n);
    const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(int32_t));
    if (profiling_enabled_) {
        profile_.d2h_calls++;
        profile_.d2h_bytes += n * sizeof(int32_t);
        profile_.d2h_ns += elapsed_ns(started);
    }
    return true;
}

}  // namespace sam3d
