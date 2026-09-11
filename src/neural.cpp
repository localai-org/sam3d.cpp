// DINOv3 patch contract: Copyright (c) Meta Platforms, Inc. and affiliates.
// Adaptation details and DINOv3 terms: NOTICE / LICENSES/DINOv3.md.
#include "neural.hpp"
#include "bf16.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace sam3d {
namespace {
std::mutex registry_mutex;
struct loaded_module {ggml_backend_reg_t reg;bool strict_f32;};
std::map<std::string, loaded_module> modules;
bool strict_vulkan_environment(){
    for(auto key:{"GGML_VK_DISABLE_F16","GGML_VK_DISABLE_COOPMAT","GGML_VK_DISABLE_COOPMAT2"}){
        auto value=std::getenv(key);if(!value || std::string(value)!="1")return false;
    }return true;
}
bool experimental_cm2_environment(){
    auto selected=std::getenv("SAM3D_BF16_COOPMAT2");
    if(!selected || std::string_view(selected)!="1")return false;
    for(auto key:{"GGML_VK_DISABLE_F16"}){
        auto value=std::getenv(key);if(!value || std::string_view(value)!="1")return false;
    }
    // CM2 requires the KHR cooperative-matrix extension/features too. Keep
    // them enabled; the backend handshake below rejects an actual CM1 fallback.
    return std::getenv("GGML_VK_DISABLE_COOPMAT2")==nullptr && std::getenv("GGML_VK_DISABLE_COOPMAT")==nullptr;
}
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}
void finite(std::span<const float> values) {
    require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }),
            "non-finite patch input");
}
std::vector<float> read_tensor(ggml_tensor *tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    finite(values);
    return values;
}
}

neural_session::neural_session(const std::string &module, const std::string &kind,
                               uint32_t index, uint32_t threads,
                               const std::string &expected_device_description,bool require_strict_f32) {
    require(kind == "CPU" || kind == "Vulkan", "backend must be CPU or Vulkan");
    require(threads >= 1 && threads <= 1024, "invalid CPU thread count");
    require(std::filesystem::is_regular_file(module), "backend module does not exist");
    const auto path = std::filesystem::canonical(module).string();
    // GGML's dynamic registry is process-global and not internally synchronized.
    // Serialize discovery/init and keep its modules loaded for all live sessions.
    std::lock_guard lock(registry_mutex);
    auto found = modules.find(path);
    bool strict=strict_vulkan_environment();
    const bool hybrid=experimental_cm2_environment();
    if(kind=="Vulkan" && require_strict_f32){
        require(strict || hybrid,"F32 Vulkan requires strict environment flags or explicitly selected patched BF16/CM2 mode");
        require(found==modules.end() || found->second.strict_f32,"Vulkan was previously loaded without strict F32; use a fresh process");
        require(found!=modules.end() || !ggml_backend_reg_by_name("Vulkan"),"cannot verify externally initialized Vulkan precision; initialize through sam3d first");
    }
    auto reg = found == modules.end() ? ggml_backend_load(path.c_str()) : found->second.reg;
    if (!reg) throw std::runtime_error("cannot load requested backend module");
    if(kind=="Vulkan" && hybrid){
        require(ggml_backend_reg_get_proc_address(reg,"ggml_backend_vk_f32_matmul_cm2_v1")!=nullptr,"experimental BF16/CM2 requires the reviewed build-copy F32 matmul patch");
        strict=true;
    }
    modules.emplace(path, loaded_module{reg,strict});
    require(kind == ggml_backend_reg_name(reg), "loaded module is not requested backend");
    require(index < ggml_backend_reg_dev_count(reg), "requested backend device unavailable");
    auto device = ggml_backend_reg_dev_get(reg, index);
    require(expected_device_description.empty() ||
            expected_device_description == ggml_backend_dev_description(device),
            "selected device does not match required description");
    description_ = std::string(ggml_backend_dev_name(device)) + ": " + ggml_backend_dev_description(device);
    auto setter = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (kind == "CPU" && !setter) throw std::runtime_error("CPU backend lacks thread control");
    backend_ = ggml_backend_dev_init(device, nullptr);
    if (!backend_) throw std::runtime_error("requested backend initialization failed");
    if(kind=="Vulkan" && hybrid){
        auto capability=reinterpret_cast<bool (*)(ggml_backend_t)>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_vk_f32_matmul_cm2_v1"));
        if(!capability(backend_)){
            ggml_backend_free(backend_);backend_=nullptr;
            throw std::runtime_error("experimental precision path requires actual BF16 CM2 support; CM1 fallback is not permitted");
        }
    }
    if (setter) setter(backend_, static_cast<int>(threads));
}

void validate_patch_shape(patch_shape s) {
    require(s.batch >= 1 && s.batch <= 4 && s.channels >= 1 && s.channels <= 4,
            "invalid patch batch/channels");
    require(s.width >= 1 && s.width <= 1024 && s.height >= 1 && s.height <= 1024,
            "invalid patch image size");
    require(s.patch >= 1 && s.patch <= 32 && s.patch <= s.width && s.patch <= s.height,
            "invalid patch size");
    require(s.outputs >= 1 && s.outputs <= 2048, "invalid patch output channels");
    require(uint64_t(s.batch)*(s.height/s.patch)*(s.width/s.patch)*s.outputs <= 32*1024*1024,
            "patch activations exceed diagnostic limit");
}

patch_taps patch_embed(neural_session &session, patch_shape s, std::span<const float> image,
                       std::span<const float> weights, std::span<const float> biases,bool capture_all,bool bf16) {
    validate_patch_shape(s);
    const int64_t k = s.channels*s.patch*s.patch;
    const int64_t tokens = (s.width/s.patch)*(s.height/s.patch);
    require(image.size() == uint64_t(s.batch)*s.channels*s.height*s.width &&
            weights.size() == uint64_t(s.outputs)*k && biases.size() == s.outputs,
            "patch input tensor size mismatch");
    finite(image); finite(weights); finite(biases);
    context_ptr ctx(ggml_init({ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false),
                              nullptr, true}), ggml_free);
    if (!ctx) throw std::bad_alloc();
    auto x = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, s.width, s.height, s.channels, s.batch);
    auto w = ggml_new_tensor_4d(ctx.get(), bf16?GGML_TYPE_BF16:GGML_TYPE_F32, s.patch, s.patch, s.channels, s.outputs);
    auto b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, s.outputs);
    // ggml_conv_2d currently defaults F32 kernels to F16 im2col. Build explicitly
    // to preserve F32 activations at this reference boundary.
    auto patches = ggml_im2col(ctx.get(), w, x, s.patch, s.patch, 0, 0, 1, 1, true, GGML_TYPE_F32);
    ggml_set_name(patches, "patch.im2col");
    auto projected = ggml_mul_mat(ctx.get(), ggml_reshape_2d(ctx.get(), w, k, s.outputs),
                                  ggml_reshape_2d(ctx.get(), patches, k, tokens*s.batch));
    ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
    // cuDNN's upstream BF16 Conv2d materializes the convolution before its
    // separate bias add (unlike Linear). Preserve both BF16 rounding points.
    if(bf16)projected=bf16_round(ctx.get(),projected);
    ggml_set_name(projected, "patch.projection");
    auto output = ggml_add(ctx.get(), projected, b);
    if(bf16)output=bf16_round(ctx.get(),output);
    ggml_set_name(output, "patch.tokens");
    // Diagnostic taps must remain materialized even when a backend fuses ops.
    if(capture_all){ggml_set_output(patches);ggml_set_output(projected);}
    ggml_set_output(output);
    auto graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(session.backend(), node))
            throw std::runtime_error(std::string("requested backend cannot execute ") +
                                     ggml_op_name(node->op) + " (" + node->name + ")");
    }
    auto buffer=session.allocate(ctx.get());
    if (!buffer) throw std::bad_alloc();
    if(bf16){
        auto rounded_image=bf16_values(image),rounded_bias=bf16_values(biases);
        ggml_backend_tensor_set(x,rounded_image.data(),0,ggml_nbytes(x));
        set_bf16_tensor(w,weights);
        ggml_backend_tensor_set(b,rounded_bias.data(),0,ggml_nbytes(b));
    }else{
        ggml_backend_tensor_set(x, image.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(b, biases.data(), 0, ggml_nbytes(b));
    }
    if (ggml_backend_graph_compute(session.backend(), graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("patch graph computation failed");
    return {capture_all?read_tensor(patches):std::vector<float>{},
            capture_all?read_tensor(projected):std::vector<float>{},read_tensor(output)};
}
}
