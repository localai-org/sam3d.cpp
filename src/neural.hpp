#pragma once

#include "ggml-backend.h"
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include "validated_weights.hpp"
#include <memory>
#include <map>

namespace sam3d {
// Internal API: module is an explicit caller-trusted library path, not a model
// filename. No implicit backend discovery, device substitution or CPU fallback.
class neural_session {
public:
    neural_session(const std::string &module, const std::string &kind,
                   uint32_t device_index, uint32_t threads = 1,
                   const std::string &expected_device_description = {},bool require_strict_f32=false);
    ~neural_session();
    neural_session(const neural_session &) = delete;
    neural_session &operator=(const neural_session &) = delete;
    ggml_backend_t backend() const { return backend_; }
    const std::string &description() const { return description_; }
    struct scratch_release {
        neural_session *session;
        void operator()(ggml_backend_buffer_t) const noexcept;
    };
    using scratch_part = std::unique_ptr<ggml_backend_buffer,scratch_release>;
    struct scratch_buffer {
        std::vector<scratch_part> parts;
        explicit operator bool() const { return !parts.empty(); }
        ggml_backend_buffer_t get() const { return parts.empty()?nullptr:parts.front().get(); }
    };
    // Lease a non-overlapping scratch buffer until graph readback completes.
    // The session must outlive all leases; public model calls serialize access.
    scratch_buffer allocate(ggml_context *);
    // Only immutable validated snapshots are eligible for one-time upload.
    ggml_tensor *constant(ggml_context *,const validated_weights &,uint64_t columns,uint64_t rows);
    // Canonical flat storage, reshape-only views for immutable model parameters.
    ggml_tensor *parameter(ggml_context *,const validated_weights &,uint64_t columns,uint64_t rows);
    struct upload { ggml_tensor *tensor; const void *data; size_t bytes; };
    struct download { const ggml_tensor *tensor; void *data; size_t bytes; };
    // Synchronous caller contract, optionally batched through same-device pinned
    // storage. All queued work completes before return (also on exceptions).
    ggml_status compute(ggml_cgraph *,std::span<const upload>,std::span<const download>);
    std::map<std::string,std::vector<float>> evaluate_f32(ggml_cgraph *,
        std::span<const std::pair<ggml_tensor *,std::span<const float>>>,
        const std::map<std::string,ggml_tensor *> &);
    uint64_t batched_transfer_count() const { return transfer_batches_; }
private:
    friend struct scratch_release;
    void recycle(ggml_backend_buffer_t) noexcept;
    struct resident_weight;
    std::vector<std::shared_ptr<resident_weight>> constants_;
    uint64_t constant_bytes_ = 0;
    std::vector<ggml_backend_buffer_t> scratch_;
    ggml_backend_buffer_t transfer_buffer_ = nullptr;
    ggml_backend_buffer_type_t transfer_type_ = nullptr;
    bool transfer_type_checked_ = false;
    uint64_t transfer_batches_ = 0;
    ggml_backend_t backend_ = nullptr;
    std::string description_;
};

struct patch_shape { uint32_t batch, channels, height, width, outputs, patch; };
struct patch_taps {
    std::vector<float> patches; // [B,OH,OW,C*P*P]
    std::vector<float> projection; // [B,OH*OW,D], before bias
    std::vector<float> tokens; // [B,OH*OW,D], after bias
};
// F32 convolution/flatten used by DINOv3 PatchEmbed, no optional patch norm.
patch_taps patch_embed(neural_session &, patch_shape, std::span<const float> image,
                       std::span<const float> weight, std::span<const float> bias,
                       bool capture_all = true,bool bf16 = false);
void validate_patch_shape(patch_shape);
}
