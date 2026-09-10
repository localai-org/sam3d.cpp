#pragma once
#include "dino_block.hpp"
#include "tensor_archive.hpp"
#include <functional>

namespace sam3d {
struct backbone_shape {
    uint32_t batch, height, width, patch, dim, heads, hidden, depth, storage;
};
void validate_backbone_shape(backbone_shape);
std::vector<std::pair<std::string,uint64_t>> backbone_parameter_sizes(backbone_shape);
// Providers return only model parameters, never upstream intermediate features.
using parameter_reader = std::function<validated_weights(const std::string &,uint64_t)>;
// Borrowed tap data is valid only during the callback. Exceptions propagate to
// the internal C++ caller; public C boundaries must contain them.
using backbone_observer = std::function<void(const std::string &,std::span<const float>)>;
using backbone_block_observer = std::function<void(uint32_t,const std::string &,std::span<const float>)>;
// Resident transformer stack. Parameters are checked/uploaded once, without
// retaining a second host copy. One graph keeps all block activations on device.
// Session must outlive this object; callers serialize its use.
class dino_resident_stack {
public:
    dino_resident_stack(neural_session &,backbone_shape,const parameter_reader &,bool bf16=false);
    ~dino_resident_stack();
    dino_resident_stack(const dino_resident_stack &)=delete;
    dino_resident_stack &operator=(const dino_resident_stack &)=delete;
    std::vector<float> run(backbone_shape,std::span<const float>,const backbone_observer &);
    // Complete stack + final normalization/layout on device. Same observer
    // boundaries as the streamed path, without host round-trips between them.
    std::vector<float> run_features(backbone_shape,std::span<const float>,const backbone_observer &);
    // Complete image encoder, including patch convolution and prefix assembly.
    // One device graph avoids patch/token host roundtrips; captures still expose
    // every original boundary. Input is finite normalized RGB NCHW.
    std::vector<float> run_image(backbone_shape,std::span<const float>,const backbone_observer &);
    validated_weights parameter(const std::string &,uint64_t count) const;
    bool bf16() const;
private:
    std::vector<float> execute(backbone_shape,std::span<const float>,const backbone_observer &,bool features);
    std::vector<float> collect(backbone_shape,const backbone_observer &,bool features);
    struct impl;
    std::unique_ptr<impl> impl_;
};
// F32 RGB NCHW -> normalized patch NCHW. No masks/training/extra embeddings.
// Blocks stream sequentially for bounded memory; this is not optimized residency.
std::vector<float> dino_backbone(neural_session &, backbone_shape,
    std::span<const float> image, const parameter_reader &, const backbone_observer & = {},
    const backbone_block_observer & = {},dino_resident_stack *resident=nullptr,bool bf16=false);
// Checked complete H+ archive, fixed official 512x512 single-person contract.
std::vector<float> body_backbone(neural_session &, tensor_archive &,
    std::span<const float> normalized_rgb, const backbone_observer & = {},
    const backbone_block_observer & = {});
}
