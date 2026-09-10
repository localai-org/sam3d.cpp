#pragma once
#include "dino_block.hpp"

namespace sam3d {
struct decoder_shape {
    uint32_t batch, tokens, context_tokens, token_dim, context_dim, heads, head_dim, hidden;
    bool repeat_pe, skip_first_pe, twoway;
};
void validate_decoder_shape(decoder_shape);
std::vector<std::pair<std::string,uint64_t>> decoder_parameter_sizes(decoder_shape);
// Per-inference immutable device snapshot. Session must outlive this object.
// Reusing it avoids uploading and rescanning the same image/PE at every layer.
// It is not a cross-request cache and never aliases mutable caller memory.
class decoder_image_snapshot {
public:
    decoder_image_snapshot(neural_session &,decoder_shape,std::span<const float> image,std::span<const float> image_pe);
    ~decoder_image_snapshot();
    decoder_image_snapshot(const decoder_image_snapshot &)=delete;
    decoder_image_snapshot &operator=(const decoder_image_snapshot &)=delete;
    bool has_position() const;
    std::pair<ggml_tensor *,ggml_tensor *> views(neural_session &,decoder_shape,ggml_context *) const;
private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
// Original Body TransformerDecoderLayer, evaluation/F32/origin GELU FFN,
// LayerNorm eps=1e-6, no LayerScale. All activations are B,N,D.
// Optional PE may broadcast batch=1. Optional mask is B,N, exactly zero/one.
// This is one layer, NOT PromptableDecoder's head/geometry feedback loop.
named_floats body_decoder_layer(neural_session &, decoder_shape,
    std::span<const float> tokens, std::span<const float> context,
    std::span<const float> token_pe, std::span<const float> context_pe,
    std::span<const float> token_mask, const weight_map &parameters,
    bool capture_all = true,bool return_context = true,
    const decoder_image_snapshot *resident_image = nullptr);
// With resident_image, context/context_pe spans must be empty (not ignored).
}
