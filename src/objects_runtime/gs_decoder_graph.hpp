// SLat Gaussian decoder (slat_decoder_gs, gsdec.*) - full decode graph.
// Structure (SLatGaussianDecoder / SparseTransformerBase, attn_mode=swin):
//   SparseLinear(8->768) -> +APE -> 12 x (LayerNorm(no-affine) -> swin
//   windowed self-attention (window 8, shift 4 on odd blocks) -> + ->
//   LayerNorm(no-affine) -> MLP(768->3072 tanh-GELU->768) -> +) ->
//   LayerNorm(no-affine, F.layer_norm eps 1e-5) -> SparseLinear(768->448).
// There is NO adaLN/timestep conditioning and NO sparse conv in this decoder.
// The window partition (fixed coords) lives in GsTables; same-length windows
// are compacted into buckets so each bucket is one batched manual attention
// (the CUDA flash kernel rejects B>1, and manual 4D mul_mat attention is the
// pointpatch-verified path).
// Output raw feats (448, N) ordered [_xyz(96) _features_dc(96) _scaling(96)
// _rotation(128) _opacity(32)] per token, 32 gaussians per token; the
// to_representation host math (tanh offsets, perturbation) lives in gs_io.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph_builder.hpp"
#include "gguf_loader.hpp"
#include "sparse_ops.hpp"

namespace sam3d {

struct GsDecoderGraph {
    GraphContext* g = nullptr;
    const GGUFModel* m = nullptr;
    const GsTables* tb = nullptr;   // host window tables
    std::string debug_stage;        // input_layer | ape | b0_attn | b0_mlp | b0..b11 | out_layer
    std::string prefix = "gsdec";
    bool torso_only = false;        // return the shared sparse-transformer output before GS head

    ggml_tensor* x = nullptr;       // input (8, N) slat feats
    ggml_tensor* coord_i32 = nullptr;  // input (4, N) I32 coords (for APE)
    std::vector<ggml_tensor*> inputs;
    std::vector<std::shared_ptr<std::vector<int32_t>>> table_data;

    // returns {raw_out (448, N)}; to_representation runs host-side
    std::vector<ggml_tensor*> build();
};

}  // namespace sam3d
