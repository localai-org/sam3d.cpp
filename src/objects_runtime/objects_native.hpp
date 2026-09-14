// SAM 3D Objects inference on ggml — internal C++23 runtime surface.
//
// The library loads GGUF exports of the official SAM 3D Objects generative
// checkpoints and runs the conditioned native graph:
//   DINOv2 + PointPatchEmbed conditioners -> SparseStructure DiT -> SS decoder
//   -> SLat DiT -> geometry or Gaussian decoder -> GLB/PLY export.
//
// The CLI accepts either a masked RGBA scene and runs native MoGe, or a
// directory of captured condition tensors. Official rendering and textured
// GLB postprocessing remain outside this C++ API.
#ifndef SAM3D_GGML_H
#define SAM3D_GGML_H

#include <memory>
#include <string>
#include <vector>

namespace sam3d {

struct CliOptions {
    // model directory containing the converted GGUF files
    std::string models_dir = "cpp_ggml/models/gguf";
    // Explicit dynamically loaded backend module and device. Supported names
    // are CPU and Vulkan; no backend substitution is performed.
    std::string backend_module;
    std::string backend = "CPU";
    uint32_t backend_device = 0;
    std::string expected_device_description;
    int n_threads = 8;
    // One homogeneous converted family for the initial parity path.
    std::string model_dtype = "f16";
    int ss_steps = 25;
    int slat_steps = 25;
    int ss_cfg_strength = 7;
    float ss_cfg_strength_pm = 0.0f;
    int slat_cfg_strength = 1;
    int seed = 42;
    // Native masked-image input. The compact S3DOBJ01 file stores width,
    // height and tightly packed RGBA8 pixels; alpha is the selected object.
    std::string input_rgba;
    std::string pointmap_bin;
    // Directory produced by scripts/dump_e2e_stages.py.  This is the
    // deterministic, parity-testable input form for the C++ session.
    std::string condition_dir;
    // Reserved source-image metadata for future encoded-image entry points.
    std::string image_png;
    std::string mask_png;
    // outputs
    std::string out_ply;     // gaussian splat PLY output
    std::string out_glb;     // FlexiCubes geometry with learned vertex colour
    std::string out_gs_bin;  // raw gaussian parameters for python-side rendering
    std::string out_meta;    // json with timing + counts
    std::string dump_dir;    // if set: dump intermediate tensors for parity
    // benchmarking
    bool bench = false;
    int warmup = 5;
    int iters = 20;
};

struct RunStats {
    double load_ms = 0;
    double pointmap_ms = 0;
    double condition_ms = 0;
    double ss_flow_ms = 0;
    double ss_decode_ms = 0;
    double slat_flow_ms = 0;
    double gs_decode_ms = 0;
    double mesh_decode_ms = 0;
    int n_active_voxels = 0;
    int n_gaussians = 0;
    int n_mesh_vertices = 0;
    int n_mesh_faces = 0;
};

struct RunOutput {
    RunStats stats;
    std::vector<int> coords;   // n_voxels * 3
    std::vector<float> gs;     // n_gaussians packed attributes (see gs_io.h)
};

// Run the full pipeline. Returns nullptr-terminated error string on failure
// inside the returned object (ok == false).
struct RunResult {
    bool ok = false;
    std::string error;
    RunOutput output;
};

// Execute the full native conditioned generation graph and write either a
// geometry GLB or Gaussian PLY.
// Exactly one of opts.input_rgba and opts.condition_dir is required.
RunResult run_pipeline(const CliOptions& opts);

// Version of the graph format emitted by the converter understood here.
inline constexpr const char* kGraphFormatVersion = "1";

}  // namespace sam3d

#endif  // SAM3D_GGML_H
