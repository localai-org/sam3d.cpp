/* Experimental native SAM 3D Objects conditioned-inference runner.
 *
 * The learned graphs are adapted from Meta's pinned SAM 3D Objects sources
 * and from Asher-1/sam-3d-objects-ggml at revision
 * 1c14b7c3c3e8d9109b943ddc83a0a39c73744246. See NOTICE and the SAM license.
 * This CLI accepts either a raw scene/mask container for end-to-end inference
 * or a parity-stage tensor directory for numerical boundary investigation.
 */
#include "objects_native.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

bool parse_u32(const char *text, uint32_t &value) {
    char *end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno || end == text || *end || parsed > UINT32_MAX) return false;
    value = static_cast<uint32_t>(parsed);
    return true;
}

void usage(const char *program) {
    std::cerr
        << "Usage: " << program
        << " MODULE CPU|Vulkan DEVICE DESCRIPTION|- MODELS_DIR INPUT OUT.glb|OUT.ply THREADS"
           " [f16|f32|q8_0] [SEED]\n"
        << "INPUT is a native S3DOBJ01 RGBA file (alpha selects the object), or a"
           " parity condition directory.\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 9 || argc > 11) {
        usage(argv[0]);
        return 2;
    }
    sam3d::CliOptions options;
    options.backend_module = argv[1];
    options.backend = argv[2];
    if (options.backend != "CPU" && options.backend != "Vulkan") {
        usage(argv[0]);
        return 2;
    }
    if (!parse_u32(argv[3], options.backend_device)) {
        usage(argv[0]);
        return 2;
    }
    if (std::strcmp(argv[4], "-")) options.expected_device_description = argv[4];
    options.models_dir = argv[5];
    if (std::filesystem::is_directory(argv[6])) options.condition_dir = argv[6];
    else options.input_rgba = argv[6];
    const std::filesystem::path output_path = argv[7];
    if (output_path.extension() == ".glb") options.out_glb = output_path.string();
    else if (output_path.extension() == ".ply") options.out_ply = output_path.string();
    else {
        usage(argv[0]);
        return 2;
    }
    uint32_t threads = 0;
    if (!parse_u32(argv[8], threads) || threads < 1 || threads > 1024) {
        usage(argv[0]);
        return 2;
    }
    options.n_threads = static_cast<int>(threads);
    if (argc >= 10) options.model_dtype = argv[9];
    if (options.model_dtype != "f16" && options.model_dtype != "f32" &&
        options.model_dtype != "q8_0") {
        usage(argv[0]);
        return 2;
    }
    if (argc == 11) {
        uint32_t seed = 0;
        if (!parse_u32(argv[10], seed)) {
            usage(argv[0]);
            return 2;
        }
        options.seed = static_cast<int>(seed);
    }
    if (const char *dump = std::getenv("SAM3D_OBJECTS_DUMP_DIR")) options.dump_dir = dump;
    const sam3d::RunResult result = sam3d::run_pipeline(options);
    if (!result.ok) {
        std::cerr << result.error << '\n';
        return 1;
    }
    return 0;
}
