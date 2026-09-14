// End-to-end pipeline session: condition embedders -> SS flow sampling ->
// occupancy -> coords -> SLat flow sampling -> mesh/GLB or Gaussian/PLY decode.
//
// All stage graphs are the parity-verified ones (dino_graph / pointpatch_graph
// / ss_flow_graph / ss_decoder_graph / slat_flow_graph / gs_decoder_graph).
// The sampling loops live host-side (Euler integration + CFG blending); the
// graphs are built once per stage and re-run with fresh inputs.
//
// Inputs (first milestone): the preprocessed condition tensors dumped by
// scripts/dump_e2e_stages.py (ss_input_*.samt). Noise defaults to a
// deterministic C++ PRNG; --noise-dir replays the torch reference noise.
#include "dino_graph.hpp"
#include "moge_graph.hpp"
#include "pointpatch_graph.hpp"
#include "ss_flow_graph.hpp"
#include "ss_decoder_graph.hpp"
#include "slat_flow_graph.hpp"
#include "sparse_ops.hpp"
#include "gs_decoder_graph.hpp"
#include "mesh_decoder_graph.hpp"
#include "mesh_extractor.hpp"
#include "mesh_io.hpp"
#include "graph_builder.hpp"
#include "gguf_loader.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "objects_native.hpp"
#include "objects_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>
#include <array>
#include <string>
#include <vector>

namespace sam3d {

namespace {

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

struct RawImg {           // HWC f32, ne = [C, W, H] (dino graph convention)
    std::vector<float> data;
    int64_t w = 0, h = 0, c = 0;
};

// torch (1, C, H, W) SAMT dump (ne = [W, H, C], CHW memory) -> HWC buffer
RawImg chw_to_hwc(const RawTensor& t) {
    const int64_t C = t.ne[2], H = t.ne[1], W = t.ne[0];
    GGML_ASSERT(static_cast<size_t>(C * H * W) == t.data.size() / sizeof(float));
    const float* src = (const float*)t.data.data();
    RawImg out;
    out.c = C; out.h = H; out.w = W;
    out.data.resize((size_t)C * H * W);
    for (int64_t y = 0; y < H; y++)
        for (int64_t x = 0; x < W; x++)
            for (int64_t c = 0; c < C; c++)
                out.data[(y * W + x) * C + c] = src[c * H * W + y * W + x];
    return out;
}

// torch (1, C, H, W) SAMT dump kept in CHW order (pointpatch graph convention)
std::vector<float> as_chw(const RawTensor& t) {
    return std::vector<float>((const float*)t.data.data(),
                              (const float*)t.data.data() + t.data.size() / 4);
}

struct Stage {
    std::unique_ptr<Backend> backend;
    std::unique_ptr<GGUFModel> model;

    bool open(const std::string& gguf, const CliOptions& options,
              const char* profile_label = nullptr) {
        backend = Backend::create(options.backend_module, options.backend,
                                  options.backend_device, options.n_threads,
                                  options.expected_device_description);
        if (!backend) return false;
        backend->set_profile_label(profile_label ? profile_label : "unlabeled");
        model = std::make_unique<GGUFModel>();
        return model->load(gguf, backend->weights_buffer_type());
    }
    void close() { model.reset(); backend.reset(); }   // frees GPU memory
};

struct EulerStep {
    float t = 0.0f;   // model time in [0, 1]
    float dt = 0.0f;  // integration interval to the next model time
};

// Mirrors FlowMatching._prepare_t and ODESolver.solve_iter in the shipped
// Python pipeline. The model receives t * 1000, while Euler integrates over
// the rescaled [0, 1] interval.
std::vector<EulerStep> make_euler_schedule(int steps, float rescale_t) {
    GGML_ASSERT(steps > 0);
    std::vector<float> times(static_cast<size_t>(steps) + 1);
    for (int i = 0; i <= steps; ++i) {
        const double u = static_cast<double>(i) / steps;
        const double t = rescale_t != 0.0f
            ? u / (1.0 + (static_cast<double>(rescale_t) - 1.0) * (1.0 - u))
            : u;
        times[static_cast<size_t>(i)] = static_cast<float>(t);
    }
    std::vector<EulerStep> schedule;
    schedule.reserve(static_cast<size_t>(steps));
    for (int i = 0; i < steps; ++i) {
        schedule.push_back({times[static_cast<size_t>(i)],
                            times[static_cast<size_t>(i + 1)] - times[static_cast<size_t>(i)]});
    }
    return schedule;
}

uint64_t coordinate_key(int32_t b, int32_t x, int32_t y, int32_t z) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 48) |
           (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(y)) << 16) |
           static_cast<uint64_t>(static_cast<uint32_t>(z) & 0xffffu);
}

// Mirrors prune_sparse_structure(..., max_neighbor_axes_dist=1). The source
// occupancy is binary and coordinates are unique, so testing the 3^3
// neighbourhood directly is identical to the Python conv3d count while
// preserving torch.argwhere's input order.
std::vector<int32_t> prune_surface_coords(const std::vector<int32_t>& coords,
                                          int max_neighbor_axes_dist = 1) {
    GGML_ASSERT(coords.size() % 4 == 0);
    std::unordered_set<uint64_t> occupied;
    occupied.reserve(coords.size() / 2);
    for (size_t i = 0; i < coords.size(); i += 4) {
        occupied.insert(coordinate_key(coords[i], coords[i + 1], coords[i + 2], coords[i + 3]));
    }

    std::vector<int32_t> surface;
    surface.reserve(coords.size());
    for (size_t i = 0; i < coords.size(); i += 4) {
        const int32_t b = coords[i];
        bool is_surface = false;
        for (int dx = -max_neighbor_axes_dist; dx <= max_neighbor_axes_dist && !is_surface; ++dx) {
            for (int dy = -max_neighbor_axes_dist; dy <= max_neighbor_axes_dist && !is_surface; ++dy) {
                for (int dz = -max_neighbor_axes_dist; dz <= max_neighbor_axes_dist; ++dz) {
                    if (occupied.find(coordinate_key(b, coords[i + 1] + dx,
                                                      coords[i + 2] + dy,
                                                      coords[i + 3] + dz)) == occupied.end()) {
                        is_surface = true;
                        break;
                    }
                }
            }
        }
        if (is_surface) surface.insert(surface.end(), coords.begin() + static_cast<ptrdiff_t>(i),
                                       coords.begin() + static_cast<ptrdiff_t>(i + 4));
    }
    return surface;
}

// Match sam3d_objects.pipeline.inference_utils.downsample_sparse_structure.
// The sparse DiT uses int32 coordinate tables, so keeping this policy in the
// C++ runtime is required for large (dense) occupancy fields.
std::vector<int32_t> downsample_coords(const std::vector<int32_t>& coords,
                                       int64_t max_coords = 42000,
                                       int downsample_factor = 2,
                                       uint32_t seed = 42) {
    const int64_t n = static_cast<int64_t>(coords.size() / 4);
    if (n <= max_coords) return coords;
    if (n == 0) return {};

    std::array<int32_t, 3> lo{coords[1], coords[2], coords[3]};
    std::array<int32_t, 3> hi = lo;
    for (int64_t i = 1; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            const int32_t v = coords[static_cast<size_t>(i) * 4 + c + 1];
            lo[c] = std::min(lo[c], v);
            hi[c] = std::max(hi[c], v);
        }
    }
    std::array<double, 3> original{}, target{}, target_min{}, target_max{};
    std::array<int32_t, 3> target_min_i{}, target_max_i{};
    for (int c = 0; c < 3; ++c) {
        original[c] = static_cast<double>(hi[c] - lo[c] + 1);
        target[c] = original[c] / static_cast<double>(downsample_factor);
        const double offset = (original[c] - target[c]) * 0.5;
        target_min[c] = static_cast<double>(lo[c]) + offset;
        target_max[c] = target_min[c] + target[c] - 1.0;
        // torch.Tensor.int() truncates toward zero (rather than floor/ceil).
        target_min_i[c] = static_cast<int32_t>(target_min[c]);
        target_max_i[c] = static_cast<int32_t>(target_max[c]);
    }

    std::vector<int32_t> unique;
    unique.reserve(static_cast<size_t>(n));
    std::unordered_set<uint64_t> seen;
    seen.reserve(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t* p = &coords[static_cast<size_t>(i) * 4];
        int32_t out[4] = {p[0], 0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            const double denom = static_cast<double>(hi[c] - lo[c]);
            const double normalized = denom > 0.0
                ? (static_cast<double>(p[c + 1]) - lo[c]) / denom : 0.0;
            const double value = normalized * (target[c] - 1.0) + target_min[c];
            int64_t rounded = static_cast<int64_t>(std::floor(value + 0.5));
            rounded = std::max<int64_t>(rounded, target_min_i[c]);
            rounded = std::min<int64_t>(rounded, target_max_i[c]);
            out[c + 1] = static_cast<int32_t>(rounded);
        }
        const uint64_t k = coordinate_key(out[0], out[1], out[2], out[3]);
        if (seen.insert(k).second) unique.insert(unique.end(), out, out + 4);
    }
    const size_t unique_n = unique.size() / 4;
    if (unique_n > static_cast<size_t>(max_coords)) {
        std::mt19937 rng(seed);
        std::vector<size_t> order(unique_n);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<int32_t> sampled;
        sampled.reserve(static_cast<size_t>(max_coords) * 4);
        for (size_t j = 0; j < static_cast<size_t>(max_coords); ++j) {
            const int32_t* p = &unique[order[j] * 4];
            sampled.insert(sampled.end(), p, p + 4);
        }
        return sampled;
    }
    return unique;
}

// batched DINO forwards: ONE graph per call, several (prefix, image) pairs
// sharing the same weights model - avoids the multi-alloc CUDA graph-capture
// trouble of running many small graphs on one backend.
std::vector<std::vector<float>> run_dino_batch(
        Stage& st, const std::vector<std::pair<std::string, const RawImg*>>& jobs,
        bool prenorm, int64_t& out_n) {
    GraphContext gctx;
    struct Job { DinoGraph dg; ggml_tensor* in; ggml_tensor* out; };
    std::vector<Job> js;
    for (auto& [prefix, img] : jobs) {
        js.emplace_back();
        js.back().dg.g = &gctx;
        js.back().dg.m = st.model.get();
        js.back().dg.prefix = prefix;
        js.back().dg.prenorm = prenorm;
        js.back().in = gctx.input_f32("image", {img->c, img->w, img->h});
        js.back().dg.inputs.push_back(js.back().in);
        js.back().dg.table_data.push_back(nullptr);
        js.back().out = js.back().dg.build(js.back().in);
    }
    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 32768, false);
    for (auto& j : js) ggml_build_forward_expand(graph, j.out);
    if (!st.backend->alloc(graph)) { out_n = -1; return {}; }
    // upload everything (images + build-internal tables)
    for (size_t k = 0; k < js.size(); k++) {
        const RawImg* img = jobs[k].second;
        if (!st.backend->set_input_f32(js[k].in, img->data.data(),
                                       (size_t)img->c * img->w * img->h)) { out_n = -1; return {}; }
    }
    for (auto& j : js) {
        for (size_t ti = 0; ti < j.dg.inputs.size(); ti++) {
            ggml_tensor* t = j.dg.inputs[ti];
            if (!t->buffer || !j.dg.table_data[ti]) continue;
            const auto& host = j.dg.table_data[ti];
            bool ok = t->type == GGML_TYPE_F32
                ? st.backend->set_input_f32(t, (const float*)host->data(), host->size())
                : st.backend->set_input_i32(t, host->data(), host->size());
            if (!ok) { out_n = -1; return {}; }
        }
    }
    if (!st.backend->run(graph)) { out_n = -1; return {}; }
    std::vector<std::vector<float>> outs(js.size());
    for (size_t k = 0; k < js.size(); k++) st.backend->get_tensor_f32(js[k].out, outs[k]);
    out_n = js[0].out->ne[1];
    return outs;
}

// batched PointPatch forwards: one graph, one run per input pair
std::vector<std::vector<float>> run_pointpatch_batch(
        Stage& st, const std::string& prefix,
        const std::vector<std::pair<const std::vector<float>*, std::pair<int64_t, int64_t>>>& jobs) {
    GraphContext gctx;
    struct Job { PointPatchGraph pp; ggml_tensor* in; ggml_tensor* out; };
    std::vector<Job> js;
    for (auto& [pm, wh] : jobs) {
        js.emplace_back();
        js.back().pp.g = &gctx;
        js.back().pp.m = st.model.get();
        js.back().pp.prefix = prefix;
        js.back().in = gctx.input_f32("pointmap", {wh.first, wh.second, 3});
        js.back().pp.inputs.push_back(js.back().in);
        js.back().pp.table_data.push_back(nullptr);
        js.back().out = js.back().pp.build(js.back().in);
    }
    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 32768, false);
    for (auto& j : js) ggml_build_forward_expand(graph, j.out);
    if (!st.backend->alloc(graph)) return {};
    struct PreparedPointmap {
        std::vector<float> safe;
        std::vector<float> valid;
        std::vector<float> invalid;
    };
    std::vector<PreparedPointmap> prepared(jobs.size());
    for (size_t k = 0; k < js.size(); k++) {
        const auto& source = *jobs[k].first;
        const int64_t src_w = jobs[k].second.first;
        const int64_t src_h = jobs[k].second.second;
        const int64_t src_plane = src_w * src_h;
        auto& host = prepared[k];
        host.safe = source;
        for (int64_t i = 0; i < src_plane; ++i) {
            const bool finite = std::isfinite(source[i]) &&
                                std::isfinite(source[src_plane + i]) &&
                                std::isfinite(source[2 * src_plane + i]);
            if (!finite) {
                host.safe[i] = 0.0f;
                host.safe[src_plane + i] = 0.0f;
                host.safe[2 * src_plane + i] = 0.0f;
            }
        }
        const int64_t dst_count = ggml_nelements(js[k].pp.mask_inputs.front());
        const int64_t dst_side = static_cast<int64_t>(std::llround(std::sqrt(dst_count)));
        if (dst_side * dst_side != dst_count) return {};
        host.valid.resize(dst_count);
        host.invalid.resize(dst_count);
        for (int64_t y = 0; y < dst_side; ++y) {
            const int64_t sy = std::min(src_h - 1, y * src_h / dst_side);
            for (int64_t x = 0; x < dst_side; ++x) {
                const int64_t sx = std::min(src_w - 1, x * src_w / dst_side);
                const int64_t source_i = sy * src_w + sx;
                const int64_t dest_i = y * dst_side + x;
                const float valid = std::isfinite(source[source_i]) &&
                                    std::isfinite(source[src_plane + source_i]) &&
                                    std::isfinite(source[2 * src_plane + source_i]) ? 1.0f : 0.0f;
                host.valid[dest_i] = valid;
                host.invalid[dest_i] = 1.0f - valid;
            }
        }
        if (!st.backend->set_input_f32(js[k].in, host.safe.data(), host.safe.size())) return {};
        for (ggml_tensor* mk : js[k].pp.mask_inputs) {
            if (!mk->buffer) continue;
            const bool is_valid = !strcmp(mk->name, "pp_valid");
            if (!st.backend->set_input_f32(mk,
                                           is_valid ? host.valid.data() : host.invalid.data(),
                                           (size_t)ggml_nelements(mk))) return {};
        }
    }
    // upload the build-internal gather tables (resize indices, cls indices)
    for (auto& j : js) {
        for (size_t ti = 0; ti < j.pp.inputs.size(); ti++) {
            ggml_tensor* t = j.pp.inputs[ti];
            if (!t->buffer || !j.pp.table_data[ti]) continue;
            const auto& host = j.pp.table_data[ti];
            bool ok = t->type == GGML_TYPE_F32
                ? st.backend->set_input_f32(t, (const float*)host->data(), host->size())
                : st.backend->set_input_i32(t, host->data(), host->size());
            if (!ok) return {};
        }
    }
    if (!st.backend->run(graph)) return {};
    std::vector<std::vector<float>> outs(js.size());
    for (size_t k = 0; k < js.size(); k++) st.backend->get_tensor_f32(js[k].out, outs[k]);
    return outs;
}

// EmbedderFuser in-graph: per segment LayerNorm -> SwiGLU -> + idx_emb, concat.
// Segments may share a projection net (same embedder, different inputs).
struct FuserSeg {
    int64_t in_ch;         // token channel count (1024 dino / 512 pointpatch)
    int64_t n;
    int embedder;          // proj prefix index (cemb.emb{E})
    int pos_idx;           // idx_emb column
    std::vector<float> host;
    ggml_tensor* tokens = nullptr;  // created inside build_fuser
};

// builds the fuser subgraph; returns the concat output and fills upload list
ggml_tensor* build_fuser(GraphContext& gctx, GGUFModel& m, std::vector<FuserSeg>& segs,
                         const std::string& cemb_prefix) {
    ggml_context* ctx = gctx.ctx();
    auto as_f32 = [&](ggml_tensor* t) {
        return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
    };
    auto matmul_weight = [&](ggml_tensor* t) {
        return ggml_is_quantized(t->type) ? t : as_f32(t);
    };
    ggml_tensor* idx_emb = as_f32(m.get(cemb_prefix + ".idx_emb"));  // ne = [C, n_groups]
    const int64_t C = idx_emb->ne[0];
    ggml_tensor* cat = nullptr;
    for (auto& s : segs) {
        const std::string p = cemb_prefix + ".emb" + std::to_string(s.embedder);
        ggml_tensor* x = gctx.input_f32("fuse_seg", {s.in_ch, s.n});
        s.tokens = x;
        ggml_tensor* h = gb_layer_norm(ctx, x, as_f32(m.get(p + ".proj_ln.weight")),
                                       as_f32(m.get(p + ".proj_ln.bias")), 1e-5f);
        ggml_tensor* g1 = gb_linear(ctx, matmul_weight(m.get(p + ".proj.w1.weight")), nullptr, h);
        ggml_tensor* g3 = gb_linear(ctx, matmul_weight(m.get(p + ".proj.w3.weight")), nullptr, h);
        ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g1), g3);
        ggml_tensor* o = gb_linear(ctx, matmul_weight(m.get(p + ".proj.w2.weight")), nullptr, act);
        ggml_tensor* pe = ggml_view_1d(ctx, idx_emb, C,
                                       s.pos_idx * C * sizeof(float));
        o = ggml_add(ctx, o, pe);              // (C,1) broadcasts over (C,n)
        cat = cat ? ggml_concat(ctx, cat, o, 1) : o;
    }
    return cat;
}

// load condition inputs dumped by dump_e2e_stages.py
struct CondInputs {
    RawImg image, rgb_image, mask3, rgb_image_mask3;   // HWC (dino)
    std::vector<float> pointmap, rgb_pointmap;         // CHW (pointpatch)
    int64_t W = 518, H = 518;
};

bool load_object_rgba(const std::string& path, std::vector<uint8_t>& rgba,
                      uint32_t& width, uint32_t& height) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    input.read(reinterpret_cast<char*>(&width), sizeof(width));
    input.read(reinterpret_cast<char*>(&height), sizeof(height));
    if (!input || std::string_view(magic.data(), magic.size()) != "S3DOBJ01" ||
        width < 3 || height < 3 || width > 4096 || height > 4096 ||
        static_cast<uint64_t>(width) * height > 16 * 1024 * 1024) return false;
    rgba.resize(static_cast<size_t>(width) * height * 4);
    input.read(reinterpret_cast<char*>(rgba.data()), static_cast<std::streamsize>(rgba.size()));
    char trailing = 0;
    return input && !input.read(&trailing, 1);
}

RawImg chw_to_raw(std::span<const float> chw, int64_t width, int64_t height) {
    GGML_ASSERT(chw.size() == static_cast<size_t>(3 * width * height));
    RawImg result;
    result.w = width; result.h = height; result.c = 3;
    result.data.resize(chw.size());
    const size_t plane = static_cast<size_t>(width * height);
    for (int64_t y = 0; y < height; ++y)
        for (int64_t x = 0; x < width; ++x)
            for (int64_t c = 0; c < 3; ++c)
                result.data[static_cast<size_t>((y * width + x) * 3 + c)] =
                    chw[static_cast<size_t>(c) * plane + static_cast<size_t>(y * width + x)];
    return result;
}

// MoGe v1 fits one camera-space Z translation and focal value against the
// predicted point map. The focal value has a closed-form solution for every
// candidate shift, leaving a stable one-dimensional Gauss-Newton solve.
std::pair<float, float> recover_moge_focal_shift(const std::vector<float>& points,
                                                  const std::vector<float>& mask,
                                                  uint32_t width, uint32_t height) {
    const size_t plane = static_cast<size_t>(width) * height;
    GGML_ASSERT(points.size() == 3 * plane && mask.size() == plane);
    struct Sample { double u, v, x, y, z; };
    std::vector<Sample> samples;
    samples.reserve(64 * 64);
    const double diagonal = std::hypot(static_cast<double>(width), static_cast<double>(height));
    for (uint32_t oy = 0; oy < 64; ++oy) {
        const uint32_t y = std::min(static_cast<uint32_t>(std::floor(oy * (height / 64.0))), height - 1);
        for (uint32_t ox = 0; ox < 64; ++ox) {
            const uint32_t x = std::min(static_cast<uint32_t>(std::floor(ox * (width / 64.0))), width - 1);
            const size_t i = static_cast<size_t>(y) * width + x;
            if (!(mask[i] > 0.5f)) continue;
            const double px = points[i], py = points[plane + i], pz = points[2 * plane + i];
            if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) continue;
            samples.push_back({((x + 0.5) / width * 2.0 - 1.0) * width / diagonal,
                               ((y + 0.5) / height * 2.0 - 1.0) * height / diagonal,
                               px, py, pz});
        }
    }
    if (samples.size() < 2) return {1.0f, 0.0f};
    double shift = 0.0;
    for (int iteration = 0; iteration < 64; ++iteration) {
        double numerator = 0.0, denominator = 0.0, numerator_prime = 0.0;
        double denominator_prime = 0.0;
        bool valid = true;
        for (const auto& p : samples) {
            const double z = p.z + shift;
            if (std::abs(z) < 1e-9) { valid = false; break; }
            const double ax = p.x / z, ay = p.y / z;
            const double axp = -p.x / (z * z), ayp = -p.y / (z * z);
            numerator += ax * p.u + ay * p.v;
            denominator += ax * ax + ay * ay;
            numerator_prime += axp * p.u + ayp * p.v;
            denominator_prime += 2.0 * (ax * axp + ay * ayp);
        }
        if (!valid || denominator < 1e-18) break;
        const double focal = numerator / denominator;
        const double focal_prime =
            (numerator_prime * denominator - numerator * denominator_prime) /
            (denominator * denominator);
        double jr = 0.0, jj = 0.0;
        for (const auto& p : samples) {
            const double z = p.z + shift;
            const double ax = p.x / z, ay = p.y / z;
            const double rx = focal * ax - p.u, ry = focal * ay - p.v;
            const double jx = focal_prime * ax - focal * p.x / (z * z);
            const double jy = focal_prime * ay - focal * p.y / (z * z);
            jr += rx * jx + ry * jy;
            jj += jx * jx + jy * jy;
        }
        if (jj < 1e-18) break;
        double step = -jr / jj;
        if (!std::isfinite(step)) break;
        step = std::clamp(step, -10.0, 10.0);
        const double next = shift + step;
        bool pole = false;
        for (const auto& p : samples) pole = pole || std::abs(p.z + next) < 1e-7;
        if (pole) step *= 0.5;
        shift += step;
        if (std::abs(step) <= 1e-6 * std::max(1.0, std::abs(shift))) break;
    }
    double numerator = 0.0, denominator = 0.0;
    for (const auto& p : samples) {
        const double z = p.z + shift;
        const double ax = p.x / z, ay = p.y / z;
        numerator += ax * p.u + ay * p.v;
        denominator += ax * ax + ay * ay;
    }
    const double focal = denominator > 1e-18 ? numerator / denominator : 1.0;
    return {static_cast<float>(focal), static_cast<float>(shift)};
}

bool infer_cond_inputs(const CliOptions& options, CondInputs& ci) {
    std::vector<uint8_t> rgba;
    uint32_t width = 0, height = 0;
    if (!load_object_rgba(options.input_rgba, rgba, width, height)) {
        LOGE("e2e: invalid S3DOBJ01 input: %s", options.input_rgba.c_str());
        return false;
    }
    const size_t plane = static_cast<size_t>(width) * height;
    std::vector<float> rgb(3 * plane);
    for (size_t i = 0; i < plane; ++i)
        for (size_t c = 0; c < 3; ++c)
            rgb[c * plane + i] = static_cast<float>(rgba[i * 4 + c]) / 255.0f;

    // Official MoGe v1 resolution level 9 selects floor(500000 / 14^2)
    // tokens and truncates both scaled dimensions toward zero.
    constexpr int64_t tokens = 500000 / (14 * 14);
    const double factor = std::sqrt(static_cast<double>(tokens * 14 * 14) /
                                    static_cast<double>(plane));
    const uint32_t resized_width = std::max(14u, static_cast<uint32_t>(width * factor));
    const uint32_t resized_height = std::max(14u, static_cast<uint32_t>(height * factor));
    auto resized_chw = objects_resize_chw(rgb, 3, height, width, resized_height,
                                           resized_width, true);
    RawImg resized = chw_to_raw(resized_chw, resized_width, resized_height);

    Stage moge;
    if (!moge.open(options.models_dir + "/moge_vitl-" + options.model_dtype + ".gguf",
                   options, "moge")) return false;
    GraphContext graph_context;
    ggml_tensor* image = graph_context.input_f32("moge_image", {3, resized_width, resized_height});
    MogeGraph graph_builder;
    graph_builder.g = &graph_context; graph_builder.m = moge.model.get();
    graph_builder.output_width = width; graph_builder.output_height = height;
    MogeOutputs outputs = graph_builder.build(image);
    ggml_cgraph* graph = ggml_new_graph_custom(graph_context.ctx(), 65536, false);
    ggml_set_output(outputs.points); ggml_set_output(outputs.mask_logits);
    ggml_build_forward_expand(graph, outputs.points);
    ggml_build_forward_expand(graph, outputs.mask_logits);
    if (!options.dump_dir.empty()) {
        ggml_set_output(outputs.backbone_input);
        ggml_build_forward_expand(graph, outputs.backbone_input);
        if (!outputs.backbone_block_outputs.empty()) {
            ggml_set_output(outputs.backbone_block_outputs.front());
            ggml_build_forward_expand(graph, outputs.backbone_block_outputs.front());
        }
        ggml_set_output(outputs.projected_features);
        ggml_build_forward_expand(graph, outputs.projected_features);
        for (ggml_tensor* tensor : outputs.upsample_outputs) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
        }
        for (ggml_tensor* tensor : outputs.third_upsample_stages) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
        }
        ggml_set_output(outputs.third_transpose_weight_f32);
        ggml_build_forward_expand(graph, outputs.third_transpose_weight_f32);
        ggml_set_output(outputs.output_block_input);
        ggml_build_forward_expand(graph, outputs.output_block_input);
        for (ggml_tensor* tensor : outputs.output_branch_hidden) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
        }
        for (ggml_tensor* tensor : outputs.output_branch_raw) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
        }
    }
    if (!moge.backend->alloc(graph) ||
        !moge.backend->set_input_f32(image, resized.data.data(), resized.data.size())) return false;
    size_t f32_index = 0, i32_index = 0;
    for (ggml_tensor* tensor : graph_builder.inputs) {
        if (tensor->type == GGML_TYPE_F32 && f32_index < graph_builder.f32_data.size()) {
            const auto& values = *graph_builder.f32_data[f32_index++];
            if (!moge.backend->set_input_f32(tensor, values.data(), values.size())) return false;
        } else if (tensor->type == GGML_TYPE_I32 && i32_index < graph_builder.i32_data.size()) {
            const auto& values = *graph_builder.i32_data[i32_index++];
            if (!moge.backend->set_input_i32(tensor, values.data(), values.size())) return false;
        } else {
            return false;
        }
    }
    if (!moge.backend->run(graph)) return false;
    std::vector<float> points, predicted_mask;
    if (!moge.backend->get_tensor_f32(outputs.points, points) ||
        !moge.backend->get_tensor_f32(outputs.mask_logits, predicted_mask) ||
        points.size() != 3 * plane || predicted_mask.size() != plane) return false;
    const auto [point_min, point_max] = std::minmax_element(points.begin(), points.end());
    const auto [mask_min, mask_max] = std::minmax_element(predicted_mask.begin(), predicted_mask.end());
    const size_t valid_mask = static_cast<size_t>(std::count_if(predicted_mask.begin(),
        predicted_mask.end(), [](float value) { return value > 0.5f; }));
    LOGI("e2e: MoGe raw points=[%.7g,%.7g] mask=[%.7g,%.7g] valid=%zu/%zu",
         *point_min, *point_max, *mask_min, *mask_max, valid_mask, predicted_mask.size());
    if (!options.dump_dir.empty()) {
        auto save_tensor = [&](const std::string& name, ggml_tensor* tensor) {
            std::vector<float> values;
            if (!moge.backend->get_tensor_f32(tensor, values)) return false;
            std::vector<int64_t> shape;
            for (int dim = 0; dim < GGML_MAX_DIMS && tensor->ne[dim] > 1; ++dim) {
                shape.push_back(tensor->ne[dim]);
            }
            if (shape.empty()) shape.push_back(1);
            return save_raw_tensor_f32(options.dump_dir + "/" + name + ".samt",
                                       shape, values.data());
        };
        std::vector<float> backbone_input, block0;
        if (!moge.backend->get_tensor_f32(outputs.backbone_input, backbone_input) ||
            outputs.backbone_block_outputs.empty() ||
            !moge.backend->get_tensor_f32(outputs.backbone_block_outputs.front(), block0) ||
            !save_raw_tensor_f32(options.dump_dir + "/moge_backbone_input.samt",
                                 {outputs.backbone_input->ne[0], outputs.backbone_input->ne[1]},
                                 backbone_input.data()) ||
            !save_raw_tensor_f32(options.dump_dir + "/moge_block0.samt",
                                 {outputs.backbone_block_outputs.front()->ne[0],
                                  outputs.backbone_block_outputs.front()->ne[1]},
                                 block0.data()) ||
            !save_tensor("moge_projected_features", outputs.projected_features) ||
            !save_tensor("moge_upsample_2_weight_f32", outputs.third_transpose_weight_f32) ||
            !save_tensor("moge_output_block_input", outputs.output_block_input)) return false;
        for (size_t i = 0; i < outputs.upsample_outputs.size(); ++i) {
            if (!save_tensor("moge_upsample_" + std::to_string(i),
                             outputs.upsample_outputs[i])) return false;
        }
        for (size_t i = 0; i < outputs.third_upsample_stages.size(); ++i) {
            if (!save_tensor("moge_upsample_2_stage_" + std::to_string(i),
                             outputs.third_upsample_stages[i])) return false;
        }
        for (size_t i = 0; i < outputs.output_branch_hidden.size(); ++i) {
            if (!save_tensor("moge_output_hidden_" + std::to_string(i),
                             outputs.output_branch_hidden[i]) ||
                !save_tensor("moge_output_raw_" + std::to_string(i),
                             outputs.output_branch_raw[i])) return false;
        }
        if (!save_raw_tensor_f32(options.dump_dir + "/moge_points.samt",
                                 {width, height, 3}, points.data()) ||
            !save_raw_tensor_f32(options.dump_dir + "/moge_mask.samt",
                                 {width, height}, predicted_mask.data())) return false;
    }
    moge.close();

    const auto [focal, shift] = recover_moge_focal_shift(points, predicted_mask, width, height);
    LOGI("e2e: MoGe camera fit focal=%.7g shift=%.7g", focal, shift);
    for (size_t i = 0; i < plane; ++i) {
        // PyTorch3D camera convention transform for eye=(0,0,-1), up=(0,-1,0).
        points[i] = -points[i];
        points[plane + i] = -points[plane + i];
        points[2 * plane + i] += shift;
    }
    objects_preprocess_options preprocessing;
    preprocessing.image_side = 518; preprocessing.point_side = 518;
    preprocessing.box_factor = 1.2; preprocessing.padding = 0.0;
    preprocessing.object_normalizer.mode = objects_ssi_mode::object_scene;
    preprocessing.object_normalizer.allow_override = true;
    preprocessing.full_normalizer = preprocessing.object_normalizer;
    const auto prepared = objects_preprocess_pointmap(rgba, width, height, width * 4,
        points, height, width, preprocessing);
    ci.image = chw_to_raw(prepared.at("image"), 518, 518);
    ci.rgb_image = chw_to_raw(prepared.at("rgb_image"), 518, 518);
    ci.mask3 = chw_to_raw(std::vector<float>(prepared.at("mask").size() * 3), 518, 518);
    ci.rgb_image_mask3 = chw_to_raw(
        std::vector<float>(prepared.at("rgb_image_mask").size() * 3), 518, 518);
    auto repeat_mask = [](const std::vector<float>& source, RawImg& target) {
        const size_t count = source.size();
        for (size_t i = 0; i < count; ++i)
            for (size_t c = 0; c < 3; ++c) target.data[i * 3 + c] = source[i];
    };
    repeat_mask(prepared.at("mask"), ci.mask3);
    repeat_mask(prepared.at("rgb_image_mask"), ci.rgb_image_mask3);
    ci.pointmap = prepared.at("pointmap");
    ci.rgb_pointmap = prepared.at("rgb_pointmap");
    ci.W = 518; ci.H = 518;
    return true;
}

bool load_cond_inputs(const std::string& dir, CondInputs& ci) {
    auto load = [&](const char* name, RawTensor& t) {
        if (!load_raw_tensor(dir + "/" + name, t)) {
            LOGE("e2e: missing %s/%s", dir.c_str(), name);
            return false;
        }
        return true;
    };
    RawTensor image, rgb_image, mask, rgb_image_mask, pointmap, rgb_pointmap;
    if (!load("ss_input_image.samt", image)) return false;
    if (!load("ss_input_rgb_image.samt", rgb_image)) return false;
    if (!load("ss_input_mask.samt", mask)) return false;
    if (!load("ss_input_rgb_image_mask.samt", rgb_image_mask)) return false;
    if (!load("ss_input_pointmap.samt", pointmap)) return false;
    if (!load("ss_input_rgb_pointmap.samt", rgb_pointmap)) return false;

    ci.image = chw_to_hwc(image);
    ci.rgb_image = chw_to_hwc(rgb_image);
    ci.W = ci.image.w; ci.H = ci.image.h;
    ci.mask3 = chw_to_hwc(mask);
    ci.rgb_image_mask3 = chw_to_hwc(rgb_image_mask);
    // 1-channel masks: the Dino wrapper repeats them to 3 channels
    auto rep3 = [](RawImg& m) {
        if (m.c != 1) return;
        std::vector<float> m3(m.data.size() * 3);
        for (size_t i = 0; i < m.data.size(); i++)
            for (int c = 0; c < 3; c++) m3[i * 3 + c] = m.data[i];
        m.c = 3; m.data = std::move(m3);
    };
    rep3(ci.mask3);
    rep3(ci.rgb_image_mask3);
    ci.pointmap = as_chw(pointmap);
    ci.rgb_pointmap = as_chw(rgb_pointmap);
    return true;
}

// deterministic normal noise (C++ PRNG); --noise-dir replays torch noise
void fill_normal(std::vector<float>& v, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& x : v) x = nd(rng);
}

bool load_noise_opt(const std::string& dir, const char* name, std::vector<float>& v,
                    int64_t expect_elems) {
    if (dir.empty()) return false;
    RawTensor t;
    if (!load_raw_tensor(dir + "/" + name, t)) return false;
    const size_t n = t.data.size() / 4;
    if ((int64_t)n != expect_elems) {
        LOGE("noise %s: expected %lld elems, got %zu", name, (long long)expect_elems, n);
        return false;
    }
    v.assign((const float*)t.data.data(), (const float*)t.data.data() + n);
    return true;
}

// Replaying an official SLat noise tensor by position, rather than by its
// compact token index, makes an SS-boundary change reproducible. Most target
// cells retain the exact official initial state; cells introduced by the
// backend's occupancy decision receive a stable coordinate-derived state.
bool load_slat_noise_for_coords(const std::string& dir,
                                const std::vector<int32_t>& target_coords,
                                unsigned seed,
                                std::vector<float>& target_noise) {
    if (dir.empty() || target_coords.size() % 4 != 0) return false;
    RawTensor ref_noise;
    RawTensor ref_coords;
    if (!load_raw_tensor(dir + "/slat_x0.samt", ref_noise) ||
        !load_raw_tensor(dir + "/slat_coords.samt", ref_coords) ||
        ref_noise.type != GGML_TYPE_F32 || ref_coords.type != GGML_TYPE_I32) {
        return false;
    }
    const size_t n_ref = ref_noise.data.size() / (8 * sizeof(float));
    if (n_ref == 0 || ref_noise.data.size() != n_ref * 8 * sizeof(float) ||
        ref_coords.data.size() != n_ref * 4 * sizeof(int32_t)) {
        LOGE("slat noise replay: malformed reference support");
        return false;
    }

    const int32_t* ref_coord_data = reinterpret_cast<const int32_t*>(ref_coords.data.data());
    const float* ref_noise_data = reinterpret_cast<const float*>(ref_noise.data.data());
    std::unordered_map<uint64_t, size_t> reference_index;
    reference_index.reserve(n_ref * 2);
    for (size_t i = 0; i < n_ref; ++i) {
        const int32_t* coord = ref_coord_data + 4 * i;
        reference_index.emplace(coordinate_key(coord[0], coord[1], coord[2], coord[3]), i);
    }

    const size_t n_target = target_coords.size() / 4;
    target_noise.resize(n_target * 8);
    size_t replayed = 0;
    for (size_t i = 0; i < n_target; ++i) {
        const int32_t* coord = target_coords.data() + 4 * i;
        const uint64_t key = coordinate_key(coord[0], coord[1], coord[2], coord[3]);
        const auto it = reference_index.find(key);
        float* output = target_noise.data() + 8 * i;
        if (it != reference_index.end()) {
            std::memcpy(output, ref_noise_data + 8 * it->second, 8 * sizeof(float));
            ++replayed;
            continue;
        }
        const uint32_t local_seed = static_cast<uint32_t>(key ^ (key >> 32) ^ seed);
        std::mt19937 local_rng(local_seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (int channel = 0; channel < 8; ++channel) output[channel] = normal(local_rng);
    }
    LOGI("slat noise replay: %zu/%zu cells matched official support", replayed, n_target);
    return true;
}

// --- SLat denorm constants (checkpoints/hf/pipeline.yaml) -----------------
// The inference_utils defaults describe a different checkpoint family. The
// shipped pipeline overrides both vectors, and these are therefore part of
// the model contract rather than generic runtime defaults.
constexpr float SLAT_STD[8] = {2.37326008f, 2.13174402f, 2.24139530f,
                               2.30589401f, 2.11918940f, 1.89695110f,
                               2.41684989f, 2.08374642f};
constexpr float SLAT_MEAN[8] = {0.12211431f, 0.37204156f, -1.26521907f,
                                -2.05276058f, -3.10432536f, -0.11294304f,
                                -0.85146744f, 0.45506954f};

// Gaussian decoder representation constants (slat_decoder_gs.yaml)
constexpr int GS_NGAUSS = 32;
constexpr float GS_VOXEL_SIZE = 1.5f;
constexpr float GS_SCALING_BIAS = 0.004f;
constexpr float GS_OPACITY_BIAS = 0.1f;
constexpr float GS_MIN_KERNEL = 0.0009f;

inline float softplus(float x) { return x > 20.f ? x : logf(1.f + expf(x)); }
inline float softplus_inv(float y) { return y + logf(1.f - expf(-y)); }   // y > 0
inline float inv_sigmoid(float p) { return logf(p / (1.f - p)); }

// write the official 3DGS PLY (matches gaussian_model.save_ply: sh_degree 0)
bool write_gaussian_ply(const std::string& path, int64_t n_gs,
                        const std::vector<float>& xyz, const std::vector<float>& fdc,
                        const std::vector<float>& opacity, const std::vector<float>& scaling,
                        const std::vector<float>& rotation) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    char head[1024];
    snprintf(head, sizeof(head),
             "ply\nformat binary_little_endian 1.0\ncomment NeoGaussian\n"
             "element vertex %lld\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property float nx\nproperty float ny\nproperty float nz\n"
             "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
             "property float opacity\n"
             "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
             "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
             "end_header\n", (long long)n_gs);
    fwrite(head, 1, strlen(head), f);
    std::vector<float> row(3 + 3 + 3 + 1 + 3 + 4);
    const float scale_bias = softplus_inv(GS_SCALING_BIAS);
    const float op_bias = inv_sigmoid(GS_OPACITY_BIAS);
    for (int64_t i = 0; i < n_gs; i++) {
        // PLY xyz = _xyz * aabb[3:] + aabb[:3] with aabb = [-.5,-.5,-.5,1,1,1]
        row[0] = xyz[i * 3 + 0] - 0.5f; row[1] = xyz[i * 3 + 1] - 0.5f;
        row[2] = xyz[i * 3 + 2] - 0.5f;
        row[3] = row[4] = row[5] = 0.f;
        for (int k = 0; k < 3; k++) row[6 + k] = fdc[i * 3 + k];
        // scale = log(sqrt(softplus(s + softplus_inv(bias))^2 + min_kernel^2))
        for (int k = 0; k < 3; k++) {
            float sp = softplus(scaling[i * 3 + k] + scale_bias);
            row[10 + k] = logf(sqrtf(sp * sp + GS_MIN_KERNEL * GS_MIN_KERNEL));
        }
        // opacity = inverse_sigmoid(sigmoid(op + inverse_sigmoid(bias))) = op + bias
        row[9] = opacity[i] + op_bias;
        // rotation = raw * lr + rots_bias [1,0,0,0] (unnormalized in the PLY)
        row[13] = rotation[i * 4 + 0] + 1.f;
        row[14] = rotation[i * 4 + 1];
        row[15] = rotation[i * 4 + 2];
        row[16] = rotation[i * 4 + 3];
        fwrite(row.data(), sizeof(float), row.size(), f);
    }
    fclose(f);
    return true;
}

bool upload_mesh_inputs(Stage& stage, const std::vector<ggml_tensor*>& inputs,
                        const std::vector<std::shared_ptr<std::vector<int32_t>>>& payloads) {
    if (inputs.size() != payloads.size()) return false;
    for (size_t index = 0; index < inputs.size(); ++index) {
        ggml_tensor* input = inputs[index];
        if (!input->buffer || !payloads[index]) continue;
        const auto& values = *payloads[index];
        const bool ok = input->type == GGML_TYPE_F32
            ? stage.backend->set_input_f32(input, reinterpret_cast<const float*>(values.data()),
                                           values.size())
            : stage.backend->set_input_i32(input, values.data(), values.size());
        if (!ok) return false;
    }
    return true;
}

bool decode_object_mesh(Stage& stage, const std::vector<int32_t>& coords,
                        const std::vector<float>& slat_features, ObjectMesh& mesh,
                        const std::string& dump_dir) {
    const int64_t count = static_cast<int64_t>(coords.size() / 4);
    GsTables base_tables;
    if (!base_tables.build(coords.data(), count)) return false;
    std::vector<float> features;
    {
        GraphContext graph_context;
        ggml_tensor* input = graph_context.input_f32("mesh_input", {8, count});
        GsDecoderGraph decoder;
        decoder.g = &graph_context;
        decoder.m = stage.model.get();
        decoder.tb = &base_tables;
        decoder.prefix = "meshdec";
        decoder.torso_only = true;
        decoder.x = input;
        decoder.inputs.push_back(input);
        decoder.table_data.push_back(nullptr);
        auto outputs = decoder.build();
        ggml_cgraph* graph = ggml_new_graph_custom(graph_context.ctx(), 65536, false);
        ggml_set_output(outputs[0]);
        ggml_build_forward_expand(graph, outputs[0]);
        if (!stage.backend->alloc(graph) ||
            !stage.backend->set_input_f32(input, slat_features.data(), slat_features.size()) ||
            !upload_mesh_inputs(stage, decoder.inputs, decoder.table_data) ||
            !stage.backend->run(graph) ||
            !stage.backend->get_tensor_f32(outputs[0], features)) return false;
    }
    if (!dump_dir.empty())
        save_raw_tensor_f32(dump_dir + "/mesh_base.samt", {768, count}, features.data());

    MeshSubdivideTables levels[2];
    for (int level = 0; level < 2; ++level) {
        const int32_t* parent_coords = level == 0 ? coords.data() : levels[0].child_coords.data();
        const int64_t parent_count = level == 0 ? count : levels[0].child_count;
        if (!levels[level].build(parent_coords, parent_count)) return false;
        std::vector<float> output;
        {
            const int64_t channels = level == 0 ? 768 : 192;
            GraphContext graph_context;
            ggml_tensor* input = graph_context.input_f32(
                "mesh_upsample_input", {channels, levels[level].parent_count});
            MeshUpsampleGraph decoder;
            decoder.g = &graph_context;
            decoder.m = stage.model.get();
            decoder.tables = &levels[level];
            decoder.level = level;
            decoder.x = input;
            decoder.inputs.push_back(input);
            decoder.table_data.push_back(nullptr);
            ggml_tensor* result = decoder.build();
            ggml_cgraph* graph = ggml_new_graph_custom(graph_context.ctx(), 65536, false);
            ggml_set_output(result);
            ggml_build_forward_expand(graph, result);
            if (!stage.backend->alloc(graph) ||
                !stage.backend->set_input_f32(input, features.data(), features.size()) ||
                !upload_mesh_inputs(stage, decoder.inputs, decoder.table_data) ||
                !stage.backend->run(graph) || !stage.backend->get_tensor_f32(result, output)) return false;
        }
        features = std::move(output);
        if (!dump_dir.empty()) {
            const int64_t channels = level == 0 ? 192 : 96;
            save_raw_tensor_f32(dump_dir + "/mesh_upsample" + std::to_string(level) + ".samt",
                                {channels, levels[level].child_count}, features.data());
        }
    }

    std::vector<float> raw;
    {
        GraphContext graph_context;
        ggml_tensor* input = graph_context.input_f32(
            "mesh_output_input", {96, levels[1].child_count});
        ggml_tensor* output = build_mesh_output_layer(graph_context, *stage.model, input);
        ggml_cgraph* graph = ggml_new_graph_custom(graph_context.ctx(), 64, false);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        if (!stage.backend->alloc(graph) ||
            !stage.backend->set_input_f32(input, features.data(), features.size()) ||
            !stage.backend->run(graph) || !stage.backend->get_tensor_f32(output, raw)) return false;
    }
    if (!dump_dir.empty()) {
        save_raw_tensor_f32(dump_dir + "/mesh_raw.samt", {101, levels[1].child_count}, raw.data());
        RawTensor coord_tensor;
        coord_tensor.ne = {4, levels[1].child_count};
        coord_tensor.type = GGML_TYPE_I32;
        coord_tensor.data.resize(levels[1].child_coords.size() * sizeof(int32_t));
        std::memcpy(coord_tensor.data.data(), levels[1].child_coords.data(), coord_tensor.data.size());
        save_raw_tensor(dump_dir + "/mesh_coords.samt", coord_tensor);
    }
    std::string extraction_error;
    if (!extract_object_mesh(raw.data(), levels[1].child_coords.data(), levels[1].child_count,
                             mesh, nullptr, &extraction_error)) {
        LOGE("e2e: mesh extraction failed: %s", extraction_error.c_str());
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// e2e command
// ---------------------------------------------------------------------------

int cmd_e2e(const CliOptions& options, const std::string& cond_dir,
            const std::string& noise_dir, const std::string& out_path,
            const std::string& dbg_dir, unsigned seed) {
    const std::string& models_dir = options.models_dir;
    const std::string& dt = options.model_dtype;
    const char* ss_dtx = getenv("SAM3D_E2E_SS_DTYPE");
    const std::string ss_dt = ss_dtx ? ss_dtx : dt;
    // The occupancy decoder determines the sparse support consumed by every
    // later stage. Keep it independently selectable so Q4 flow experiments
    // can retain this small, threshold-sensitive model without promoting the
    // multi-gigabyte SS transformer.
    const char* ss_decoder_dtx = getenv("SAM3D_E2E_SS_DECODER_DTYPE");
    const std::string ss_decoder_dt = ss_decoder_dtx ? ss_decoder_dtx : ss_dt;
    // Keep the structured-latent generator independently selectable for
    // quantization localization. The default preserves the homogeneous E2E
    // model family selected by SAM3D_E2E_DTYPE.
    const char* slat_dtx = getenv("SAM3D_E2E_SLAT_DTYPE");
    const std::string slat_dt = slat_dtx ? slat_dtx : dt;
    const char* gs_dtx = getenv("SAM3D_E2E_GS_DTYPE");
    const std::string gs_dt = gs_dtx ? gs_dtx : dt;
    const auto valid_dtype = [](const std::string& dtype) {
        return dtype == "f32" || dtype == "f16" || dtype == "q4_0" ||
               dtype == "q4_1" || dtype == "q4_k" || dtype == "q8_0";
    };
    if (!valid_dtype(dt) || !valid_dtype(ss_dt) || !valid_dtype(ss_decoder_dt) ||
        !valid_dtype(slat_dt)) {
        LOGE("e2e: unsupported model dtype: base=%s ss=%s ss_decoder=%s slat=%s",
             dt.c_str(), ss_dt.c_str(), ss_decoder_dt.c_str(), slat_dt.c_str());
        return 1;
    }
    // The decoder graph accepts every model dtype accepted above. CUDA and
    // Vulkan both have native Q4_1 matrix and copy support; rejecting the
    // exported Q4_1 decoder here made the E2E path unreachable.
    if (!valid_dtype(gs_dt)) {
        LOGE("e2e: unsupported Gaussian decoder dtype: %s", gs_dt.c_str());
        return 1;
    }
    const double t_start = (double)clock() / CLOCKS_PER_SEC;
    // Split the pipeline into independent processes when GPU memory is tight:
    //   cond: embedder/fuser -> fused condition tokens
    //   ss:   SS flow/decoder -> sparse coordinates
    //   slat: SLat flow plus the selected geometry or Gaussian decoder
    // This prevents a prior model's CUDA allocator pool from consuming the
    // memory required by the next model's graph. "flow" remains compatible.
    const char* stage_env = getenv("SAM3D_E2E_STAGE");
    const bool cond_stage = stage_env && std::string(stage_env) == "cond";
    const bool ss_stage = stage_env && std::string(stage_env) == "ss";
    const bool slat_stage = stage_env && std::string(stage_env) == "slat";
    const bool flow_stage_env = stage_env && std::string(stage_env) == "flow";

    // ---------------- stage A: sparse structure --------------------------
    // The SLat process receives coordinates and does not need SS weights.
    Stage ss;
    if (!flow_stage_env && !slat_stage) {
        if (!ss.open(models_dir + "/ss_generator-" + ss_dt + ".gguf", options,
                     "condition_ss_generator")) {
            LOGE("e2e: failed to load ss_generator"); return 1;
        }
        LOGI("e2e: ss_generator loaded (%s)", ss_dt.c_str());
    }

    // SKIP_COND=1: no embedder graphs at all (cond from SAMT)
    // SKIP_COND=2: only the fuser graph (dino/pp tokens from the dbg dumps)
    if (flow_stage_env || ss_stage || slat_stage) {
        // Isolated flow stages consume tokens written by the cond process.
        setenv("SAM3D_E2E_SKIP_COND", "1", 1);
    }
    const int skip_cond = getenv("SAM3D_E2E_SKIP_COND")
                              ? atoi(getenv("SAM3D_E2E_SKIP_COND")) : 0;
    CondInputs ci;
    std::vector<float> ss_cond;
    std::vector<float> slat_cond;
    if (skip_cond >= 1) {
        if (!slat_stage) {
            const char* ss_cond_path_env = getenv("SAM3D_E2E_SS_COND_PATH");
            const std::string ss_cond_path = ss_cond_path_env
                ? ss_cond_path_env : cond_dir + "/ss_cond_tokens.samt";
            RawTensor ct;
            if (!load_raw_tensor(ss_cond_path, ct) || ct.type != GGML_TYPE_F32) return 1;
            ss_cond.assign(reinterpret_cast<const float*>(ct.data.data()),
                           reinterpret_cast<const float*>(ct.data.data()) + ct.data.size() / 4);
        }
        if (flow_stage_env || slat_stage) {
            const char* slat_cond_path_env = getenv("SAM3D_E2E_SLAT_COND_PATH");
            const std::string slat_cond_path = slat_cond_path_env
                ? slat_cond_path_env : cond_dir + "/slat_cond_tokens.samt";
            RawTensor st3;
            if (!load_raw_tensor(slat_cond_path, st3) || st3.type != GGML_TYPE_F32) return 1;
            slat_cond.assign(reinterpret_cast<const float*>(st3.data.data()),
                             reinterpret_cast<const float*>(st3.data.data()) + st3.data.size() / 4);
        }
        LOGI("e2e: SKIP_COND - fused cond tokens loaded");
    }
    if (!skip_cond) {
        if (options.input_rgba.empty()) {
            if (!load_cond_inputs(cond_dir, ci)) return 1;
        } else if (!infer_cond_inputs(options, ci)) {
            return 1;
        }
    }
    LOGI("e2e: condition inputs %lldx%lld", (long long)ci.W, (long long)ci.H);

    if (cond_stage) {
        int64_t nd = 0;
        auto d4 = run_dino_batch(ss, {{"cemb.emb0", &ci.image}, {"cemb.emb0", &ci.rgb_image},
                                      {"cemb.emb1", &ci.mask3}, {"cemb.emb1", &ci.rgb_image_mask3}},
                                 false, nd);
        if (d4.empty() || d4[0].empty()) { LOGE("e2e-cond: dino failed"); return 1; }
        auto p2 = run_pointpatch_batch(ss, "cemb.emb2", {{&ci.pointmap, {ci.W, ci.H}},
                                                         {&ci.rgb_pointmap, {ci.W, ci.H}}});
        if (p2.empty() || p2[0].empty()) { LOGE("e2e-cond: pointpatch failed"); return 1; }
        const int64_t n_pp2 = (int64_t)p2[0].size() / 512;
        std::vector<float> cond_out;
        {
            GraphContext gctx;
            std::vector<FuserSeg> segs = {
                {1024, nd, 0, 0, std::move(d4[0])}, {1024, nd, 0, 1, std::move(d4[1])},
                {1024, nd, 1, 0, std::move(d4[2])}, {1024, nd, 1, 1, std::move(d4[3])},
                {512, n_pp2, 2, 0, std::move(p2[0])}, {512, n_pp2, 2, 1, std::move(p2[1])},
            };
            ggml_tensor* out = build_fuser(gctx, *ss.model, segs, "cemb");
            ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 16384, false);
            ggml_build_forward_expand(graph, out);
            if (!ss.backend->alloc(graph)) return 1;
            for (auto& sg2 : segs)
                if (!ss.backend->set_input_f32(sg2.tokens, sg2.host.data(), sg2.host.size())) return 1;
            if (!ss.backend->run(graph)) return 1;
            ss.backend->get_tensor_f32(out, cond_out);
        }
        save_raw_tensor_f32(out_path + ".ss_cond.samt", {1024, 7528}, cond_out.data());
        ss.close();   // free the SS weights before the slat embedder stage

        Stage slat_emb;
        if (!slat_emb.open(models_dir + "/slat_generator-" + slat_dt + ".gguf", options,
                           "condition_slat_generator"))
            return 1;
        auto sd = run_dino_batch(slat_emb, {{"cemb.emb0", &ci.image}, {"cemb.emb0", &ci.rgb_image},
                                            {"cemb.emb1", &ci.mask3}, {"cemb.emb1", &ci.rgb_image_mask3}},
                                 true, nd);
        if (sd.empty() || sd[0].empty()) { LOGE("e2e-cond: slat dino failed"); return 1; }
        std::vector<float> slat_out;
        {
            GraphContext gctx;
            std::vector<FuserSeg> segs = {
                {1024, nd, 0, 0, std::move(sd[0])}, {1024, nd, 0, 1, std::move(sd[1])},
                {1024, nd, 1, 0, std::move(sd[2])}, {1024, nd, 1, 1, std::move(sd[3])},
            };
            ggml_tensor* out = build_fuser(gctx, *slat_emb.model, segs, "cemb");
            ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 16384, false);
            ggml_build_forward_expand(graph, out);
            if (!slat_emb.backend->alloc(graph)) return 1;
            for (auto& sg2 : segs)
                if (!slat_emb.backend->set_input_f32(sg2.tokens, sg2.host.data(), sg2.host.size())) return 1;
            if (!slat_emb.backend->run(graph)) return 1;
            slat_emb.backend->get_tensor_f32(out, slat_out);
        }
        save_raw_tensor_f32(out_path + ".slat_cond.samt", {1024, 5496}, slat_out.data());
        LOGI("e2e-cond: done in %.1fs", (double)clock() / CLOCKS_PER_SEC - t_start);
        return 0;
    }


    if (!skip_cond) {
    // 4 DINO forwards: emb0(image), emb0(rgb_image), emb1(mask), emb1(rgb_mask)
    int64_t n_dino = 0;
    auto d = run_dino_batch(ss, {{"cemb.emb0", &ci.image}, {"cemb.emb0", &ci.rgb_image},
                                 {"cemb.emb1", &ci.mask3}, {"cemb.emb1", &ci.rgb_image_mask3}},
                            false, n_dino);
    if (d.empty() || d[0].empty()) { LOGE("e2e: dino forward failed"); return 1; }
    std::vector<float> d_img = std::move(d[0]), d_rgb = std::move(d[1]);
    std::vector<float> d_mask = std::move(d[2]), d_rgbm = std::move(d[3]);
    LOGI("e2e: dino tokens done (n=%lld each)", (long long)n_dino);

    // 2 PointPatch forwards (one graph)
    auto p = run_pointpatch_batch(ss, "cemb.emb2", {{&ci.pointmap, {ci.W, ci.H}},
                                                    {&ci.rgb_pointmap, {ci.W, ci.H}}});
    if (p.empty() || p[0].empty()) { LOGE("e2e: pointpatch failed"); return 1; }
    std::vector<float> p_pm = std::move(p[0]), p_rgbpm = std::move(p[1]);
    const int64_t n_pp = (int64_t)p_pm.size() / 512;
    if (!dbg_dir.empty()) {
        save_raw_tensor_f32(dbg_dir + "/e2e_pp_tokens.samt", {512, n_pp}, p_pm.data());
        save_raw_tensor_f32(dbg_dir + "/e2e_dino_tokens.samt", {1024, n_dino}, d_img.data());
    }
    LOGI("e2e: pointpatch tokens done (n=%lld each)", (long long)n_pp);

    // fuser -> ss condition tokens (1024, 7528)
        // ss_cond filled by the fuser below
    {
        GraphContext gctx;
        std::vector<FuserSeg> segs = {
            {1024, n_dino, 0, 0, std::move(d_img)},
            {1024, n_dino, 0, 1, std::move(d_rgb)},
            {1024, n_dino, 1, 0, std::move(d_mask)},
            {1024, n_dino, 1, 1, std::move(d_rgbm)},
            {512, n_pp, 2, 0, std::move(p_pm)},
            {512, n_pp, 2, 1, std::move(p_rgbpm)},
        };
        ggml_tensor* out = build_fuser(gctx, *ss.model, segs, "cemb");
        ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 16384, false);
        ggml_build_forward_expand(graph, out);
        if (!ss.backend->alloc(graph)) { LOGE("e2e: fuser alloc failed"); return 1; }
        for (auto& s : segs) {
            if (!ss.backend->set_input_f32(s.tokens, s.host.data(), s.host.size())) return 1;
        }
        if (!ss.backend->run(graph)) { LOGE("e2e: fuser run failed"); return 1; }
        ss.backend->get_tensor_f32(out, ss_cond);
        }
    }
    LOGI("e2e: ss condition tokens %lld", (long long)ss_cond.size() / 1024);
    if (!dbg_dir.empty() && !ss_cond.empty())
        save_raw_tensor_f32(dbg_dir + "/e2e_ss_cond_tokens.samt", {1024, 7528}, ss_cond.data());

    // fresh backend for the flow graph: a shared gallocr across the embedder
    // graphs and the flow graph changed its allocation layout and produced
    // NaNs on the 4th flow run (CUDA only); a clean gallocr per stage is
    // also the CLI-verified configuration.
    ss.close();

    // A strict replay consumes immutable official support. The dedicated SLat
    // stage likewise starts from coordinates produced by the dedicated SS
    // process, so neither path allocates SS model state in this process.
    const bool strict_reference_coords = getenv("SAM3D_E2E_REFERENCE_COORDS") != nullptr;
    std::vector<int32_t> coords;  // (N,4) [b, x, y, z], z fastest
    std::mt19937 rng(seed);
    if (!slat_stage && (!flow_stage_env || !strict_reference_coords)) {
    // ---- SS sampling: 25 rescaled Euler steps; only shape uses CFG ----
    // Values match pipeline.yaml plus InferencePipeline's SS defaults. The
    // Python wrapper enables no_shortcut, so d remains zero for every step.
    constexpr int kSsSteps = 25;
    constexpr float kSsRescaleT = 3.0f;
    constexpr float kSsCfgStrength = 7.0f;
    constexpr float kSsCfgStart = 0.0f;
    constexpr float kSsCfgEnd = 500.0f;
    const int64_t n_shape = 4096;
    std::vector<float> x_shape(8 * n_shape), x_6d(6), x_sc(3), x_tr(3), x_ts(1);
    if (!load_noise_opt(noise_dir, "ss_x0_shape.samt", x_shape, 8 * n_shape) ||
        !load_noise_opt(noise_dir, "ss_x0_6drotation_normalized.samt", x_6d, 6) ||
        !load_noise_opt(noise_dir, "ss_x0_scale.samt", x_sc, 3) ||
        !load_noise_opt(noise_dir, "ss_x0_translation.samt", x_tr, 3) ||
        !load_noise_opt(noise_dir, "ss_x0_translation_scale.samt", x_ts, 1)) {
        LOGI("e2e: reference noise unavailable, using PRNG seed %u", seed);
        fill_normal(x_shape, rng); fill_normal(x_6d, rng); fill_normal(x_sc, rng);
        fill_normal(x_tr, rng); fill_normal(x_ts, rng);
    }
    if (!dbg_dir.empty()) {
        // Preserve the exact stochastic inputs so the native trajectory can
        // be replayed by the pinned upstream PyTorch implementation.  This is
        // also needed when the native run uses its deterministic C++ fallback
        // PRNG, whose stream intentionally differs from torch's Philox stream.
        save_raw_tensor_f32(dbg_dir + "/ss_x0_shape.samt", {8, n_shape}, x_shape.data());
        save_raw_tensor_f32(dbg_dir + "/ss_x0_6drotation_normalized.samt", {6, 1}, x_6d.data());
        save_raw_tensor_f32(dbg_dir + "/ss_x0_scale.samt", {3, 1}, x_sc.data());
        save_raw_tensor_f32(dbg_dir + "/ss_x0_translation.samt", {3, 1}, x_tr.data());
        save_raw_tensor_f32(dbg_dir + "/ss_x0_translation_scale.samt", {1, 1}, x_ts.data());
        save_raw_tensor_f32(dbg_dir + "/ss_cond_tokens.samt", {1024, 7528}, ss_cond.data());
    }

    Stage ssf;
    if (!ssf.open(models_dir + "/ss_generator-" + ss_dt + ".gguf", options,
                  "ss_flow")) {
        LOGE("e2e: failed to reload ss_generator for the flow stage"); return 1;
    }
    GraphContext fg;
    SsFlowGraph sfg;
    sfg.g = &fg; sfg.m = ssf.model.get();
    sfg.n_cond_tokens = 7528;
    auto outs = sfg.build();   // dict order: 6drot, scale, shape, translation, ts
    ggml_cgraph* fgraph = ggml_new_graph_custom(fg.ctx(), 32768, false);
    for (auto* o : outs) { ggml_set_output(o); ggml_build_forward_expand(fgraph, o); }
    if (!ssf.backend->alloc(fgraph)) { LOGE("e2e: ss flow alloc failed"); return 1; }
    LOGI("e2e: ss flow graph built (%d nodes)", ggml_graph_n_nodes(fgraph));

    std::vector<float> zeros_cond(ss_cond.size(), 0.0f);
    // Table values are invariant for all Euler/CFG replays. A Vulkan A/B
    // experiment produced different sparse coordinates when caching this
    // input, so the path remains opt-in until buffer lifetime is proven.
    const bool cache_vulkan_ss_tables =
        strstr(ssf.backend->backend_name(), "Vulkan") != nullptr &&
        getenv("SAM3D_E2E_ENABLE_VULKAN_SS_TABLE_CACHE") != nullptr &&
        getenv("SAM3D_E2E_DISABLE_VULKAN_TABLE_CACHE") == nullptr;
    auto up_scalar = [&](ggml_tensor* t, float v) {
        return !t->buffer || ssf.backend->set_input_f32(t, &v, 1);
    };
    auto upload_ss_tables = [&]() -> bool {
        for (size_t ti = 0; ti < sfg.inputs.size(); ++ti) {
            ggml_tensor* input = sfg.inputs[ti];
            if (!input->buffer || !sfg.table_data[ti]) continue;
            const auto& host = sfg.table_data[ti];
            const bool ok = input->type == GGML_TYPE_F32
                ? ssf.backend->set_input_f32(input,
                    reinterpret_cast<const float*>(host->data()), host->size())
                : ssf.backend->set_input_i32(input, host->data(), host->size());
            if (!ok) {
                LOGE("e2e: failed to upload SS table %s", input->name);
                return false;
            }
        }
        return true;
    };
    if (cache_vulkan_ss_tables && !upload_ss_tables()) return 1;
    auto upload_and_run = [&](float t_v, bool zero_cond) -> bool {
        auto up = [&](ggml_tensor* t, const std::vector<float>& v) {
            if (!t->buffer) return true;
            return ssf.backend->set_input_f32(t, v.data(), v.size());
        };
        if (!up(sfg.x_shape, x_shape)) return false;
        if (!up(sfg.x_6drot, x_6d)) return false;
        if (!up(sfg.x_scale, x_sc)) return false;
        if (!up(sfg.x_trans, x_tr)) return false;
        if (!up(sfg.x_ts, x_ts)) return false;
        if (!up_scalar(sfg.t, t_v)) return false;
        if (!up_scalar(sfg.d, 0.0f)) return false;
        if (!up(sfg.cond, zero_cond ? zeros_cond : ss_cond)) return false;
        if (!cache_vulkan_ss_tables && !upload_ss_tables()) return false;
        if (getenv("SAM3D_E2E_VERIFY")) {
            float rb_t = -9;
            std::vector<float> rb_t1(1), rb_c(4), rb_x(4);
            ssf.backend->get_tensor_f32(sfg.t, rb_t1);
            rb_t = rb_t1[0];
            ssf.backend->get_tensor_f32(sfg.cond, rb_c);
            ssf.backend->get_tensor_f32(sfg.x_shape, rb_x);
            LOGI("verify: t=%g c0=%g x0=%g (expect t=%g c0=%g x0=%g)",
                 rb_t, rb_c[0], rb_x[0], t_v,
                 (zero_cond ? 0.0f : ss_cond[0]), x_shape[0]);
        }
        return ssf.backend->run(fgraph);
    };
    auto get_outs = [&](std::vector<std::vector<float>>& v) {
        v.resize(outs.size());
        for (size_t i = 0; i < outs.size(); i++) ssf.backend->get_tensor_f32(outs[i], v[i]);
    };

    int ss_steps = kSsSteps;
    if (const char* steps_env = getenv("SAM3D_E2E_SS_STEPS")) {
        ss_steps = atoi(steps_env);
        if (ss_steps < 1 || ss_steps > kSsSteps) {
            LOGE("e2e: SAM3D_E2E_SS_STEPS must be in [1, %d], got %s", kSsSteps, steps_env);
            return 1;
        }
    }
    const auto ss_schedule = make_euler_schedule(kSsSteps, kSsRescaleT);
    for (int step = 0; step < ss_steps; ++step) {
        const float t_v = ss_schedule[step].t * 1000.0f;
        if (!upload_and_run(t_v, false)) { LOGE("e2e: ss cond run failed"); return 1; }
        std::vector<std::vector<float>> vc;
        get_outs(vc);
        std::vector<std::vector<float>> vu;
        const bool cfg_active = t_v >= kSsCfgStart && t_v <= kSsCfgEnd;
        if (cfg_active) {
            if (!upload_and_run(t_v, true)) { LOGE("e2e: ss uncond run failed"); return 1; }
            get_outs(vu);
        }
        if (!dbg_dir.empty()) {
            for (size_t oi = 0; oi < outs.size(); oi++) {
                save_raw_tensor_f32(dbg_dir + "/e2e_ss_vc" + std::to_string(step) + "_" +
                                        std::to_string(oi) + ".samt",
                                    ggml_n_dims(outs[oi]) == 2
                                        ? std::vector<int64_t>{outs[oi]->ne[0], outs[oi]->ne[1]}
                                        : std::vector<int64_t>{outs[oi]->ne[0]},
                                    vc[oi].data());
                if (cfg_active) {
                    save_raw_tensor_f32(dbg_dir + "/e2e_ss_vu" + std::to_string(step) + "_" +
                                            std::to_string(oi) + ".samt",
                                        ggml_n_dims(outs[oi]) == 2
                                            ? std::vector<int64_t>{outs[oi]->ne[0], outs[oi]->ne[1]}
                                            : std::vector<int64_t>{outs[oi]->ne[0]},
                                        vu[oi].data());
                }
            }
            // dump the step-1 inputs for offline CLI repro
            if (step == 1) {
                save_raw_tensor_f32(dbg_dir + "/e2e_ss_x1_shape.samt", {8, n_shape}, x_shape.data());
                save_raw_tensor_f32(dbg_dir + "/e2e_ss_x1_6d.samt", {6}, x_6d.data());
            }
        }
        for (size_t i = 0; i < outs.size(); i++) {
            std::vector<float>& x = i == 2 ? x_shape
                                  : i == 0 ? x_6d : i == 1 ? x_sc
                                  : i == 3 ? x_tr : x_ts;
            for (size_t k = 0; k < x.size(); k++) {
                // ShortCut.cfg_modalities only contains "shape". Pose heads
                // always use the conditional velocity.
                const float velocity = cfg_active && i == 2
                    ? vc[i][k] + kSsCfgStrength * (vc[i][k] - vu[i][k])
                    : vc[i][k];
                x[k] += ss_schedule[step].dt * velocity;
            }
        }
        if (!dbg_dir.empty()) {
            const char* x_names[] = {"6drotation_normalized", "scale", "shape",
                                     "translation", "translation_scale"};
            const std::vector<float>* states[] = {&x_6d, &x_sc, &x_shape, &x_tr, &x_ts};
            const std::vector<int64_t> x_ne[] = {{6}, {3}, {8, n_shape}, {3}, {1}};
            for (size_t i = 0; i < 5; i++) {
                save_raw_tensor_f32(dbg_dir + "/e2e_ss_state" + std::to_string(step + 1) + "_" +
                                        x_names[i] + ".samt", x_ne[i], states[i]->data());
            }
        }
        LOGI("e2e: ss step %d/%d done", step + 1, ss_steps);
    }
    ssf.close();
    if (!dbg_dir.empty())
        save_raw_tensor_f32(dbg_dir + "/e2e_ss_shape_latent.samt", {8, n_shape}, x_shape.data());
    if (getenv("SAM3D_E2E_SS_FLOW_ONLY")) {
        LOGI("e2e: SS flow-only diagnostic done in %.1fs",
             (double)clock() / CLOCKS_PER_SEC - t_start);
        return 0;
    }

    // ---- ss_decoder -> occupancy -> coords ------------------------------
    {
        // the SS decoder lives in its own GGUF (ss_decoder-*.gguf)
        Stage dec;
        if (!dec.open(models_dir + "/ss_decoder-" + ss_decoder_dt + ".gguf", options,
                      "ss_decoder")) {
            LOGE("e2e: failed to load ss_decoder"); return 1;
        }
        GraphContext gctx;
        SsDecoderGraph sdg;
        sdg.ctx = gctx.ctx(); sdg.m = dec.model.get();
        // torch lat_in = shape_latent(1,4096,8).permute(0,2,1).view(1,8,16,16,16):
        // memory becomes channel-major c*4096 + (x*16+y)*16+z -> transpose here
        std::vector<float> lat_cm(x_shape.size());
        for (int64_t n = 0; n < n_shape; n++)
            for (int c = 0; c < 8; c++) lat_cm[c * n_shape + n] = x_shape[n * 8 + c];
        ggml_tensor* lat = gctx.input_f32("latent", {16, 16, 16, 8});
        ggml_tensor* occ = sdg.build(lat);
        ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 8192, false);
        ggml_build_forward_expand(graph, lat);
        ggml_build_forward_expand(graph, occ);
        if (!dec.backend->alloc(graph)) { LOGE("e2e: ss decoder alloc failed"); return 1; }
        if (!dec.backend->set_input_f32(lat, lat_cm.data(), lat_cm.size())) return 1;
        if (!dec.backend->run(graph)) { LOGE("e2e: ss decoder run failed"); return 1; }
        std::vector<float> occ_f;
        dec.backend->get_tensor_f32(occ, occ_f);
        dec.close();
        // torch (1,1,64,64,64) C-order: flat = (x*64 + y)*64 + z
        for (int x = 0; x < 64; x++)
            for (int y = 0; y < 64; y++)
                for (int z = 0; z < 64; z++)
                    if (occ_f[((x * 64) + y) * 64 + z] > 0.0f)
                        coords.insert(coords.end(), {0, x, y, z});
    }
    LOGI("e2e: occupancy coords %lld (of 262144)", (long long)coords.size() / 4);
    ss.close();   // free the SS weights before the SLat stage
    }

    int64_t n_coord = 0;
    if (slat_stage && !strict_reference_coords) {
        const char* coords_path_env = getenv("SAM3D_E2E_COORDS_PATH");
        if (!coords_path_env) {
            LOGE("e2e: SAM3D_E2E_COORDS_PATH is required for the slat stage");
            return 1;
        }
        RawTensor saved_coords;
        if (!load_raw_tensor(coords_path_env, saved_coords) || saved_coords.type != GGML_TYPE_I32 ||
            saved_coords.ne.size() != 2 || saved_coords.ne[0] != 4 ||
            saved_coords.data.size() % (4 * sizeof(int32_t)) != 0) {
            LOGE("e2e: invalid sparse coordinate SAMT: %s", coords_path_env);
            return 1;
        }
        const int32_t* values = reinterpret_cast<const int32_t*>(saved_coords.data.data());
        coords.assign(values, values + saved_coords.data.size() / sizeof(int32_t));
        n_coord = static_cast<int64_t>(coords.size() / 4);
        LOGI("e2e: loaded %lld sparse coords from the SS stage", static_cast<long long>(n_coord));
    } else {
        const size_t before_prune = coords.size() / 4;
        coords = prune_surface_coords(coords);
        LOGI("e2e: surface-pruned sparse coords %zu -> %zu", before_prune, coords.size() / 4);
        n_coord = static_cast<int64_t>(coords.size() / 4);
        if (n_coord > 42000) {
            const size_t before = coords.size() / 4;
            coords = downsample_coords(coords, 42000, 2, seed);
            n_coord = static_cast<int64_t>(coords.size() / 4);
            LOGI("e2e: downsampled sparse coords %zu -> %zu", before, coords.size() / 4);
        }
    }
    // A strict replay holds the sparse support fixed to the official output.
    // It isolates SLat and Gaussian-decoder quantization from a boundary flip
    // in SS occupancy; normal image-to-3D inference never sets this switch.
    if (strict_reference_coords) {
        RawTensor reference_coords;
        if (!load_raw_tensor(cond_dir + "/coords.samt", reference_coords) ||
            reference_coords.type != GGML_TYPE_I32 || reference_coords.data.size() % (4 * sizeof(int32_t)) != 0) {
            LOGE("e2e: invalid reference coords.samt");
            return 1;
        }
        const int32_t* values = reinterpret_cast<const int32_t*>(reference_coords.data.data());
        coords.assign(values, values + reference_coords.data.size() / sizeof(int32_t));
        n_coord = static_cast<int64_t>(coords.size() / 4);
        LOGI("e2e: strict replay replaces sparse support with %lld official coords",
             static_cast<long long>(n_coord));
    }
    if (ss_stage) {
        RawTensor saved_coords;
        saved_coords.ne = {4, n_coord};
        saved_coords.type = GGML_TYPE_I32;
        saved_coords.data.resize(coords.size() * sizeof(int32_t));
        std::memcpy(saved_coords.data.data(), coords.data(), saved_coords.data.size());
        const std::string coords_path = out_path + ".coords.samt";
        if (!save_raw_tensor(coords_path, saved_coords)) {
            LOGE("e2e: failed to save SS coordinates: %s", coords_path.c_str());
            return 1;
        }
        LOGI("e2e-ss: wrote %lld sparse coords to %s", static_cast<long long>(n_coord),
             coords_path.c_str());
        return 0;
    }

    // ---------------- stage B: structured latent -------------------------
    if (!skip_cond) {
    Stage slat_emb;
    if (!slat_emb.open(models_dir + "/slat_generator-" + slat_dt + ".gguf", options,
                       "slat_condition_generator")) {
        LOGE("e2e: failed to load slat_generator"); return 1;
    }
    LOGI("e2e: slat_generator loaded (%s)", slat_dt.c_str());

    // slat DINOs (prenorm mode: 1374 tokens each, registers kept) run on a
    // dedicated stage (fresh gallocr); the flow stage is opened after.
    int64_t n_sdino = 0;
    auto sd = run_dino_batch(slat_emb, {{"cemb.emb0", &ci.image}, {"cemb.emb0", &ci.rgb_image},
                                    {"cemb.emb1", &ci.mask3}, {"cemb.emb1", &ci.rgb_image_mask3}},
                             true, n_sdino);
    if (sd.empty() || sd[0].empty()) { LOGE("e2e: slat dino failed"); return 1; }
    std::vector<float> s_dimg = std::move(sd[0]), s_drgb = std::move(sd[1]);
    std::vector<float> s_dmask = std::move(sd[2]), s_drgbm = std::move(sd[3]);
    LOGI("e2e: slat dino tokens done (n=%lld each)", (long long)n_sdino);

        // slat_cond loaded (embedder or flow dump)
    {
        GraphContext gctx;
        std::vector<FuserSeg> segs = {
            {1024, n_sdino, 0, 0, std::move(s_dimg)},
            {1024, n_sdino, 0, 1, std::move(s_drgb)},
            {1024, n_sdino, 1, 0, std::move(s_dmask)},
            {1024, n_sdino, 1, 1, std::move(s_drgbm)},
        };
        ggml_tensor* out = build_fuser(gctx, *slat_emb.model, segs, "cemb");
        ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 16384, false);
        ggml_build_forward_expand(graph, out);
        if (!slat_emb.backend->alloc(graph)) { LOGE("e2e: slat fuser alloc failed"); return 1; }
        for (auto& s : segs) {
            if (!slat_emb.backend->set_input_f32(s.tokens, s.host.data(), s.host.size())) return 1;
        }
        if (!slat_emb.backend->run(graph)) { LOGE("e2e: slat fuser run failed"); return 1; }
        slat_emb.backend->get_tensor_f32(out, slat_cond);
    }
    slat_emb.close();
    }

    // fresh stage for the slat flow loop (CLI-verified single-graph gallocr)
    Stage slat;
    if (!slat.open(models_dir + "/slat_generator-" + slat_dt + ".gguf", options,
                   "slat_flow")) {
        LOGE("e2e: failed to reload slat_generator"); return 1;
    }
    const int64_t n_slat_cond = (int64_t)slat_cond.size() / 1024;
    LOGI("e2e: slat condition tokens %lld", (long long)n_slat_cond);
    if (!dbg_dir.empty())
        save_raw_tensor_f32(dbg_dir + "/e2e_slat_cond_tokens.samt", {1024, n_slat_cond},
                            slat_cond.data());

    // sparse tables from the (fixed) coords
    SlatTables tb;
    if (!tb.build(coords.data(), n_coord)) { LOGE("e2e: slat tables failed"); return 1; }
    LOGI("e2e: sparse tables nf=%lld nc=%lld", (long long)tb.nf, (long long)tb.nc);

    // ---- SLat sampling: 25 Euler steps with the pipeline CFG schedule ----
    // pipeline.yaml overrides the SLat configuration to strength=1 and
    // rescale_t=1; override_slat_generator_cfg_config supplies [0, 500].
    constexpr int kSlatSteps = 25;
    constexpr float kSlatRescaleT = 1.0f;
    constexpr float kSlatCfgStrength = 1.0f;
    constexpr float kSlatCfgStart = 0.0f;
    constexpr float kSlatCfgEnd = 500.0f;
    std::vector<float> x_slat(8 * tb.nf);
    if (!load_slat_noise_for_coords(noise_dir, coords, seed, x_slat))
        fill_normal(x_slat, rng);

    GraphContext sg;
    SlatFlowGraph slg;
    slg.g = &sg; slg.m = slat.model.get(); slg.tb = &tb;
    slg.n_cond_tokens = n_slat_cond;
    const bool split_slat_final = ggml_is_quantized(slat.model->get("dit.out_layer.weight")->type) &&
        slat.backend->has_gpu() &&
        getenv("SAM3D_DEBUG_STAGE") == nullptr;
    // Quantized final projections need a split graph to avoid changing the
    // trunk's allocation layout.  Keep an independent allocator for that
    // small final graph so both graphs stay allocated across all Euler/CFG
    // evaluations instead of reallocating them 100 times per sample.
    std::unique_ptr<Backend> slat_final_backend;
    if (split_slat_final) {
        slat_final_backend = Backend::create(options.backend_module, options.backend,
                                             options.backend_device, options.n_threads,
                                             options.expected_device_description);
        if (!slat_final_backend) {
            LOGE("e2e: failed to create SLat final-projection backend");
            return 1;
        }
        slat_final_backend->set_profile_label("slat_final_projection");
    }
    if (split_slat_final) {
        slg.debug_stage = "pre_final";
    } else if (const char* debug_stage = getenv("SAM3D_DEBUG_STAGE")) {
        slg.debug_stage = debug_stage;
    }
    auto s_outs = slg.build();
    // Matches the independently validated slat-step graph. The graph has
    // about 2.3k nodes, so 32k leaves ample headroom without changing the
    // context allocation layout between CLI and end-to-end execution.
    ggml_cgraph* s_graph = ggml_new_graph_custom(sg.ctx(), 32768, false);
    for (auto* o : s_outs) { ggml_set_output(o); ggml_build_forward_expand(s_graph, o); }
    if (!slat.backend->alloc(s_graph)) {
        LOGE("e2e: slat flow alloc failed"); return 1;
    }
    LOGI("e2e: slat flow graph built (%d nodes)", ggml_graph_n_nodes(s_graph));

    // A full Q8 SLat graph changes the gallocr liveness layout enough for the
    // CUDA get_rows kernel to read an invalid source row. Keep the large sparse
    // trunk and the final norm/projection in independent graph allocations.
    GraphContext final_ctx;
    ggml_tensor* final_in = nullptr;
    ggml_tensor* final_out = nullptr;
    ggml_cgraph* final_graph = nullptr;
    if (split_slat_final) {
        final_in = final_ctx.input_f32("slat_pre_final", {128, tb.nf});
        ggml_tensor* final_weight = slat.model->get("dit.out_layer.weight");
        if (!ggml_is_quantized(final_weight->type) && final_weight->type != GGML_TYPE_F32)
            final_weight = ggml_cast(final_ctx.ctx(), final_weight, GGML_TYPE_F32);
        ggml_tensor* final_bias = slat.model->get("dit.out_layer.bias");
        if (final_bias->type != GGML_TYPE_F32)
            final_bias = ggml_cast(final_ctx.ctx(), final_bias, GGML_TYPE_F32);
        final_out = gb_linear(final_ctx.ctx(), final_weight, final_bias,
                              gb_layer_norm(final_ctx.ctx(), final_in, nullptr, nullptr, 1e-6f));
        final_graph = ggml_new_graph_custom(final_ctx.ctx(), 64, false);
        ggml_set_output(final_out);
        ggml_build_forward_expand(final_graph, final_out);
        if (!slat_final_backend->alloc(final_graph)) {
            LOGE("e2e: SLat final projection alloc failed"); return 1;
        }
    }

    {
        const auto slat_schedule = make_euler_schedule(kSlatSteps, kSlatRescaleT);
        const std::vector<float> zeros_slat_cond(slat_cond.size(), 0.0f);
        const bool dump_slat_steps = getenv("SAM3D_E2E_DUMP_SLAT_STEPS") != nullptr;
        // Reusing these input buffers across SLat graphs is not yet safe on
        // Vulkan: the cached path changes the final rendered entity. Keep the
        // verified per-forward upload as the default until ownership across
        // graph submissions is made explicit and regression-tested.
        const bool cache_vulkan_tables =
            strstr(slat.backend->backend_name(), "Vulkan") != nullptr &&
            getenv("SAM3D_E2E_ENABLE_VULKAN_TABLE_CACHE") != nullptr;
        auto upload_slat_tables = [&]() {
            for (size_t ti = 0; ti < slg.inputs.size(); ti++) {
                ggml_tensor* input = slg.inputs[ti];
                if (!input->buffer || !slg.table_data[ti]) continue;
                const auto& host = slg.table_data[ti];
                const bool ok = input->type == GGML_TYPE_F32
                    ? slat.backend->set_input_f32(input,
                        reinterpret_cast<const float*>(host->data()), host->size())
                    : slat.backend->set_input_i32(input, host->data(), host->size());
                if (!ok) {
                    LOGE("e2e: failed to upload SLat table %s", input->name);
                    return false;
                }
            }
            return true;
        };
        if (cache_vulkan_tables && !upload_slat_tables()) return 1;
        if (dump_slat_steps && !dbg_dir.empty()) {
            save_raw_tensor_f32(dbg_dir + "/slat_ggml_x000.samt", {8, tb.nf}, x_slat.data());
        }
        size_t debug_slat_forwards = 0;
        if (const char* debug_forwards_env = getenv("SAM3D_E2E_DEBUG_SLAT_FORWARDS")) {
            char* end = nullptr;
            const unsigned long parsed = strtoul(debug_forwards_env, &end, 10);
            if (end == debug_forwards_env || *end != '\0' || parsed == 0) {
                LOGE("e2e: SAM3D_E2E_DEBUG_SLAT_FORWARDS must be a positive integer");
                return 1;
            }
            debug_slat_forwards = static_cast<size_t>(parsed);
        } else if (getenv("SAM3D_E2E_DEBUG_ONCE")) {
            debug_slat_forwards = 1;
        }
        const char* debug_slat_output = getenv("SAM3D_E2E_DEBUG_SLAT_OUTPUT");
        size_t completed_debug_slat_forwards = 0;

        for (size_t s = 0; s < slat_schedule.size(); ++s) {
            const float t_v = slat_schedule[s].t * 1000.0f;
            if (slg.x->buffer &&
                !slat.backend->set_input_f32(slg.x, x_slat.data(), x_slat.size())) return 1;
            if (slg.cond->buffer &&
                !slat.backend->set_input_f32(slg.cond, slat_cond.data(),
                                              slat_cond.size())) return 1;
            if (slg.t->buffer && !slat.backend->set_input_f32(slg.t, &t_v, 1)) return 1;
            if (!cache_vulkan_tables) {
                if (!upload_slat_tables()) return 1;
                if (getenv("SAM3D_E2E_VERIFY")) {
                    for (ggml_tensor* t : slg.inputs) {
                        if (!t->buffer || t->type != GGML_TYPE_I32) continue;
                        std::vector<int32_t> device;
                        slat.backend->get_tensor_i32(t, device);
                        const auto [lo, hi] = std::minmax_element(device.begin(), device.end());
                        LOGI("verify: %s rows [%d, %d] (%zu values)", t->name,
                             *lo, *hi, device.size());
                    }
                }
            }
            if (!slat.backend->run(s_graph)) { LOGE("e2e: slat step %zu failed", s); return 1; }
            std::vector<float> vc;
            slat.backend->get_tensor_f32(s_outs[0], vc);
            if (split_slat_final) {
                if (!slat_final_backend->set_input_f32(final_in, vc.data(), vc.size()) ||
                    !slat_final_backend->run(final_graph) ||
                    !slat_final_backend->get_tensor_f32(final_out, vc)) {
                    LOGE("e2e: SLat final projection failed"); return 1;
                }
            }
            if (debug_slat_forwards != 0 &&
                ++completed_debug_slat_forwards >= debug_slat_forwards) {
                if (debug_slat_output && *debug_slat_output) {
                    if (!save_raw_tensor_f32(debug_slat_output, {8, tb.nf}, vc.data())) {
                        LOGE("e2e: failed to write SLat debug output %s", debug_slat_output);
                        return 1;
                    }
                }
                LOGI("e2e: completed %zu SLat debug forward(s)",
                     completed_debug_slat_forwards);
                return 0;
            }
            const bool cfg_active = t_v >= kSlatCfgStart && t_v <= kSlatCfgEnd;
            std::vector<float> vu;
            if (cfg_active) {
                if (!cache_vulkan_tables && !upload_slat_tables()) return 1;
                if (slg.x->buffer &&
                    !slat.backend->set_input_f32(slg.x, x_slat.data(), x_slat.size())) return 1;
                if (slg.t->buffer && !slat.backend->set_input_f32(slg.t, &t_v, 1)) return 1;
                if (!slat.backend->set_input_f32(slg.cond, zeros_slat_cond.data(),
                                                  zeros_slat_cond.size())) return 1;
                if (!slat.backend->run(s_graph)) { LOGE("e2e: slat uncond step %zu failed", s); return 1; }
                slat.backend->get_tensor_f32(s_outs[0], vu);
                if (split_slat_final) {
                    if (!slat_final_backend->set_input_f32(final_in, vu.data(), vu.size()) ||
                        !slat_final_backend->run(final_graph) ||
                        !slat_final_backend->get_tensor_f32(final_out, vu)) {
                        LOGE("e2e: SLat final uncond projection failed"); return 1;
                    }
                }
            }
            for (size_t k = 0; k < x_slat.size(); k++) {
                const float velocity = cfg_active
                    ? vc[k] + kSlatCfgStrength * (vc[k] - vu[k])
                    : vc[k];
                x_slat[k] += slat_schedule[s].dt * velocity;
            }
            if (dump_slat_steps && !dbg_dir.empty()) {
                char name[64];
                snprintf(name, sizeof(name), "/slat_ggml_x%03zu.samt", s + 1);
                save_raw_tensor_f32(dbg_dir + name, {8, tb.nf}, x_slat.data());
            }
            if (s % 5 == 4) LOGI("e2e: slat step %zu/%zu", s + 1, slat_schedule.size());
        }
    }
    // denormalize: feats * STD + MEAN (per channel)
    for (int64_t n = 0; n < tb.nf; n++)
        for (int c = 0; c < 8; c++)
            x_slat[n * 8 + c] = x_slat[n * 8 + c] * SLAT_STD[c] + SLAT_MEAN[c];
    if (!dbg_dir.empty())
        save_raw_tensor_f32(dbg_dir + "/e2e_slat_feats.samt", {8, tb.nf}, x_slat.data());
    if (!dbg_dir.empty()) {
        RawTensor coord_tensor;
        coord_tensor.ne = {4, tb.nf};
        coord_tensor.type = GGML_TYPE_I32;
        coord_tensor.data.resize(coords.size() * sizeof(int32_t));
        std::memcpy(coord_tensor.data.data(), coords.data(), coord_tensor.data.size());
        save_raw_tensor(dbg_dir + "/e2e_slat_coords.samt", coord_tensor);
    }
    slat.close();

    if (!options.out_glb.empty()) {
        const char* mesh_dtx = getenv("SAM3D_E2E_MESH_DTYPE");
        const std::string mesh_dt = mesh_dtx ? mesh_dtx : dt;
        if (!valid_dtype(mesh_dt)) {
            LOGE("e2e: unsupported mesh decoder dtype: %s", mesh_dt.c_str());
            return 1;
        }
        Stage mesh_stage;
        if (!mesh_stage.open(models_dir + "/slat_decoder_mesh-" + mesh_dt + ".gguf", options,
                             "mesh_decoder")) {
            LOGE("e2e: failed to load slat_decoder_mesh-%s", mesh_dt.c_str());
            return 1;
        }
        ObjectMesh mesh;
        if (!decode_object_mesh(mesh_stage, coords, x_slat, mesh, dbg_dir)) {
            LOGE("e2e: mesh decoder failed");
            return 1;
        }
        mesh_stage.close();
        std::string write_error;
        if (!write_object_glb(out_path, mesh, &write_error)) {
            LOGE("e2e: %s", write_error.c_str());
            return 1;
        }
        LOGI("e2e: wrote %s (%zu vertices, %zu faces) in %.1fs total", out_path.c_str(),
             mesh.vertices.size() / 3, mesh.faces.size() / 3,
             (double)clock() / CLOCKS_PER_SEC - t_start);
        return 0;
    }

    // ---------------- stage C: Gaussian decode + PLY ---------------------
    Stage gs;
    if (!gs.open(models_dir + "/slat_decoder_gs-" + gs_dt + ".gguf", options,
                 "gaussian_decoder")) {
        LOGE("e2e: failed to load slat_decoder_gs-%s", gs_dt.c_str()); return 1;
    }
    GsTables gtb;
    if (!gtb.build(coords.data(), n_coord)) { LOGE("e2e: gs tables failed"); return 1; }
    // offset perturbation (hammersley) from the GGUF: ggml ne = [3, 32].
    // The converter writes torch's [3, 32] C-order buffer, which is already
    // flattened as [gaussian * 3 + channel] in ggml memory.
    ggml_tensor* pert = gs.model->get("gsdec.offset_perturbation");
    GGML_ASSERT(pert && pert->ne[0] == 3 && pert->ne[1] == GS_NGAUSS);
    std::vector<float> perturb;
    gs.backend->get_tensor_f32(pert, perturb);
    GGML_ASSERT(perturb.size() == static_cast<size_t>(GS_NGAUSS * 3));

    GraphContext gg;
    GsDecoderGraph gdg;
    gdg.g = &gg; gdg.m = gs.model.get(); gdg.tb = &gtb;
    ggml_tensor* g_x = gg.input_f32("x", {8, tb.nf});
    gdg.x = g_x;
    gdg.inputs.push_back(g_x);
    gdg.table_data.push_back(nullptr);
    auto g_outs = gdg.build();   // [0] = raw feats (448, N)
    ggml_cgraph* g_graph = ggml_new_graph_custom(gg.ctx(), 65536, false);
    for (auto* o : g_outs) { ggml_set_output(o); ggml_build_forward_expand(g_graph, o); }
    if (!gs.backend->alloc(g_graph)) { LOGE("e2e: gs alloc failed"); return 1; }
    if (!gs.backend->set_input_f32(g_x, x_slat.data(), x_slat.size())) return 1;
    for (size_t ti = 0; ti < gdg.inputs.size(); ti++) {
        ggml_tensor* t = gdg.inputs[ti];
        if (!t->buffer || !gdg.table_data[ti]) continue;
        const auto& host = gdg.table_data[ti];
        const bool ok = t->type == GGML_TYPE_F32
            ? gs.backend->set_input_f32(t, reinterpret_cast<const float*>(host->data()),
                                        host->size())
            : gs.backend->set_input_i32(t, host->data(), host->size());
        if (!ok) return 1;
    }
    if (!gs.backend->run(g_graph)) { LOGE("e2e: gs run failed"); return 1; }
    std::vector<float> raw;
    gs.backend->get_tensor_f32(g_outs[0], raw);
    gs.close();

    // host to_representation (decoder_gs.py) + PLY
    const int64_t n_tok = (int64_t)raw.size() / 448;
    const int64_t n_gs_total = n_tok * GS_NGAUSS;
    std::vector<float> xyz(n_gs_total * 3), fdc(n_gs_total * 3), op(n_gs_total),
        scl(n_gs_total * 3), rot(n_gs_total * 4);
    for (int64_t n = 0; n < n_tok; n++) {
        const float* r = &raw[n * 448];
        const float cb = (float)coords[n * 4 + 1];
        const float cby = (float)coords[n * 4 + 2];
        const float cbz = (float)coords[n * 4 + 3];
        for (int g = 0; g < GS_NGAUSS; g++) {
            float* X = &xyz[(n * GS_NGAUSS + g) * 3];
            X[0] = (cb + 0.5f) / 64.f; X[1] = (cby + 0.5f) / 64.f; X[2] = (cbz + 0.5f) / 64.f;
            // offset: raw*lr(1.0) + perturbation; tanh / 64 * 0.5 * voxel_size
            for (int c = 0; c < 3; c++) {
                float o = r[g * 3 + c] + perturb[g * 3 + c];
                X[c] += tanhf(o) / 64.f * 0.5f * GS_VOXEL_SIZE;
            }
            float* F = &fdc[(n * GS_NGAUSS + g) * 3];
            for (int c = 0; c < 3; c++) F[c] = r[96 + g * 3 + c];
            float* S = &scl[(n * GS_NGAUSS + g) * 3];
            for (int c = 0; c < 3; c++) S[c] = r[192 + g * 3 + c];
            float* R = &rot[(n * GS_NGAUSS + g) * 4];
            for (int c = 0; c < 4; c++) R[c] = r[288 + g * 4 + c] * 0.1f;
            op[n * GS_NGAUSS + g] = r[416 + g];
        }
    }
    if (!write_gaussian_ply(out_path, n_gs_total, xyz, fdc, op, scl, rot)) {
        LOGE("e2e: failed to write %s", out_path.c_str());
        return 1;
    }
    LOGI("e2e: wrote %s (%lld gaussians) in %.1fs total", out_path.c_str(),
         (long long)n_gs_total, (double)clock() / CLOCKS_PER_SEC - t_start);
    return 0;
}

// Legacy entry point kept for the remaining CLI commands.
RunResult run_pipeline(const CliOptions& opts) {
    RunResult r;
    if (opts.condition_dir.empty() == opts.input_rgba.empty()) {
        r.error = "provide exactly one S3DOBJ01 input or parity condition directory";
        return r;
    }
    if (opts.out_ply.empty() == opts.out_glb.empty()) {
        r.error = "provide exactly one out_glb or out_ply path";
        return r;
    }
    if (opts.models_dir.empty()) {
        r.error = "models_dir is required";
        return r;
    }
    const std::string& output_path = opts.out_glb.empty() ? opts.out_ply : opts.out_glb;
    const int rc = cmd_e2e(opts, opts.condition_dir, "", output_path,
                           opts.dump_dir, static_cast<unsigned>(opts.seed));
    r.ok = rc == 0;
    if (!r.ok) r.error = "end-to-end graph session failed; see log output";
    return r;
}

}  // namespace sam3d
