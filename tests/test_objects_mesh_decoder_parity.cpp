#include "backend.hpp"
#include "common.hpp"
#include "gguf_loader.hpp"
#include "graph_builder.hpp"
#include "gs_decoder_graph.hpp"
#include "flexicubes_tables.hpp"
#include "mesh_decoder_graph.hpp"
#include "mesh_extractor.hpp"
#include "mesh_io.hpp"
#include "sparse_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Error {
    float max_abs = 0.0f;
    double rel_l2 = 0.0;
    double sign_agreement = 1.0;
};

std::vector<float> load_f32(const std::filesystem::path& path,
                            std::span<const int64_t> expected_ne) {
    sam3d::RawTensor tensor;
    if (!sam3d::load_raw_tensor(path.string(), tensor) || tensor.type != GGML_TYPE_F32 ||
        !std::equal(tensor.ne.begin(), tensor.ne.end(), expected_ne.begin(), expected_ne.end()))
        throw std::runtime_error("invalid F32 parity tensor: " + path.string());
    const float* data = reinterpret_cast<const float*>(tensor.data.data());
    return {data, data + tensor.data.size() / sizeof(float)};
}

std::vector<int32_t> load_i32(const std::filesystem::path& path,
                              std::span<const int64_t> expected_ne) {
    sam3d::RawTensor tensor;
    if (!sam3d::load_raw_tensor(path.string(), tensor) || tensor.type != GGML_TYPE_I32 ||
        !std::equal(tensor.ne.begin(), tensor.ne.end(), expected_ne.begin(), expected_ne.end()))
        throw std::runtime_error("invalid I32 parity tensor: " + path.string());
    const int32_t* data = reinterpret_cast<const int32_t*>(tensor.data.data());
    return {data, data + tensor.data.size() / sizeof(int32_t)};
}

std::vector<float> load_f32_rows(const std::filesystem::path& path, int64_t channels,
                                 int64_t& rows) {
    sam3d::RawTensor tensor;
    if (!sam3d::load_raw_tensor(path.string(), tensor) || tensor.type != GGML_TYPE_F32 ||
        tensor.ne.size() != 2 || tensor.ne[0] != channels || tensor.ne[1] < 0)
        throw std::runtime_error("invalid F32 parity matrix: " + path.string());
    rows = tensor.ne[1];
    const float* data = reinterpret_cast<const float*>(tensor.data.data());
    return {data, data + tensor.data.size() / sizeof(float)};
}

std::vector<int32_t> load_i32_rows(const std::filesystem::path& path, int64_t channels,
                                   int64_t& rows) {
    sam3d::RawTensor tensor;
    if (!sam3d::load_raw_tensor(path.string(), tensor) || tensor.type != GGML_TYPE_I32 ||
        tensor.ne.size() != 2 || tensor.ne[0] != channels || tensor.ne[1] < 0)
        throw std::runtime_error("invalid I32 parity matrix: " + path.string());
    rows = tensor.ne[1];
    const int32_t* data = reinterpret_cast<const int32_t*>(tensor.data.data());
    return {data, data + tensor.data.size() / sizeof(int32_t)};
}

std::vector<int32_t> load_i32_vector(const std::filesystem::path& path) {
    sam3d::RawTensor tensor;
    if (!sam3d::load_raw_tensor(path.string(), tensor) || tensor.type != GGML_TYPE_I32 ||
        tensor.ne.size() != 1 || tensor.ne[0] < 0)
        throw std::runtime_error("invalid I32 parity vector: " + path.string());
    const int32_t* data = reinterpret_cast<const int32_t*>(tensor.data.data());
    return {data, data + tensor.data.size() / sizeof(int32_t)};
}

Error compare(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("parity tensor size mismatch");
    double error2 = 0.0, reference2 = 0.0;
    size_t signs = 0;
    Error result;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error("non-finite parity tensor");
        const double delta = static_cast<double>(actual[i]) - expected[i];
        result.max_abs = std::max(result.max_abs, static_cast<float>(std::abs(delta)));
        error2 += delta * delta;
        reference2 += static_cast<double>(expected[i]) * expected[i];
        signs += (actual[i] < 0.0f) == (expected[i] < 0.0f);
    }
    result.rel_l2 = std::sqrt(error2 / std::max(reference2, std::numeric_limits<double>::min()));
    result.sign_agreement = actual.empty() ? 1.0 : static_cast<double>(signs) / actual.size();
    return result;
}

bool print_gate(const char* name, const Error& error, float max_abs, double rel_l2) {
    const bool pass = error.max_abs <= max_abs && error.rel_l2 <= rel_l2;
    std::cout << name << " max_abs=" << error.max_abs << " rel_l2=" << error.rel_l2
              << " sign_agreement=" << error.sign_agreement
              << " gate=" << (pass ? "pass" : "FAIL") << '\n';
    return pass;
}

bool print_exact_gate(const char* name, std::span<const int32_t> actual,
                      std::span<const int32_t> expected) {
    size_t mismatches = actual.size() == expected.size() ? 0 :
        std::max(actual.size(), expected.size()) - std::min(actual.size(), expected.size());
    const size_t shared = std::min(actual.size(), expected.size());
    for (size_t i = 0; i < shared; ++i) mismatches += actual[i] != expected[i];
    const bool pass = mismatches == 0;
    std::cout << name << " actual=" << actual.size() << " expected=" << expected.size()
              << " mismatches=" << mismatches << " gate=" << (pass ? "pass" : "FAIL") << '\n';
    return pass;
}

void print_exact_diagnostic(const char* name, std::span<const int32_t> actual,
                            std::span<const int32_t> expected) {
    size_t mismatches = actual.size() == expected.size() ? 0 :
        std::max(actual.size(), expected.size()) - std::min(actual.size(), expected.size());
    const size_t shared = std::min(actual.size(), expected.size());
    for (size_t i = 0; i < shared; ++i) mismatches += actual[i] != expected[i];
    std::cout << name << " actual=" << actual.size() << " expected=" << expected.size()
              << " mismatches=" << mismatches
              << " status=" << (mismatches == 0 ? "match" : "floating-boundary-difference") << '\n';
}

std::vector<uint64_t> dual_keys(std::span<const int32_t> surface_cubes,
                                std::span<const int32_t> cases) {
    if (surface_cubes.size() != cases.size()) throw std::runtime_error("surface key size mismatch");
    std::vector<uint64_t> result;
    for (int dual_count = 1; dual_count <= 4; ++dual_count) {
        for (size_t cube = 0; cube < surface_cubes.size(); ++cube) {
            const int count = sam3d::flexicubes_tables::num_dual[cases[cube]];
            if (count != dual_count) continue;
            for (int local = 0; local < count; ++local)
                result.push_back((static_cast<uint64_t>(static_cast<uint32_t>(surface_cubes[cube])) << 3) |
                                 static_cast<uint64_t>(local));
        }
    }
    return result;
}

struct MatchedMeshError {
    Error vertices;
    Error attributes;
    size_t matched = 0;
};

MatchedMeshError compare_matching_vertices(
        std::span<const float> actual_vertices, std::span<const float> actual_attributes,
        std::span<const uint64_t> actual_keys, std::span<const float> expected_vertices,
        std::span<const float> expected_attributes, std::span<const uint64_t> expected_keys) {
    std::unordered_map<uint64_t, size_t> actual_index;
    actual_index.reserve(actual_keys.size() * 2);
    for (size_t index = 0; index < actual_keys.size(); ++index) actual_index.emplace(actual_keys[index], index);
    std::vector<float> matched_actual_vertices, matched_expected_vertices;
    std::vector<float> matched_actual_attributes, matched_expected_attributes;
    for (size_t expected = 0; expected < expected_keys.size(); ++expected) {
        const auto found = actual_index.find(expected_keys[expected]);
        if (found == actual_index.end()) continue;
        const size_t actual = found->second;
        matched_actual_vertices.insert(matched_actual_vertices.end(),
            actual_vertices.begin() + static_cast<ptrdiff_t>(actual * 3),
            actual_vertices.begin() + static_cast<ptrdiff_t>(actual * 3 + 3));
        matched_expected_vertices.insert(matched_expected_vertices.end(),
            expected_vertices.begin() + static_cast<ptrdiff_t>(expected * 3),
            expected_vertices.begin() + static_cast<ptrdiff_t>(expected * 3 + 3));
        matched_actual_attributes.insert(matched_actual_attributes.end(),
            actual_attributes.begin() + static_cast<ptrdiff_t>(actual * 6),
            actual_attributes.begin() + static_cast<ptrdiff_t>(actual * 6 + 6));
        matched_expected_attributes.insert(matched_expected_attributes.end(),
            expected_attributes.begin() + static_cast<ptrdiff_t>(expected * 6),
            expected_attributes.begin() + static_cast<ptrdiff_t>(expected * 6 + 6));
    }
    return {compare(matched_actual_vertices, matched_expected_vertices),
            compare(matched_actual_attributes, matched_expected_attributes),
            matched_actual_vertices.size() / 3};
}

using TriangleKey = std::array<uint64_t, 3>;

std::vector<TriangleKey> triangle_keys(std::span<const int32_t> faces,
                                       std::span<const uint64_t> vertex_keys) {
    if (faces.size() % 3) throw std::runtime_error("invalid triangle index count");
    std::vector<TriangleKey> result;
    result.reserve(faces.size() / 3);
    for (size_t face = 0; face < faces.size(); face += 3) {
        TriangleKey key{};
        for (int corner = 0; corner < 3; ++corner) {
            const int32_t index = faces[face + corner];
            if (index < 0 || static_cast<size_t>(index) >= vertex_keys.size())
                throw std::runtime_error("triangle vertex outside dual-key table");
            key[corner] = vertex_keys[static_cast<size_t>(index)];
        }
        std::sort(key.begin(), key.end());
        result.push_back(key);
    }
    std::sort(result.begin(), result.end());
    return result;
}

size_t topology_symmetric_difference(std::span<const TriangleKey> actual,
                                     std::span<const TriangleKey> expected) {
    size_t i = 0, j = 0, difference = 0;
    while (i < actual.size() && j < expected.size()) {
        if (actual[i] < expected[j]) { ++i; ++difference; }
        else if (expected[j] < actual[i]) { ++j; ++difference; }
        else { ++i; ++j; }
    }
    return difference + actual.size() - i + expected.size() - j;
}

bool upload(sam3d::Backend& backend, const std::vector<ggml_tensor*>& inputs,
            const std::vector<std::shared_ptr<std::vector<int32_t>>>& data) {
    if (inputs.size() != data.size()) return false;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!data[i]) continue;
        ggml_tensor* tensor = inputs[i];
        if (!tensor->buffer) continue;  // debug-stage graphs omit later table leaves
        const auto& values = *data[i];
        const bool ok = tensor->type == GGML_TYPE_F32
            ? backend.set_input_f32(tensor, reinterpret_cast<const float*>(values.data()), values.size())
            : backend.set_input_i32(tensor, values.data(), values.size());
        if (!ok) return false;
    }
    return true;
}

std::vector<float> run_base(sam3d::Backend& backend, const sam3d::GGUFModel& model,
                            std::span<const float> features, std::span<const int32_t> coords,
                            int64_t count) {
    sam3d::GsTables tables;
    if (!tables.build(coords.data(), count)) throw std::runtime_error("mesh base table build failed");
    sam3d::GraphContext context;
    ggml_tensor* input = context.input_f32("mesh_input", {8, count});
    sam3d::GsDecoderGraph graph_builder;
    graph_builder.g = &context;
    graph_builder.m = &model;
    graph_builder.tb = &tables;
    graph_builder.prefix = "meshdec";
    graph_builder.torso_only = true;
    graph_builder.x = input;
    graph_builder.inputs.push_back(input);
    graph_builder.table_data.push_back(nullptr);
    auto outputs = graph_builder.build();
    ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 65536, false);
    ggml_set_output(outputs[0]);
    ggml_build_forward_expand(graph, outputs[0]);
    if (!backend.alloc(graph) || !backend.set_input_f32(input, features.data(), features.size()) ||
        !upload(backend, graph_builder.inputs, graph_builder.table_data) || !backend.run(graph))
        throw std::runtime_error("mesh base execution failed");
    std::vector<float> result;
    if (!backend.get_tensor_f32(outputs[0], result))
        throw std::runtime_error("mesh base readback failed");
    return result;
}

std::vector<float> run_upsample(sam3d::Backend& backend, const sam3d::GGUFModel& model,
                                int level, const sam3d::MeshSubdivideTables& tables,
                                std::span<const float> features,
                                const std::string& debug_stage = {}) {
    const int64_t channels = level == 0 ? 768 : 192;
    sam3d::GraphContext context;
    ggml_tensor* input = context.input_f32("mesh_upsample_input", {channels, tables.parent_count});
    sam3d::MeshUpsampleGraph graph_builder;
    graph_builder.g = &context;
    graph_builder.m = &model;
    graph_builder.tables = &tables;
    graph_builder.level = level;
    graph_builder.debug_stage = debug_stage;
    graph_builder.x = input;
    graph_builder.inputs.push_back(input);
    graph_builder.table_data.push_back(nullptr);
    ggml_tensor* output = graph_builder.build();
    ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 65536, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    if (!backend.alloc(graph) || !backend.set_input_f32(input, features.data(), features.size()) ||
        !upload(backend, graph_builder.inputs, graph_builder.table_data) || !backend.run(graph))
        throw std::runtime_error("mesh upsample execution failed");
    std::vector<float> result;
    if (!backend.get_tensor_f32(output, result))
        throw std::runtime_error("mesh upsample readback failed");
    return result;
}

std::vector<float> run_output(sam3d::Backend& backend, const sam3d::GGUFModel& model,
                              std::span<const float> features, int64_t count) {
    sam3d::GraphContext context;
    ggml_tensor* input = context.input_f32("mesh_output_input", {96, count});
    ggml_tensor* output = sam3d::build_mesh_output_layer(context, model, input);
    ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 64, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    if (!backend.alloc(graph) || !backend.set_input_f32(input, features.data(), features.size()) ||
        !backend.run(graph)) throw std::runtime_error("mesh output execution failed");
    std::vector<float> result;
    if (!backend.get_tensor_f32(output, result))
        throw std::runtime_error("mesh output readback failed");
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: mesh-parity MODULE CPU|Vulkan DEVICE MODEL.gguf FIXTURE_DIR THREADS\n";
        return 2;
    }
    try {
        char* end = nullptr;
        const unsigned long device = std::strtoul(argv[3], &end, 10);
        if (end == argv[3] || *end || device > UINT32_MAX) return 2;
        const int threads = std::stoi(argv[6]);
        if (threads < 1 || threads > 8) throw std::runtime_error("mesh parity is limited to eight CPU threads");
        auto backend = sam3d::Backend::create(argv[1], argv[2], static_cast<uint32_t>(device), threads);
        if (!backend) return 1;
        sam3d::GGUFModel model;
        if (!model.load(argv[4], backend->weights_buffer_type())) return 1;
        const std::filesystem::path fixture = argv[5];

        sam3d::RawTensor feature_tensor;
        if (!sam3d::load_raw_tensor((fixture / "input.features.samt").string(), feature_tensor) ||
            feature_tensor.type != GGML_TYPE_F32 || feature_tensor.ne.size() != 2 || feature_tensor.ne[0] != 8)
            throw std::runtime_error("invalid mesh input feature fixture");
        const int64_t count = feature_tensor.ne[1];
        auto coords = load_i32(fixture / "input.coords.samt", std::vector<int64_t>{4, count});
        const float* feature_data = reinterpret_cast<const float*>(feature_tensor.data.data());
        std::vector<float> features(feature_data, feature_data + feature_tensor.data.size() / sizeof(float));

        bool pass = true;
        auto base = run_base(*backend, model, features, coords, count);
        auto base_ref = load_f32(fixture / "base.block11.feats.samt", std::vector<int64_t>{768, count});
        pass &= print_gate("base.block11", compare(base, base_ref), 1.0f, 5e-3);

        sam3d::MeshSubdivideTables level0;
        if (!level0.build(coords.data(), count)) throw std::runtime_error("level 0 coordinate subdivision failed");
        auto level0_ref_coords = load_i32(fixture / "upsample0.output.coords.samt",
                                          std::vector<int64_t>{4, level0.child_count});
        if (level0.child_coords != level0_ref_coords)
            throw std::runtime_error("level 0 child coordinates differ from upstream");
        auto up0 = run_upsample(*backend, model, 0, level0, base);
        auto up0_ref = load_f32(fixture / "upsample0.output.feats.samt",
                                std::vector<int64_t>{192, level0.child_count});
        pass &= print_gate("upsample0.output", compare(up0, up0_ref), 0.5f, 7.5e-3);
        if (std::getenv("SAM3D_MESH_DIAGNOSTICS")) {
            struct Tap { const char* name; int64_t channels; int64_t rows; };
            const std::vector<Tap> taps{{"norm1", 768, count}, {"silu1", 768, count},
                {"subdivided", 768, level0.child_count}, {"conv1", 192, level0.child_count},
                {"norm2", 192, level0.child_count}, {"silu2", 192, level0.child_count},
                {"conv2", 192, level0.child_count}, {"skip", 192, level0.child_count}};
            for (const Tap& tap : taps) {
                auto actual = run_upsample(*backend, model, 0, level0, base, tap.name);
                auto expected = load_f32(fixture / (std::string("upsample0.") + tap.name + ".feats.samt"),
                                         std::vector<int64_t>{tap.channels, tap.rows});
                print_gate((std::string("upsample0.") + tap.name).c_str(),
                           compare(actual, expected), 0.15f, 1.5e-2);
                if (std::string(tap.name) == "skip") {
                    std::cout << "skip sample native/upstream:";
                    for (size_t i = 0; i < std::min<size_t>(8, actual.size()); ++i)
                        std::cout << ' ' << actual[i] << '/' << expected[i];
                    std::cout << '\n';
                }
            }
            auto skip_input = run_upsample(*backend, model, 0, level0, base, "skip_input");
            std::vector<float> expected_skip_input((size_t)level0.child_count * 768);
            for (int64_t parent = 0; parent < count; ++parent)
                for (int child = 0; child < 8; ++child)
                    std::copy_n(base_ref.data() + parent * 768, 768,
                                expected_skip_input.data() + (parent * 8 + child) * 768);
            print_gate("upsample0.skip_input", compare(skip_input, expected_skip_input), 0.08f, 8e-3);
        }

        sam3d::MeshSubdivideTables level1;
        if (!level1.build(level0.child_coords.data(), level0.child_count))
            throw std::runtime_error("level 1 coordinate subdivision failed");
        auto level1_ref_coords = load_i32(fixture / "upsample1.output.coords.samt",
                                          std::vector<int64_t>{4, level1.child_count});
        if (level1.child_coords != level1_ref_coords)
            throw std::runtime_error("level 1 child coordinates differ from upstream");
        auto up1 = run_upsample(*backend, model, 1, level1, up0);
        auto up1_ref = load_f32(fixture / "upsample1.output.feats.samt",
                                std::vector<int64_t>{96, level1.child_count});
        pass &= print_gate("upsample1.output", compare(up1, up1_ref), 0.20f, 1e-2);

        auto raw = run_output(*backend, model, up1, level1.child_count);
        auto raw_ref = load_f32(fixture / "decoder.raw.feats.samt",
                                std::vector<int64_t>{101, level1.child_count});
        const Error raw_error = compare(raw, raw_ref);
        pass &= print_gate("decoder.raw", raw_error, 0.30f, 1e-2);

        // Isolate extraction parity from accumulated neural error first. This
        // must reproduce upstream's discrete topology exactly when fed its raw
        // decoder tensor, then the native decoder is checked end to end below.
        sam3d::ObjectMesh reference_input_mesh;
        sam3d::MeshExtractionTaps reference_input_taps;
        std::string extraction_error;
        if (!sam3d::extract_object_mesh(raw_ref.data(), level1.child_coords.data(),
                                        level1.child_count, reference_input_mesh,
                                        &reference_input_taps, &extraction_error))
            throw std::runtime_error("upstream-raw extraction failed: " + extraction_error);

        int64_t aggregate_rows = 0, surface_edge_rows = 0, vertex_rows = 0,
                attribute_rows = 0, face_rows = 0;
        const auto aggregate_coords_ref = load_i32_rows(
            fixture / "aggregate.vertex_coords.samt", 3, aggregate_rows);
        int64_t aggregate_attribute_rows = 0;
        const auto aggregate_attrs_ref = load_f32_rows(
            fixture / "aggregate.vertex_attributes.samt", 10, aggregate_attribute_rows);
        if (aggregate_attribute_rows != aggregate_rows)
            throw std::runtime_error("aggregate fixture row mismatch");
        const auto surface_cubes_ref = load_i32_vector(fixture / "mesh.surface_cube_indices.samt");
        const auto cases_ref = load_i32_vector(fixture / "mesh.case_ids.samt");
        const auto surface_edges_ref = load_i32_rows(
            fixture / "mesh.surface_edges.samt", 2, surface_edge_rows);
        const auto vertices_ref = load_f32_rows(fixture / "output.vertices.samt", 3, vertex_rows);
        const auto attributes_ref = load_f32_rows(
            fixture / "output.vertex_attributes.samt", 6, attribute_rows);
        const auto faces_ref = load_i32_rows(fixture / "output.faces.samt", 3, face_rows);
        if (attribute_rows != vertex_rows || cases_ref.size() != surface_cubes_ref.size())
            throw std::runtime_error("mesh fixture row mismatch");
        std::cout << "reference.mesh aggregate_vertices=" << aggregate_rows
                  << " surface_edges=" << surface_edge_rows << " vertices=" << vertex_rows
                  << " faces=" << face_rows << '\n';

        pass &= print_exact_gate("extract.aggregate.coords", reference_input_taps.aggregate_vertex_coords,
                                 aggregate_coords_ref);
        pass &= print_gate("extract.aggregate.attributes",
                           compare(reference_input_taps.aggregate_vertex_attributes, aggregate_attrs_ref),
                           2e-5f, 2e-6);
        pass &= print_exact_gate("extract.surface_cubes", reference_input_taps.surface_cube_indices,
                                 surface_cubes_ref);
        pass &= print_exact_gate("extract.case_ids", reference_input_taps.case_ids, cases_ref);
        pass &= print_exact_gate("extract.surface_edges", reference_input_taps.surface_edges,
                                 surface_edges_ref);
        pass &= print_gate("extract.vertices", compare(reference_input_mesh.vertices, vertices_ref),
                           2e-5f, 2e-5);
        pass &= print_gate("extract.attributes",
                           compare(reference_input_mesh.vertex_attributes, attributes_ref), 2e-5f, 2e-5);
        pass &= print_exact_gate("extract.faces", reference_input_mesh.faces, faces_ref);
        if (const char* glb_path = std::getenv("SAM3D_MESH_PARITY_GLB")) {
            std::string write_error;
            if (!sam3d::write_object_glb(glb_path, reference_input_mesh, &write_error))
                throw std::runtime_error("parity GLB export failed: " + write_error);
        }

        sam3d::ObjectMesh native_mesh;
        sam3d::MeshExtractionTaps native_taps;
        if (!sam3d::extract_object_mesh(raw.data(), level1.child_coords.data(), level1.child_count,
                                        native_mesh, &native_taps, &extraction_error))
            throw std::runtime_error("native extraction failed: " + extraction_error);
        pass &= print_gate("end_to_end.aggregate.attributes",
                           compare(native_taps.aggregate_vertex_attributes, aggregate_attrs_ref),
                           0.20f, 2e-2);
        size_t sdf_sign_mismatches = 0;
        float largest_reference_sdf_at_mismatch = 0.0f;
        float largest_native_sdf_at_mismatch = 0.0f;
        for (size_t row = 0; row < aggregate_attrs_ref.size() / 10; ++row) {
            const float actual_sdf = native_taps.aggregate_vertex_attributes[row * 10];
            const float expected_sdf = aggregate_attrs_ref[row * 10];
            if ((actual_sdf < 0.0f) != (expected_sdf < 0.0f)) {
                ++sdf_sign_mismatches;
                largest_reference_sdf_at_mismatch = std::max(
                    largest_reference_sdf_at_mismatch, std::abs(expected_sdf));
                largest_native_sdf_at_mismatch = std::max(
                    largest_native_sdf_at_mismatch, std::abs(actual_sdf));
                if (sdf_sign_mismatches <= 8)
                    std::cout << "sdf sign mismatch row=" << row << " native=" << actual_sdf
                              << " upstream=" << expected_sdf << '\n';
            }
        }
        const size_t sdf_mismatch_limit = std::max<size_t>(2, aggregate_rows / 1000);
        const bool sdf_gate = sdf_sign_mismatches <= sdf_mismatch_limit &&
                              largest_reference_sdf_at_mismatch <= 5e-4f &&
                              largest_native_sdf_at_mismatch <= 5e-4f;
        std::cout << "end_to_end.sdf_signs mismatches=" << sdf_sign_mismatches
                  << " limit=" << sdf_mismatch_limit
                  << " largest_upstream_margin=" << largest_reference_sdf_at_mismatch
                  << " largest_native_margin=" << largest_native_sdf_at_mismatch
                  << " gate=" << (sdf_gate ? "pass" : "FAIL") << '\n';
        pass &= sdf_gate;

        // Exact reports remain useful diagnostics. The acceptance comparison
        // below keys dual vertices by their source cube, so a single SDF sign
        // at numerical zero does not shift every subsequent vertex index.
        print_exact_diagnostic("end_to_end.surface_cubes.exact", native_taps.surface_cube_indices,
                               surface_cubes_ref);
        print_exact_diagnostic("end_to_end.case_ids.exact", native_taps.case_ids, cases_ref);
        print_exact_diagnostic("end_to_end.faces.exact", native_mesh.faces, faces_ref);
        const auto expected_keys = dual_keys(surface_cubes_ref, cases_ref);
        const auto actual_keys = dual_keys(native_taps.surface_cube_indices, native_taps.case_ids);
        if (expected_keys.size() != vertices_ref.size() / 3 ||
            actual_keys.size() != native_mesh.vertices.size() / 3)
            throw std::runtime_error("dual-key count differs from mesh vertex count");
        const MatchedMeshError matched = compare_matching_vertices(
            native_mesh.vertices, native_mesh.vertex_attributes, actual_keys,
            vertices_ref, attributes_ref, expected_keys);
        const size_t unmatched = actual_keys.size() + expected_keys.size() - 2 * matched.matched;
        const size_t unmatched_limit = std::max<size_t>(2, expected_keys.size() / 250);
        const bool structure_gate = unmatched <= unmatched_limit;
        std::cout << "end_to_end.dual_structure matched=" << matched.matched
                  << " unmatched=" << unmatched << " limit=" << unmatched_limit
                  << " gate=" << (structure_gate ? "pass" : "FAIL") << '\n';
        pass &= structure_gate;
        pass &= print_gate("end_to_end.matched_vertices", matched.vertices, 1.0f / 256.0f, 5e-3);
        pass &= print_gate("end_to_end.matched_attributes", matched.attributes, 0.125f, 5e-3);

        const auto expected_triangles = triangle_keys(faces_ref, expected_keys);
        const auto actual_triangles = triangle_keys(native_mesh.faces, actual_keys);
        const size_t triangle_difference = topology_symmetric_difference(
            actual_triangles, expected_triangles);
        const size_t triangle_difference_limit = std::max<size_t>(32, expected_triangles.size() / 25);
        const bool topology_gate = triangle_difference <= triangle_difference_limit;
        std::cout << "end_to_end.topology symmetric_triangle_difference=" << triangle_difference
                  << " limit=" << triangle_difference_limit
                  << " native_faces=" << actual_triangles.size()
                  << " upstream_faces=" << expected_triangles.size()
                  << " gate=" << (topology_gate ? "pass" : "FAIL") << '\n';
        pass &= topology_gate;
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
