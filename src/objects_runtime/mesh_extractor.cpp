#include "mesh_extractor.hpp"

#include "flexicubes_tables.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

namespace sam3d {
namespace {

constexpr int kResolution = 256;
constexpr int kVertexResolution = kResolution + 1;
constexpr int kRawChannels = 101;
constexpr int kAttrChannels = 10;
constexpr int kColorChannels = 6;
constexpr std::array<std::array<int32_t, 3>, 8> kCorners{{
    {{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{1, 1, 0}},
    {{0, 0, 1}}, {{1, 0, 1}}, {{0, 1, 1}}, {{1, 1, 1}},
}};
constexpr std::array<int32_t, 24> kEdges{{
    0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7,
    6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4,
}};
constexpr std::array<int32_t, 6> kSplit1{{0, 1, 2, 0, 2, 3}};
constexpr std::array<int32_t, 6> kSplit2{{0, 1, 3, 3, 1, 2}};

int32_t vertex_id(int32_t x, int32_t y, int32_t z) {
    return (x * kVertexResolution + y) * kVertexResolution + z;
}

int32_t cube_id(int32_t x, int32_t y, int32_t z) {
    return (x * kResolution + y) * kResolution + z;
}

std::array<int32_t, 3> vertex_coord(int32_t id) {
    return {id / (kVertexResolution * kVertexResolution),
            (id / kVertexResolution) % kVertexResolution,
            id % kVertexResolution};
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

void fail(std::string* error, const char* message) {
    if (error) *error = message;
}

struct Aggregate {
    std::array<float, kAttrChannels> sum{};
    int32_t count = 0;
};

struct SurfaceCube {
    int32_t dense_id = 0;
    int32_t source_row = -1;
    uint8_t occupancy = 0;
    int32_t case_id = 0;
    std::array<int32_t, 8> vertices{};
};

struct TransformedVertex {
    std::array<float, 3> position{};
    std::array<float, kColorChannels> color{};
};

TransformedVertex transformed_vertex(
    int32_t dense_vertex,
    const std::unordered_map<int32_t, TransformedVertex>& attributes) {
    const auto found = attributes.find(dense_vertex);
    if (found != attributes.end()) return found->second;
    TransformedVertex result;
    const auto coord = vertex_coord(dense_vertex);
    for (int axis = 0; axis < 3; ++axis)
        result.position[axis] = static_cast<float>(coord[axis]) / kResolution - 0.5f;
    result.color.fill(0.5f);
    return result;
}

uint64_t edge_key(int32_t first, int32_t second) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(first)) << 32) |
           static_cast<uint32_t>(second);
}

}  // namespace

bool extract_object_mesh(const float* raw, const int32_t* coords, int64_t cube_count,
                         ObjectMesh& output, MeshExtractionTaps* taps,
                         std::string* error) {
    output = {};
    if (taps) *taps = {};
    if (!raw || !coords || cube_count <= 0 || cube_count > INT32_MAX) {
        fail(error, "invalid mesh decoder tensor");
        return false;
    }

    // sparse_cube2verts: torch.unique(..., dim=0) establishes lexicographic
    // coordinate order before the include_self=False mean reduction.
    std::unordered_map<int32_t, Aggregate> aggregate;
    aggregate.max_load_factor(0.7f);
    aggregate.reserve(static_cast<size_t>(cube_count) * 2);
    std::unordered_map<int32_t, int32_t> cube_source;
    cube_source.reserve((size_t)cube_count * 2);
    for (int64_t row = 0; row < cube_count; ++row) {
        if (coords[row * 4] != 0) {
            fail(error, "mesh extraction currently requires batch one");
            return false;
        }
        const int32_t x = coords[row * 4 + 1];
        const int32_t y = coords[row * 4 + 2];
        const int32_t z = coords[row * 4 + 3];
        if (x < 0 || y < 0 || z < 0 || x >= kResolution || y >= kResolution || z >= kResolution) {
            fail(error, "mesh cube coordinate outside resolution 256");
            return false;
        }
        if (!cube_source.emplace(cube_id(x, y, z), static_cast<int32_t>(row)).second) {
            fail(error, "duplicate mesh cube coordinate");
            return false;
        }
        const float* features = raw + row * kRawChannels;
        for (int corner = 0; corner < 8; ++corner) {
            const int32_t dense = vertex_id(x + kCorners[corner][0],
                                            y + kCorners[corner][1],
                                            z + kCorners[corner][2]);
            auto& value = aggregate[dense];
            value.sum[0] += features[corner] - 1.0f / kResolution;
            for (int axis = 0; axis < 3; ++axis)
                value.sum[1 + axis] += features[8 + corner * 3 + axis];
            for (int channel = 0; channel < kColorChannels; ++channel)
                value.sum[4 + channel] += features[53 + corner * kColorChannels + channel];
            ++value.count;
        }
    }

    const size_t dense_vertex_count = (size_t)kVertexResolution * kVertexResolution * kVertexResolution;
    std::vector<float> sdf(dense_vertex_count, 1.0f);
    std::unordered_map<int32_t, TransformedVertex> vertex_attrs;
    vertex_attrs.reserve(aggregate.size() * 2);
    if (taps) {
        taps->aggregate_vertex_coords.reserve(aggregate.size() * 3);
        taps->aggregate_vertex_attributes.reserve(aggregate.size() * kAttrChannels);
    }
    const auto store_aggregate = [&](int32_t dense, const Aggregate& value) {
        const auto coord = vertex_coord(dense);
        std::array<float, kAttrChannels> mean{};
        for (int channel = 0; channel < kAttrChannels; ++channel)
            mean[channel] = value.sum[channel] / value.count;
        sdf[(size_t)dense] = mean[0];
        TransformedVertex transformed;
        for (int axis = 0; axis < 3; ++axis) {
            transformed.position[axis] = static_cast<float>(coord[axis]) / kResolution - 0.5f +
                                         (1.0f / (kResolution * 2.0f)) *
                                             std::tanh(mean[1 + axis]);
        }
        for (int channel = 0; channel < kColorChannels; ++channel)
            transformed.color[channel] = sigmoid(mean[4 + channel]);
        vertex_attrs.emplace(dense, transformed);
        if (taps) {
            taps->aggregate_vertex_coords.insert(
                taps->aggregate_vertex_coords.end(), {coord[0], coord[1], coord[2]});
            taps->aggregate_vertex_attributes.insert(taps->aggregate_vertex_attributes.end(), mean.begin(), mean.end());
        }
    };
    if (taps) {
        // Captures follow torch.unique's lexicographic coordinate order.
        std::vector<int32_t> aggregate_vertices;
        aggregate_vertices.reserve(aggregate.size());
        for (const auto& [dense, unused] : aggregate) aggregate_vertices.push_back(dense);
        std::sort(aggregate_vertices.begin(), aggregate_vertices.end());
        for (int32_t dense : aggregate_vertices) store_aggregate(dense, aggregate.at(dense));
    } else {
        // Production extraction addresses these values by dense vertex id, so
        // iteration order cannot affect geometry and sorting would be wasted.
        for (const auto& [dense, value] : aggregate) store_aggregate(dense, value);
    }

    // Identify surface cubes in the same dense lexicographic order produced
    // by torch.nonzero over the [256,256,256] mask.
    std::vector<SurfaceCube> surfaces;
    std::vector<int16_t> dense_case((size_t)kResolution * kResolution * kResolution, -1);
    for (int32_t x = 0; x < kResolution; ++x) {
        for (int32_t y = 0; y < kResolution; ++y) {
            for (int32_t z = 0; z < kResolution; ++z) {
                SurfaceCube cube;
                cube.dense_id = cube_id(x, y, z);
                auto source = cube_source.find(cube.dense_id);
                cube.source_row = source == cube_source.end() ? -1 : source->second;
                int occupied = 0;
                for (int corner = 0; corner < 8; ++corner) {
                    const int32_t id = vertex_id(x + kCorners[corner][0],
                                                 y + kCorners[corner][1],
                                                 z + kCorners[corner][2]);
                    cube.vertices[corner] = id;
                    if (sdf[(size_t)id] < 0.0f) {
                        cube.occupancy |= static_cast<uint8_t>(1u << corner);
                        ++occupied;
                    }
                }
                if (occupied == 0 || occupied == 8) continue;
                cube.case_id = cube.occupancy;
                dense_case[(size_t)cube.dense_id] = static_cast<int16_t>(cube.case_id);
                surfaces.push_back(cube);
            }
        }
    }

    // Resolve the same adjacent ambiguous configurations as _get_case_id.
    for (SurfaceCube& cube : surfaces) {
        const int16_t* check = flexicubes_tables::check[cube.case_id];
        if (check[0] != 1) continue;
        const int32_t x = cube.dense_id / (kResolution * kResolution);
        const int32_t y = (cube.dense_id / kResolution) % kResolution;
        const int32_t z = cube.dense_id % kResolution;
        const int32_t nx = x + check[1], ny = y + check[2], nz = z + check[3];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= kResolution || ny >= kResolution || nz >= kResolution)
            continue;
        const int neighbor_case = dense_case[(size_t)cube_id(nx, ny, nz)];
        if (neighbor_case >= 0 && flexicubes_tables::check[neighbor_case][0] == 1)
            cube.case_id = check[4];
    }
    if (taps) {
        taps->surface_cube_indices.reserve(surfaces.size());
        taps->case_ids.reserve(surfaces.size());
        for (const SurfaceCube& cube : surfaces) {
            taps->surface_cube_indices.push_back(cube.dense_id);
            taps->case_ids.push_back(cube.case_id);
        }
    }

    struct EdgeInfo { int32_t count = 0; int32_t surface_id = -1; };
    std::unordered_map<uint64_t, EdgeInfo> unique_edges;
    unique_edges.max_load_factor(0.7f);
    unique_edges.reserve(surfaces.size() * 4);
    for (const SurfaceCube& cube : surfaces) {
        for (int edge = 0; edge < 12; ++edge) {
            const uint64_t key = edge_key(cube.vertices[kEdges[edge * 2]],
                                          cube.vertices[kEdges[edge * 2 + 1]]);
            ++unique_edges[key].count;
        }
    }
    std::vector<uint64_t> edge_keys;
    edge_keys.reserve(unique_edges.size());
    for (const auto& [key, unused] : unique_edges) edge_keys.push_back(key);
    std::sort(edge_keys.begin(), edge_keys.end());
    std::vector<std::array<int32_t, 2>> surface_edges;
    for (uint64_t key : edge_keys) {
        const int32_t first = static_cast<int32_t>(key >> 32);
        const int32_t second = static_cast<int32_t>(key);
        EdgeInfo& info = unique_edges.at(key);
        if ((sdf[(size_t)first] < 0.0f) != (sdf[(size_t)second] < 0.0f)) {
            info.surface_id = static_cast<int32_t>(surface_edges.size());
            surface_edges.push_back({first, second});
            if (taps) taps->surface_edges.insert(taps->surface_edges.end(), {first, second});
        }
    }
    const size_t occurrence_count = surfaces.size() * 12;
    std::vector<int32_t> edge_map(occurrence_count, -1), edge_count(occurrence_count, 0);
    std::vector<uint8_t> edge_surface(occurrence_count, 0);
    for (size_t cube_index = 0; cube_index < surfaces.size(); ++cube_index) {
        const SurfaceCube& cube = surfaces[cube_index];
        for (int edge = 0; edge < 12; ++edge) {
            const uint64_t key = edge_key(cube.vertices[kEdges[edge * 2]],
                                          cube.vertices[kEdges[edge * 2 + 1]]);
            const EdgeInfo& info = unique_edges.at(key);
            const size_t occurrence = cube_index * 12 + edge;
            edge_map[occurrence] = info.surface_id;
            edge_count[occurrence] = info.count;
            edge_surface[occurrence] = info.surface_id >= 0;
        }
    }

    auto raw_weight = [&](const SurfaceCube& cube, int channel) {
        return cube.source_row < 0 ? 0.0f : raw[(size_t)cube.source_row * kRawChannels + 32 + channel];
    };
    std::vector<std::array<float, 12>> beta(surfaces.size());
    std::vector<std::array<float, 8>> alpha(surfaces.size());
    std::vector<float> gamma(surfaces.size());
    for (size_t i = 0; i < surfaces.size(); ++i) {
        for (int edge = 0; edge < 12; ++edge) beta[i][edge] = std::tanh(raw_weight(surfaces[i], edge)) * 0.99f + 1.0f;
        for (int corner = 0; corner < 8; ++corner) alpha[i][corner] = std::tanh(raw_weight(surfaces[i], 12 + corner)) * 0.99f + 1.0f;
        gamma[i] = sigmoid(raw_weight(surfaces[i], 20)) * 0.99f + 0.005f;
    }

    std::vector<int32_t> dual_index_map(occurrence_count, 0);
    std::vector<float> dual_gamma;
    std::set<int> dual_counts;
    for (const SurfaceCube& cube : surfaces) dual_counts.insert(flexicubes_tables::num_dual[cube.case_id]);
    for (int count : dual_counts) {
        for (size_t cube_index = 0; cube_index < surfaces.size(); ++cube_index) {
            const SurfaceCube& cube = surfaces[cube_index];
            if (flexicubes_tables::num_dual[cube.case_id] != count) continue;
            for (int local_dual = 0; local_dual < count; ++local_dual) {
                std::array<float, 3> position{};
                std::array<float, kColorChannels> color{};
                float beta_sum = 0.0f;
                const int32_t dual_id = static_cast<int32_t>(output.vertices.size() / 3);
                for (int table_slot = 0; table_slot < 7; ++table_slot) {
                    const int edge = flexicubes_tables::dmc[cube.case_id][local_dual][table_slot];
                    if (edge < 0) continue;
                    const int32_t a = cube.vertices[kEdges[edge * 2]];
                    const int32_t b = cube.vertices[kEdges[edge * 2 + 1]];
                    const float wa = sdf[(size_t)a] * alpha[cube_index][kEdges[edge * 2]];
                    const float wb = sdf[(size_t)b] * alpha[cube_index][kEdges[edge * 2 + 1]];
                    const float denominator = wb - wa;
                    const TransformedVertex va = transformed_vertex(a, vertex_attrs);
                    const TransformedVertex vb = transformed_vertex(b, vertex_attrs);
                    const float edge_beta = beta[cube_index][edge];
                    beta_sum += edge_beta;
                    for (int axis = 0; axis < 3; ++axis)
                        position[axis] += ((va.position[axis] * wb - vb.position[axis] * wa) /
                                           denominator) * edge_beta;
                    for (int channel = 0; channel < kColorChannels; ++channel)
                        color[channel] += ((va.color[channel] * wb - vb.color[channel] * wa) /
                                          denominator) * edge_beta;
                    dual_index_map[cube_index * 12 + edge] = dual_id;
                }
                for (float& value : position) value /= beta_sum;
                for (float& value : color) value /= beta_sum;
                output.vertices.insert(output.vertices.end(), position.begin(), position.end());
                output.vertex_attributes.insert(output.vertex_attributes.end(), color.begin(), color.end());
                dual_gamma.push_back(gamma[cube_index]);
            }
        }
    }

    std::vector<size_t> selected_occurrences;
    selected_occurrences.reserve(occurrence_count);
    for (size_t occurrence = 0; occurrence < occurrence_count; ++occurrence)
        if (edge_count[occurrence] == 4 && edge_surface[occurrence]) selected_occurrences.push_back(occurrence);
    std::stable_sort(selected_occurrences.begin(), selected_occurrences.end(),
        [&](size_t a, size_t b) { return edge_map[a] < edge_map[b]; });
    if (selected_occurrences.size() % 4 != 0) {
        fail(error, "FlexiCubes edge group does not contain four incident cubes");
        return false;
    }

    struct Quad { std::array<int32_t, 4> vertices; bool flip; };
    std::vector<Quad> quads;
    quads.reserve(selected_occurrences.size() / 4);
    for (size_t group = 0; group < selected_occurrences.size(); group += 4) {
        Quad quad{};
        const int32_t edge_id = edge_map[selected_occurrences[group]];
        for (int i = 0; i < 4; ++i)
            quad.vertices[i] = dual_index_map[selected_occurrences[group + i]];
        quad.flip = sdf[(size_t)surface_edges[(size_t)edge_id][0]] > 0.0f;
        quads.push_back(quad);
    }
    auto emit_quad = [&](const Quad& source) {
        std::array<int32_t, 4> quad = source.flip
            ? std::array<int32_t, 4>{source.vertices[0], source.vertices[1], source.vertices[3], source.vertices[2]}
            : std::array<int32_t, 4>{source.vertices[2], source.vertices[3], source.vertices[1], source.vertices[0]};
        const bool first = dual_gamma[(size_t)quad[0]] * dual_gamma[(size_t)quad[2]] >
                           dual_gamma[(size_t)quad[1]] * dual_gamma[(size_t)quad[3]];
        const auto& split = first ? kSplit1 : kSplit2;
        for (int index : split) output.faces.push_back(quad[index]);
    };
    // torch.cat((flipped_quads, nonflipped_quads)) changes group order here.
    for (const Quad& quad : quads) if (quad.flip) emit_quad(quad);
    for (const Quad& quad : quads) if (!quad.flip) emit_quad(quad);

    if (taps) {
        taps->dual_vertices = output.vertices;
        taps->dual_attributes = output.vertex_attributes;
        taps->faces = output.faces;
    }
    return true;
}

}  // namespace sam3d
