// Inference-only SAM 3D Objects FlexiCubes extraction.
//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam3d {

struct ObjectMesh {
    std::vector<float> vertices;          // xyz, three floats per vertex
    std::vector<int32_t> faces;           // three indices per triangle
    std::vector<float> vertex_attributes; // six sigmoid/interpolated channels per vertex
};

struct MeshExtractionTaps {
    std::vector<int32_t> aggregate_vertex_coords;
    std::vector<float> aggregate_vertex_attributes; // sdf, deform xyz, six attributes
    std::vector<int32_t> surface_cube_indices;
    std::vector<int32_t> case_ids;
    std::vector<int32_t> surface_edges;
    std::vector<float> dual_vertices;
    std::vector<float> dual_attributes;
    std::vector<int32_t> faces;
};

// raw is [cube_count,101] in row-major storage and coords is
// [cube_count,4] (batch,x,y,z). This mirrors SparseFeatures2Mesh followed by
// inference-mode FlexiCubes at resolution 256.
bool extract_object_mesh(const float* raw, const int32_t* coords, int64_t cube_count,
                         ObjectMesh& output, MeshExtractionTaps* taps,
                         std::string* error = nullptr);

}  // namespace sam3d

