// SAM 3D Objects sparse mesh decoder head.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph_builder.hpp"
#include "gguf_loader.hpp"
#include "sparse_ops.hpp"

namespace sam3d {

struct MeshUpsampleGraph {
    GraphContext* g = nullptr;
    const GGUFModel* m = nullptr;
    const MeshSubdivideTables* tables = nullptr;
    int level = 0;
    std::string debug_stage;  // norm1|silu1|subdivided|conv1|norm2|silu2|conv2|skip|output

    ggml_tensor* x = nullptr;  // parent features [Cin, parent_count]
    std::vector<ggml_tensor*> inputs;
    // F32 payloads are held as their exact I32 bit pattern, matching the
    // table upload convention used by the existing SLat decoder graphs.
    std::vector<std::shared_ptr<std::vector<int32_t>>> table_data;

    ggml_tensor* build();
};

ggml_tensor* build_mesh_output_layer(GraphContext& graph, const GGUFModel& model,
                                     ggml_tensor* features);

}  // namespace sam3d

