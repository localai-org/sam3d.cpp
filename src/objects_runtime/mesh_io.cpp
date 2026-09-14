#include "mesh_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

namespace sam3d {
namespace {

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value >> 16));
    output.push_back(static_cast<uint8_t>(value >> 24));
}

void append_bytes(std::vector<uint8_t>& output, const void* source, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(source);
    output.insert(output.end(), bytes, bytes + size);
}

void align4(std::vector<uint8_t>& output, uint8_t fill = 0) {
    while (output.size() % 4) output.push_back(fill);
}

bool fail(std::string* error, const char* message) {
    if (error) *error = message;
    return false;
}

}  // namespace

bool write_object_glb(const std::string& path, const ObjectMesh& mesh, std::string* error) {
    if (mesh.vertices.empty() || mesh.vertices.size() % 3 || mesh.faces.empty() ||
        mesh.faces.size() % 3 || mesh.vertex_attributes.size() / 6 != mesh.vertices.size() / 3)
        return fail(error, "invalid object mesh");
    const size_t vertex_count = mesh.vertices.size() / 3;
    if (vertex_count > UINT32_MAX || mesh.faces.size() / 3 > UINT32_MAX)
        return fail(error, "object mesh exceeds GLB limits");

    std::vector<float> positions(mesh.vertices.size());
    std::vector<float> colors(vertex_count * 3);
    std::array<float, 3> minimum{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    std::array<float, 3> maximum{
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
        // Same row-vector transform as upstream to_glb: (x, y, z) -> (x, z, -y).
        const float converted[3] = {mesh.vertices[vertex * 3], mesh.vertices[vertex * 3 + 2],
                                    -mesh.vertices[vertex * 3 + 1]};
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(converted[axis])) return fail(error, "non-finite mesh position");
            positions[vertex * 3 + axis] = converted[axis];
            minimum[axis] = std::min(minimum[axis], converted[axis]);
            maximum[axis] = std::max(maximum[axis], converted[axis]);
        }
        for (int channel = 0; channel < 3; ++channel) {
            const float value = mesh.vertex_attributes[vertex * 6 + channel];
            if (!std::isfinite(value)) return fail(error, "non-finite mesh colour");
            colors[vertex * 3 + channel] = std::clamp(value, 0.0f, 1.0f);
        }
    }
    for (int32_t index : mesh.faces)
        if (index < 0 || static_cast<size_t>(index) >= vertex_count)
            return fail(error, "mesh face index out of range");

    std::vector<uint8_t> binary;
    const size_t position_offset = binary.size();
    append_bytes(binary, positions.data(), positions.size() * sizeof(float));
    align4(binary);
    const size_t color_offset = binary.size();
    append_bytes(binary, colors.data(), colors.size() * sizeof(float));
    align4(binary);
    const size_t index_offset = binary.size();
    append_bytes(binary, mesh.faces.data(), mesh.faces.size() * sizeof(int32_t));
    align4(binary);

    std::ostringstream json;
    json << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"sam3d.cpp\"},"
         << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
         << "\"meshes\":[{\"name\":\"SAM 3D Object\",\"primitives\":[{\"attributes\":{"
         << "\"POSITION\":0,\"COLOR_0\":1},\"indices\":2,\"mode\":4,\"material\":0}]}],"
         << "\"materials\":[{\"name\":\"Learned vertex colour\",\"pbrMetallicRoughness\":{"
         << "\"baseColorFactor\":[1,1,1,1],\"metallicFactor\":0,\"roughnessFactor\":1},"
         << "\"doubleSided\":true}],\"buffers\":[{\"byteLength\":" << binary.size() << "}],"
         << "\"bufferViews\":["
         << "{\"buffer\":0,\"byteOffset\":" << position_offset << ",\"byteLength\":"
         << positions.size() * sizeof(float) << ",\"target\":34962},"
         << "{\"buffer\":0,\"byteOffset\":" << color_offset << ",\"byteLength\":"
         << colors.size() * sizeof(float) << ",\"target\":34962},"
         << "{\"buffer\":0,\"byteOffset\":" << index_offset << ",\"byteLength\":"
         << mesh.faces.size() * sizeof(int32_t) << ",\"target\":34963}],"
         << "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,\"count\":" << vertex_count
         << ",\"type\":\"VEC3\",\"min\":[" << minimum[0] << ',' << minimum[1] << ',' << minimum[2]
         << "],\"max\":[" << maximum[0] << ',' << maximum[1] << ',' << maximum[2] << "]},"
         << "{\"bufferView\":1,\"componentType\":5126,\"count\":" << vertex_count
         << ",\"type\":\"VEC3\"},{\"bufferView\":2,\"componentType\":5125,\"count\":"
         << mesh.faces.size() << ",\"type\":\"SCALAR\"}]}";
    std::string json_text = json.str();
    while (json_text.size() % 4) json_text.push_back(' ');

    const uint64_t total_size = 12ull + 8 + json_text.size() + 8 + binary.size();
    if (total_size > UINT32_MAX) return fail(error, "GLB exceeds four gigabytes");
    std::vector<uint8_t> glb;
    glb.reserve(static_cast<size_t>(total_size));
    append_u32(glb, 0x46546c67);  // glTF
    append_u32(glb, 2);
    append_u32(glb, static_cast<uint32_t>(total_size));
    append_u32(glb, static_cast<uint32_t>(json_text.size()));
    append_u32(glb, 0x4e4f534a);  // JSON
    append_bytes(glb, json_text.data(), json_text.size());
    append_u32(glb, static_cast<uint32_t>(binary.size()));
    append_u32(glb, 0x004e4942);  // BIN
    append_bytes(glb, binary.data(), binary.size());

    std::ofstream output(path, std::ios::binary);
    if (!output || !output.write(reinterpret_cast<const char*>(glb.data()),
                                 static_cast<std::streamsize>(glb.size())))
        return fail(error, "failed to write object GLB");
    return true;
}

}  // namespace sam3d
