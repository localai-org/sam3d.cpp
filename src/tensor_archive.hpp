#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "validated_weights.hpp"

namespace sam3d {
// Internal model-component loader. No backend, Python or geometry allocation.
// Expected shapes are a compiled architecture contract, never file-supplied.
using tensor_shapes = std::map<std::string,std::vector<uint64_t>>; // PyTorch axis order
tensor_shapes body_dino_shapes();
tensor_shapes body_branch_shapes();
bool body_branch_integer_tensor(const std::string &name);
tensor_shapes mhr_lod1_shapes();
bool mhr_integer_tensor(const std::string &name);
class tensor_archive {
public:
    tensor_archive(const std::filesystem::path &, const std::string &expected_architecture,
                   const tensor_shapes &expected_shapes);
    std::vector<float> read(const std::string &name, uint64_t max_bytes);
    validated_weights load(const std::string &name, uint64_t max_bytes);
    // Retain an immutable checked snapshot; budget applies to total pinned
    // payload, not only one read. Does not skip validation of fresh file reads.
    validated_weights pin(const std::string &name, uint64_t max_bytes);
    // Fixed MHR COO -> dense layout conversion, cached in the same byte budget.
    // Source index/value validation is identical to fresh reads; never a caller
    // supplied cache key or mutable buffer masquerading as immutable weights.
    validated_weights pin_mhr_sparse_projection();
    std::vector<int32_t> read_i32(const std::string &name, uint64_t max_bytes);
    const std::string &metadata(const std::string &key) const;
    size_t tensor_count() const { return tensors_.size(); }
private:
    struct tensor_info { uint64_t offset, bytes; uint32_t type; };
    std::ifstream file_;
    std::mutex mutex_;
    std::map<std::string,tensor_info> tensors_;
    std::map<std::string,std::string> metadata_;
    std::mutex pinned_mutex_;
    std::map<std::string,validated_weights> pinned_;
    uint64_t pinned_bytes_ = 0;
    static constexpr uint64_t pinned_limit_ = 768ULL*1024*1024;
};
}
