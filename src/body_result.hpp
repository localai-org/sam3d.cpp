#pragma once
#include "sam3d_model.h"
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>
// Internal conversion from the native branch's own output, not a public hook.
namespace sam3d {
struct body_output_field {const char *name,*source;std::vector<uint64_t> shape;};
const std::vector<body_output_field> &body_output_fields();
std::unique_ptr<s3d_body_result,decltype(&s3d_body_result_free)> make_body_result(
    std::map<std::string,std::vector<float>> values,std::span<const int32_t> faces);
}
