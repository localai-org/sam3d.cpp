#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "validated_weights.hpp"
#include <cmath>
#include <limits>

namespace sam3d {
// BF16 arithmetic boundaries with F32 carriers for GGML elementwise operations
// which do not accept BF16. This is not F16 and does not suppress BF16 rounding.
inline ggml_tensor *bf16_round(ggml_context *c,ggml_tensor *x){
    return ggml_cast(c,ggml_cast(c,x,GGML_TYPE_BF16),GGML_TYPE_F32);
}
inline std::vector<ggml_bf16_t> bf16_storage(std::span<const float> values){
    // Pinned GGML row helpers use an int loop even though their count is int64.
    if(values.size()>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("BF16 row exceeds conversion bound");
    std::vector<ggml_bf16_t> result(values.size());
    // Use the reference row conversion: hardware BF16 conversion may flush
    // subnormals, which would change the established scalar rounding contract.
    ggml_fp32_to_bf16_row_ref(values.data(),result.data(),int64_t(values.size()));
    for(auto value:result)if((value.bits&0x7f80u)==0x7f80u)
        throw std::invalid_argument("nonfinite BF16 source or conversion overflow");
    return result;
}
inline std::vector<float> bf16_values(std::span<const float> values){
    auto data=bf16_storage(values);std::vector<float> result(values.size());
    ggml_bf16_to_fp32_row(data.data(),result.data(),int64_t(values.size()));
    return result;
}
inline void set_bf16_tensor(ggml_tensor *target,std::span<const float> values){
    if(target->type!=GGML_TYPE_BF16 || uint64_t(ggml_nelements(target))!=values.size())
        throw std::invalid_argument("BF16 tensor upload mismatch");
    auto data=bf16_storage(values);
    ggml_backend_tensor_set(target,data.data(),0,data.size()*sizeof(ggml_bf16_t));
}
}
