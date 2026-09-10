#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace sam3d {
// Layout-only copy, with cache-sized tiles. No arithmetic, filtering or
// reinterpretation of the input values; callers retain finite-value checks.
inline void transpose_f32(std::span<const float> source,std::span<float> target,
                          uint64_t rows,uint64_t columns){
    if(!rows || !columns || rows>source.size()/columns || rows*columns!=source.size() || target.size()!=source.size())
        throw std::invalid_argument("transpose shape mismatch");
    constexpr uint64_t tile=32;
    for(uint64_t r=0;r<rows;r+=tile)for(uint64_t c=0;c<columns;c+=tile)
        for(uint64_t i=r;i<std::min(rows,r+tile);++i)for(uint64_t j=c;j<std::min(columns,c+tile);++j)
            target[j*rows+i]=source[i*columns+j];
}
}
