#pragma once
#include <bit>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace sam3d {
// Check every value with baseline SSE2 on x86 and the original scalar scan
// elsewhere. No native-only ISA flags, fast-math, approximate arithmetic or
// validation bypass. Loads never extend beyond the supplied span.
// Exponent 255 covers BOTH signs of infinity and every quiet/signaling NaN.
inline bool all_finite_f32(std::span<const float> values) noexcept {
    static_assert(sizeof(float)==sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
#if defined(__SSE2__) || defined(_M_X64)
    const __m128i exponent=_mm_set1_epi32(0x7f800000);
    __m128i invalid=_mm_setzero_si128();
    size_t i=0;
    for(;values.size()-i>=4;i+=4){
        const __m128i bits=_mm_castps_si128(_mm_loadu_ps(values.data()+i));
        invalid=_mm_or_si128(invalid,_mm_cmpeq_epi32(_mm_and_si128(bits,exponent),exponent));
    }
    if(_mm_movemask_epi8(invalid))return false;
    for(;i<values.size();++i)
        if((std::bit_cast<uint32_t>(values[i])&0x7f800000u)==0x7f800000u)return false;
    return true;
#else
    return std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);});
#endif
}
}
