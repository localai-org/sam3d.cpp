// Archived experiment, not compiled: no useful improvement over scalar gather.
#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace sam3d {
// Same four integer interpolation terms and F32 normalization as the scalar
// image C API. Only three bytes are read per pixel, including at row edges.
inline bool image_gather_sse2(const uint8_t *a,const uint8_t *b,const uint8_t *c,const uint8_t *d,
    int wa,int wb,int wc,int wd,uint8_t *rgb,float *normalized,size_t plane) {
#if defined(__SSE2__)
    auto unpack=[](const uint8_t *p) {
        const uint32_t packed=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16);
        return _mm_unpacklo_epi8(_mm_cvtsi32_si128(int(packed)),_mm_setzero_si128());
    };
    const auto ab=_mm_unpacklo_epi16(unpack(a),unpack(b));
    const auto cd=_mm_unpacklo_epi16(unpack(c),unpack(d));
    const auto sum=_mm_add_epi32(_mm_madd_epi16(ab,_mm_set1_epi32(wa|(wb<<16))),
                                 _mm_madd_epi16(cd,_mm_set1_epi32(wc|(wd<<16))));
    const auto value=_mm_srli_epi32(_mm_add_epi32(sum,_mm_set1_epi32(512)),10);
    const uint32_t packed=uint32_t(_mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(value,_mm_setzero_si128()),_mm_setzero_si128())));
    rgb[0]=uint8_t(packed);rgb[1]=uint8_t(packed>>8);rgb[2]=uint8_t(packed>>16);
    const auto unit=_mm_div_ps(_mm_cvtepi32_ps(value),_mm_set1_ps(255.f));
    const auto centered=_mm_sub_ps(unit,_mm_set_ps(0,.406f,.456f,.485f));
    alignas(16) float out[4];
    _mm_store_ps(out,_mm_div_ps(centered,_mm_set_ps(1,.225f,.224f,.229f)));
    normalized[0]=out[0];normalized[plane]=out[1];normalized[2*plane]=out[2];
    return true;
#else
    (void)a;(void)b;(void)c;(void)d;(void)wa;(void)wb;(void)wc;(void)wd;(void)rgb;(void)normalized;(void)plane;
    return false;
#endif
}
}
