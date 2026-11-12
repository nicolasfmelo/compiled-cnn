#pragma once
#include <immintrin.h>

namespace engine {
// exp(x): nearest-integer range reduction by ln(2), degree-7 Taylor on
// [-ln(2)/2, ln(2)/2]. Exact-polynomial truncation error < 7.1e-9 relative;
// actual FP32 error (including reduction, FMA and subnormal rounding) is
// measured by test_kernels over 1,000,001 points on [-88,88]. No fast-math.
// A split scale preserves subnormals down to exp(-103.97), rather than
// manufacturing a zero exponent for n=-127. Intended finite domain <=88.
inline __m256 exp8(__m256 x) {
    const __m256 original = x;
    x = _mm256_max_ps(_mm256_set1_ps(-104.f), _mm256_min_ps(x, _mm256_set1_ps(88.f)));
    __m256 nf = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(nf, _mm256_set1_ps(0.693145751953125f), x);
    r = _mm256_fnmadd_ps(nf, _mm256_set1_ps(1.428606765330187e-6f), r);
    __m256 p = _mm256_set1_ps(1.f/5040);
    p = _mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/720));
    p = _mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/120));
    p = _mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/24));
    p = _mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/6));
    p = _mm256_fmadd_ps(p,r,_mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p,r,_mm256_set1_ps(1));
    p = _mm256_fmadd_ps(p,r,_mm256_set1_ps(1));
    __m256i n = _mm256_cvttps_epi32(nf);
    __m256 tiny = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(-126), n));
    __m256i normal = _mm256_slli_epi32(_mm256_add_epi32(n,_mm256_set1_epi32(127)),23);
    __m256i shifted = _mm256_slli_epi32(_mm256_add_epi32(n,_mm256_set1_epi32(191)),23);
    __m256 scale = _mm256_blendv_ps(_mm256_castsi256_ps(normal), _mm256_castsi256_ps(shifted), tiny);
    __m256 v = _mm256_mul_ps(p,scale);
    v = _mm256_mul_ps(v, _mm256_blendv_ps(_mm256_set1_ps(1), _mm256_set1_ps(0x1p-64f), tiny));
    return _mm256_blendv_ps(v, _mm256_setzero_ps(), _mm256_cmp_ps(original,_mm256_set1_ps(-103.972084f),_CMP_LT_OQ));
}
inline __m256 sigmoid8(__m256 x) {
    __m256 negative_abs = _mm256_or_ps(x, _mm256_set1_ps(-0.f));
    __m256 e = exp8(negative_abs);
    __m256 inv = _mm256_div_ps(_mm256_set1_ps(1), _mm256_add_ps(_mm256_set1_ps(1),e));
    return _mm256_blendv_ps(inv, _mm256_mul_ps(e,inv), _mm256_cmp_ps(x,_mm256_setzero_ps(),_CMP_LT_OQ));
}
// Common finite range: the same range reduction/polynomial, but all scales
// are normal and exp(-x) cannot overflow. Keep the stable path for extremes.
inline __m256 silu8(__m256 x) {
    __m256 magnitude=_mm256_andnot_ps(_mm256_set1_ps(-0.f),x);
    if (_mm256_movemask_ps(_mm256_cmp_ps(magnitude,_mm256_set1_ps(80.f),_CMP_LE_OQ))!=255)
        return _mm256_mul_ps(x,sigmoid8(x));
    __m256 arg=_mm256_sub_ps(_mm256_setzero_ps(),x);
    __m256 nf=_mm256_round_ps(_mm256_mul_ps(arg,_mm256_set1_ps(1.4426950408889634f)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256 r=_mm256_fnmadd_ps(nf,_mm256_set1_ps(0.693145751953125f),arg);
    r=_mm256_fnmadd_ps(nf,_mm256_set1_ps(1.428606765330187e-6f),r);
    __m256 p=_mm256_set1_ps(1.f/5040);
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/720));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/120));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/24));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/6));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(0.5f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1));
    __m256i scale=_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(nf),_mm256_set1_epi32(127)),23);
    __m256 e=_mm256_mul_ps(p,_mm256_castsi256_ps(scale));
    return _mm256_div_ps(x,_mm256_add_ps(_mm256_set1_ps(1),e));
}
}
