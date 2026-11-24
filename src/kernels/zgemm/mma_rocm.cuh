#pragma once

#include <cstdint>
#include <type_traits>
#include "common.h"
#include "../utils.cuh"

// HIP implementations of the mma.sync shapes used by the kernels. Operands and results keep the per-lane
// fragment layouts of the PTX instructions they replace, so the packed weights and all code built on these
// functions are shared with the CUDA path.
//
// The functions in this file are the reference implementation: each lane gathers the rows of A and the
// columns of B that its outputs need from the other lanes of its 32-lane warp and computes them with scalar
// arithmetic. Integer results are exact. Faster paths built on matrix instructions are checked bit for bit
// against these.

namespace nunchaku::kernels {

namespace mma_helper {
struct f32 {};
struct f16 {};
struct bf16 {};
struct s32 {};
struct s4 {};
struct u4 {};
struct s8 {};

template<bool is_bf16>
using f16bf16 = std::conditional_t<is_bf16, bf16, f16>;
template<bool is_unsigned>
using s4u4 = std::conditional_t<is_unsigned, u4, s4>;
}; // namespace mma_helper

namespace mma_rocm {

// Calls f(a_lo, a_hi, b_lo, b_hi) once per 32-bit k-chunk, in increasing k order, with
//   a_lo: chunk of row (lane / 4) of A,     a_hi: chunk of row (lane / 4 + 8) of A,
//   b_lo: chunk of column 2 * (lane % 4),   b_hi: chunk of column 2 * (lane % 4) + 1 of B,
// for the m16n8 fragment layouts where A is {a.x, a.y, a.z, a.w} = {(r, k0), (r + 8, k0), (r, k1), (r + 8, k1)}
// and B is {b.x, b.y} = {(c, k0), (c, k1)}, with k0 the low and k1 the high half of K.
template<typename F>
__device__ __forceinline__ static void gather_m16n8(uint4 a, uint2 b, F &&f) {
    const int lane = __lane_id() % 32;
    const int g    = lane / 4;
    const int t    = lane % 4;
#pragma unroll
    for (int h = 0; h < 2; h++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t a_lo = shfl_warp(h ? a.z : a.x, 4 * g + j);
            const uint32_t a_hi = shfl_warp(h ? a.w : a.y, 4 * g + j);
            const uint32_t b_lo = shfl_warp(h ? b.y : b.x, 8 * t + j);
            const uint32_t b_hi = shfl_warp(h ? b.y : b.x, 8 * t + 4 + j);
            f(a_lo, a_hi, b_lo, b_hi);
        }
    }
}

template<typename T>
__device__ __forceinline__ static int unpack_int(uint32_t word, int idx) = delete;

template<>
__device__ __forceinline__ int unpack_int<mma_helper::s4>(uint32_t word, int idx) {
    return (int)(word << (28 - 4 * idx)) >> 28;
}
template<>
__device__ __forceinline__ int unpack_int<mma_helper::u4>(uint32_t word, int idx) {
    return (word >> (4 * idx)) & 0xf;
}
template<>
__device__ __forceinline__ int unpack_int<mma_helper::s8>(uint32_t word, int idx) {
    return (int)(word << (24 - 8 * idx)) >> 24;
}

template<typename AType, typename BType>
__device__ __forceinline__ static int dot_int(uint32_t a, uint32_t b) {
    constexpr int N = std::is_same_v<AType, mma_helper::s8> ? 4 : 8;
    int sum         = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
        sum += unpack_int<AType>(a, i) * unpack_int<BType>(b, i);
    }
    return sum;
}

template<bool is_bf16>
__device__ __forceinline__ static float2 unpack_f16x2(uint32_t word) {
    if constexpr (is_bf16) {
        return __bfloat1622float2(bit_copy<__nv_bfloat162>(word));
    } else {
        return __half22float2(bit_copy<half2>(word));
    }
}

template<bool is_bf16>
__device__ __forceinline__ static float dot_f16x2(uint32_t a, uint32_t b, float acc) {
    const float2 fa = unpack_f16x2<is_bf16>(a);
    const float2 fb = unpack_f16x2<is_bf16>(b);
    acc             = fmaf(fa.x, fb.x, acc);
    acc             = fmaf(fa.y, fb.y, acc);
    return acc;
}

// Eight 4-bit integers widened to eight bytes in the same order, for the 8-bit matrix instructions.
template<bool is_unsigned>
__device__ __forceinline__ static uint64_t widen_int4x8(uint32_t word) {
    const uint32_t even = word & 0x0f0f0f0f;        // elements 0, 2, 4, 6
    const uint32_t odd  = (word >> 4) & 0x0f0f0f0f; // elements 1, 3, 5, 7
    uint32_t lo         = __builtin_amdgcn_perm(odd, even, 0x05010400);
    uint32_t hi         = __builtin_amdgcn_perm(odd, even, 0x07030602);
    if constexpr (!is_unsigned) {
        // sign-extend each byte from bit 3; 0x08 * 0x1e = 0xf0 never carries into the next byte
        lo |= (lo & 0x08080808) * 0x1e;
        hi |= (hi & 0x08080808) * 0x1e;
    }
    return (uint64_t)hi << 32 | lo;
}

} // namespace mma_rocm

// Matrix-instruction path of the W4A4 GEMM main loop on CDNA GPUs (64-lane wavefronts). Selected per device
// target; building with NUNCHAKU_ROCM_MMA_REFERENCE defined keeps the reference implementation above instead.
#if !defined(NUNCHAKU_ROCM_MMA_REFERENCE) && (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
#define NUNCHAKU_ROCM_MFMA_INT4 1
#endif

// m16n8k16, f16 inputs, f16 accumulator
__device__ __forceinline__ static uint2 mma_m16n8k16_f16f16f16f16(uint4 a, uint2 b, uint2 c) {
    const float2 c_lo = __half22float2(bit_copy<half2>(c.x));
    const float2 c_hi = __half22float2(bit_copy<half2>(c.y));
    float d[4]        = {c_lo.x, c_lo.y, c_hi.x, c_hi.y};
    mma_rocm::gather_m16n8(a, b, [&](uint32_t a_lo, uint32_t a_hi, uint32_t b_lo, uint32_t b_hi) {
        d[0] = mma_rocm::dot_f16x2<false>(a_lo, b_lo, d[0]);
        d[1] = mma_rocm::dot_f16x2<false>(a_lo, b_hi, d[1]);
        d[2] = mma_rocm::dot_f16x2<false>(a_hi, b_lo, d[2]);
        d[3] = mma_rocm::dot_f16x2<false>(a_hi, b_hi, d[3]);
    });
    return uint2(bit_copy<uint32_t>(__float22half2_rn(make_float2(d[0], d[1]))),
                 bit_copy<uint32_t>(__float22half2_rn(make_float2(d[2], d[3]))));
}

// m16n8k16, f16 or bf16 inputs, f32 accumulator
template<bool is_bf16>
__device__ __forceinline__ static uint4 mma_m16n8k16_f32f16f16f32(uint4 a, uint2 b, uint4 c) {
    float d[4] = {__uint_as_float(c.x), __uint_as_float(c.y), __uint_as_float(c.z), __uint_as_float(c.w)};
    mma_rocm::gather_m16n8(a, b, [&](uint32_t a_lo, uint32_t a_hi, uint32_t b_lo, uint32_t b_hi) {
        d[0] = mma_rocm::dot_f16x2<is_bf16>(a_lo, b_lo, d[0]);
        d[1] = mma_rocm::dot_f16x2<is_bf16>(a_lo, b_hi, d[1]);
        d[2] = mma_rocm::dot_f16x2<is_bf16>(a_hi, b_lo, d[2]);
        d[3] = mma_rocm::dot_f16x2<is_bf16>(a_hi, b_hi, d[3]);
    });
    return uint4(__float_as_uint(d[0]), __float_as_uint(d[1]), __float_as_uint(d[2]), __float_as_uint(d[3]));
}

// m16n8k64 for s4/u4 x s4, m16n8k32 for s8 x s8, s32 accumulator
template<typename AType, typename BType>
__device__ __forceinline__ static uint4 mma_m16n8kx_s32common(uint4 a, uint2 b, uint4 c) {
    int d[4] = {(int)c.x, (int)c.y, (int)c.z, (int)c.w};
    mma_rocm::gather_m16n8(a, b, [&](uint32_t a_lo, uint32_t a_hi, uint32_t b_lo, uint32_t b_hi) {
        d[0] += mma_rocm::dot_int<AType, BType>(a_lo, b_lo);
        d[1] += mma_rocm::dot_int<AType, BType>(a_lo, b_hi);
        d[2] += mma_rocm::dot_int<AType, BType>(a_hi, b_lo);
        d[3] += mma_rocm::dot_int<AType, BType>(a_hi, b_hi);
    });
    return uint4((uint32_t)d[0], (uint32_t)d[1], (uint32_t)d[2], (uint32_t)d[3]);
}

}; // namespace nunchaku::kernels
