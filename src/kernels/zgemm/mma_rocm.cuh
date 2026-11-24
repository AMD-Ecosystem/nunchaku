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

// The same on RDNA GPUs (32-lane wavefronts) with WMMA, which also computes the 16-bit floating-point products
// (mma_m16n16k16_* below).
#if !defined(NUNCHAKU_ROCM_MMA_REFERENCE) && (defined(__GFX11__) || defined(__GFX12__))
#define NUNCHAKU_ROCM_WMMA_INT4 1
#define NUNCHAKU_ROCM_WMMA_F16 1
#endif

#ifdef NUNCHAKU_ROCM_WMMA_F16
namespace mma_rocm {

// Moves a 16x16 matrix of 16-bit values from mma.sync fragment registers into the WMMA operand layout. p0, p1,
// p2, p3 hold (index < 8, k < 8), (index >= 8, k < 8), (index < 8, k >= 8), (index >= 8, k >= 8), where index is
// the row of A or the column of B and lane 4 * (index % 8) + t holds k 2t and 2t + 1 (+ 8). On return lane l
// holds index l % 16 with k 0-15 as eight words in k order. In step s, lanes taking index < 8 read word s and
// lanes taking index >= 8 word s + 2, so that each source lane is read for one register per step.
__device__ __forceinline__ static void
wmma_operand_words(uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3, uint32_t (&out)[8]) {
    const int lane   = __lane_id() % 32;
    const bool upper = lane % 16 >= 8;
    uint32_t lo[4], hi[4];
#pragma unroll
    for (int s = 0; s < 4; s++) {
        const int src          = 4 * (lane % 8) + ((s + (upper ? 2 : 0)) & 3);
        const bool sends_lower = lane % 4 == s; // as a source this lane is read for index lane / 4 in step lane % 4
        lo[s]                  = shfl_warp(sends_lower ? p0 : p1, src);
        hi[s]                  = shfl_warp(sends_lower ? p2 : p3, src);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        out[j]     = upper ? lo[(j + 2) & 3] : lo[j];
        out[j + 4] = upper ? hi[(j + 2) & 3] : hi[j];
    }
}

template<typename T>
__device__ __forceinline__ static T select4(T v0, T v1, T v2, T v3, int i) {
    return (i & 2) ? ((i & 1) ? v3 : v2) : ((i & 1) ? v1 : v0);
}

// d0 and d1 = A * B for the m16n8k16 fragments a, {b.x, b.y} and a, {b.z, b.w}, in the accumulator fragment
// layout. One WMMA computes the transposed product: its A operand is B (row n) and its B operand is A (column m),
// so that each result lane holds one row m of A * B and the move back to the fragment layout reads every source
// lane for one register pair per step.
template<bool is_bf16>
__device__ __forceinline__ static void wmma_m16n16k16_f32(uint4 a, uint4 b, float (&d0)[4], float (&d1)[4]) {
    uint32_t wa[8], wb[8];
    wmma_operand_words(b.x, b.z, b.y, b.w, wa);
    wmma_operand_words(a.x, a.y, a.z, a.w, wb);

    const int lane                                = __lane_id() % 32;
    float __attribute__((ext_vector_type(8))) acc = {};
#if defined(__GFX11__)
    // lanes 16-31 repeat lanes 0-15; result register v of lane l: row n = 2v + l / 16, column m = l % 16
    using operand_t    = std::conditional_t<is_bf16, short, _Float16> __attribute__((ext_vector_type(16)));
    const operand_t va = __builtin_bit_cast(operand_t, wa);
    const operand_t vb = __builtin_bit_cast(operand_t, wb);
    if constexpr (is_bf16) {
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, acc);
    } else {
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, vb, acc);
    }
#else
    // lanes 0-15 hold k 0-7, lanes 16-31 k 8-15; result register v of lane l: row n = v + 8 * (l / 16), column m
    // = l % 16
    using operand_t = std::conditional_t<is_bf16, short, _Float16> __attribute__((ext_vector_type(8)));
    uint32_t ha[4], hb[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        ha[j] = lane < 16 ? wa[j] : wa[j + 4];
        hb[j] = lane < 16 ? wb[j] : wb[j + 4];
    }
    const operand_t va = __builtin_bit_cast(operand_t, ha);
    const operand_t vb = __builtin_bit_cast(operand_t, hb);
    if constexpr (is_bf16) {
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va, vb, acc);
    } else {
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, vb, acc);
    }
#endif

    // Lane 4g + t needs rows m = g, g + 8 and columns n = 2t, 2t + 1, 2t + 8, 2t + 9, held by the four lanes
    // 16h + g + 8u. In step q it reads source i = (q + t) % 4 (u = i % 2, h = i / 2), which is read in that step
    // only by the lanes with t = (i - q) % 4 and sends them the two registers for that t.
    const int t   = lane % 4;
    const int g   = lane / 4;
    const int i_s = (lane % 16 >= 8) + 2 * (lane / 16); // this lane's index as a source
    float v0[4], v1[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const int i   = (q + t) & 3;
        const int src = 16 * (i >> 1) + g + 8 * (i & 1);
        const int r   = (i_s - q) & 3;
#if defined(__GFX11__)
        // registers r and r + 4: columns n = 2r + h and 2r + 8 + h
        const float s0 = select4(acc[0], acc[1], acc[2], acc[3], r);
        const float s1 = select4(acc[4], acc[5], acc[6], acc[7], r);
#else
        // registers 2r and 2r + 1: columns n = 2r + 8h and 2r + 1 + 8h
        const float s0 = select4(acc[0], acc[2], acc[4], acc[6], r);
        const float s1 = select4(acc[1], acc[3], acc[5], acc[7], r);
#endif
        v0[q] = shfl_warp(s0, src);
        v1[q] = shfl_warp(s1, src);
    }

    // the value from source i arrived in step (i - t) % 4
    auto from = [&](const float (&v)[4], int i) { return select4(v[0], v[1], v[2], v[3], (i - t) & 3); };
    // fragment register order: (m = g, n = 2t), (g, 2t + 1), (g + 8, 2t), (g + 8, 2t + 1)
#if defined(__GFX11__)
    // source i = u + 2h: column 2t + h of row g + 8u, in tile 0 (v0) and tile 1 (v1)
    d0[0] = from(v0, 0);
    d0[1] = from(v0, 2);
    d0[2] = from(v0, 1);
    d0[3] = from(v0, 3);
    d1[0] = from(v1, 0);
    d1[1] = from(v1, 2);
    d1[2] = from(v1, 1);
    d1[3] = from(v1, 3);
#else
    // source i = u + 2h: columns 2t (v0) and 2t + 1 (v1) of row g + 8u, in tile h
    d0[0] = from(v0, 0);
    d0[1] = from(v1, 0);
    d0[2] = from(v0, 1);
    d0[3] = from(v1, 1);
    d1[0] = from(v0, 2);
    d1[1] = from(v1, 2);
    d1[2] = from(v0, 3);
    d1[3] = from(v1, 3);
#endif
}

} // namespace mma_rocm
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

// m16n16k16 as two m16n8k16 with the same A: b holds the B fragments of columns 0-7 and 8-15, {b0.x, b0.y, b1.x,
// b1.y}, and c0, c1 the matching accumulator fragments. With WMMA the products of each k step are summed in the
// matrix unit's order and then added to c, so the result is not bit-identical to the reference, which adds one
// product at a time to c.
template<bool is_bf16>
__device__ __forceinline__ static void mma_m16n16k16_f32f16f16f32(uint4 a, uint4 b, uint4 &c0, uint4 &c1) {
#ifdef NUNCHAKU_ROCM_WMMA_F16
    float d0[4], d1[4];
    mma_rocm::wmma_m16n16k16_f32<is_bf16>(a, b, d0, d1);
    c0 = uint4(__float_as_uint(__uint_as_float(c0.x) + d0[0]),
               __float_as_uint(__uint_as_float(c0.y) + d0[1]),
               __float_as_uint(__uint_as_float(c0.z) + d0[2]),
               __float_as_uint(__uint_as_float(c0.w) + d0[3]));
    c1 = uint4(__float_as_uint(__uint_as_float(c1.x) + d1[0]),
               __float_as_uint(__uint_as_float(c1.y) + d1[1]),
               __float_as_uint(__uint_as_float(c1.z) + d1[2]),
               __float_as_uint(__uint_as_float(c1.w) + d1[3]));
#else
    c0 = mma_m16n8k16_f32f16f16f32<is_bf16>(a, uint2(b.x, b.y), c0);
    c1 = mma_m16n8k16_f32f16f16f32<is_bf16>(a, uint2(b.z, b.w), c1);
#endif
}

// The same with f16 accumulators: c = {c0.x, c0.y, c1.x, c1.y}.
__device__ __forceinline__ static uint4 mma_m16n16k16_f16f16f16f16(uint4 a, uint4 b, uint4 c) {
#ifdef NUNCHAKU_ROCM_WMMA_F16
    float d0[4], d1[4];
    mma_rocm::wmma_m16n16k16_f32<false>(a, b, d0, d1);
    auto add = [](uint32_t c, float x, float y) {
        const float2 f = __half22float2(bit_copy<half2>(c));
        return bit_copy<uint32_t>(__float22half2_rn(make_float2(f.x + x, f.y + y)));
    };
    return uint4(add(c.x, d0[0], d0[1]), add(c.y, d0[2], d0[3]), add(c.z, d1[0], d1[1]), add(c.w, d1[2], d1[3]));
#else
    const uint2 out1 = mma_m16n8k16_f16f16f16f16(a, uint2(b.x, b.y), uint2(c.x, c.y));
    const uint2 out2 = mma_m16n8k16_f16f16f16f16(a, uint2(b.z, b.w), uint2(c.z, c.w));
    return uint4(out1.x, out1.y, out2.x, out2.y);
#endif
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
