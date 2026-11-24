// Adated from FasterTransformer,
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
#pragma once

#include <cassert>
#include <cstdint>
#include <cfloat>
#include <type_traits>

#include <cstdio>

#include <cuda_fp16.h>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#ifdef USE_ROCM
// Give the printf above a trap a chance to reach the host before the wave is killed.
__device__ __forceinline__ static void trap_after_sleep() {
    for (int i = 0; i < 64; i++) {
        __builtin_amdgcn_s_sleep(127);
    }
    __builtin_trap();
}
#endif

__device__ __forceinline__ static void trap_unsupported_arch() {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        printf("This kernel is not supported on your GPU\n");
    }
    __syncthreads();
#ifdef USE_ROCM
    trap_after_sleep();
#else
    __nanosleep(1000000);
    __trap();
#endif
}

// Shuffles within a 32-lane warp. All kernels here index a 32-lane warp; on AMD GPUs with a 64-lane
// wavefront, a width of 32 keeps the two warps of a wavefront independent.
#ifdef USE_ROCM
// Bit copy between types of equal size; unlike __builtin_bit_cast it accepts the HIP half and bfloat16 types,
// which are not trivially copyable.
template<typename To, typename From>
__device__ __forceinline__ static To bit_copy(const From &from) {
    static_assert(sizeof(To) == sizeof(From));
    To to;
    __builtin_memcpy((void *)&to, (const void *)&from, sizeof(To));
    return to;
}

template<typename T, typename F>
__device__ __forceinline__ static T shfl_bits(T val, F &&shfl) {
    static_assert(sizeof(T) == 2 || sizeof(T) % 4 == 0);
    if constexpr (sizeof(T) == 2) {
        return bit_copy<T>((unsigned short)shfl((int)bit_copy<unsigned short>(val)));
    } else {
        struct words {
            int w[sizeof(T) / 4];
        };
        words v = bit_copy<words>(val);
#pragma unroll
        for (int i = 0; i < sizeof(T) / 4; i++) {
            v.w[i] = shfl(v.w[i]);
        }
        return bit_copy<T>(v);
    }
}
#endif

template<typename T>
__device__ __forceinline__ static T shfl_warp(T val, int srcLane) {
#ifdef USE_ROCM
    return shfl_bits(val, [srcLane](int v) { return __shfl(v, srcLane, 32); });
#else
    return __shfl_sync(~0, val, srcLane);
#endif
}

template<typename T>
__device__ __forceinline__ static T shfl_xor_warp(T val, int laneMask) {
#ifdef USE_ROCM
    return shfl_bits(val, [laneMask](int v) { return __shfl_xor(v, laneMask, 32); });
#else
    return __shfl_xor_sync(~0, val, laneMask);
#endif
}

// HIP has no __hmax2 for half2.
template<typename T>
__device__ __forceinline__ static T hmax2(T a, T b) {
#ifdef USE_ROCM
    return T(__hmax(a.x, b.x), __hmax(a.y, b.y));
#else
    return __hmax2(a, b);
#endif
}

__device__ __forceinline__ static void syncwarp() {
#ifdef USE_ROCM
    // A 32-lane warp never spans two wavefronts, so its lanes execute in lockstep; this only orders memory.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#else
    __syncwarp();
#endif
}

#ifdef USE_ROCM
// cvt.rni.sat.s8: round to nearest even, saturate to [-128, 127], NaN to 0.
__device__ __forceinline__ static int float_to_int8_sat_rn(float x) {
    return x != x ? 0 : (int)fminf(fmaxf(rintf(x), -128.0f), 127.0f);
}
#endif

#if defined(ENABLE_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ static __nv_bfloat162
__hfma2(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c) {
    trap_unsupported_arch();
    return __nv_bfloat162(0.0f, 0.0f);
}
#endif

template<typename T>
struct num_elems;
template<>
struct num_elems<float> {
    static constexpr int value = 1;
};
template<>
struct num_elems<float2> {
    static constexpr int value = 2;
};
template<>
struct num_elems<float4> {
    static constexpr int value = 4;
};
template<>
struct num_elems<half> {
    static constexpr int value = 1;
};
template<>
struct num_elems<half2> {
    static constexpr int value = 2;
};
#ifdef ENABLE_BF16
template<>
struct num_elems<__nv_bfloat16> {
    static constexpr int value = 1;
};
template<>
struct num_elems<__nv_bfloat162> {
    static constexpr int value = 2;
};
#endif
#ifdef ENABLE_FP8
template<>
struct num_elems<__nv_fp8_e4m3> {
    static constexpr int value = 1;
};
template<>
struct num_elems<__nv_fp8x2_e4m3> {
    static constexpr int value = 2;
};
#endif

template<typename T, int num>
struct packed_as;
template<typename T>
struct packed_as<T, 1> {
    using type = T;
};
template<>
struct packed_as<half, 2> {
    using type = half2;
};
template<>
struct packed_as<float, 2> {
    using type = float2;
};
template<>
struct packed_as<int8_t, 2> {
    using type = int16_t;
};
template<>
struct packed_as<int32_t, 2> {
    using type = int2;
};
template<>
struct packed_as<half2, 1> {
    using type = half;
};
template<>
struct packed_as<float2, 1> {
    using type = float;
};
#ifdef ENABLE_BF16
template<>
struct packed_as<__nv_bfloat16, 2> {
    using type = __nv_bfloat162;
};
template<>
struct packed_as<__nv_bfloat162, 1> {
    using type = __nv_bfloat16;
};
#endif
#ifdef ENABLE_FP8
template<>
struct packed_as<__nv_fp8_e4m3, 2> {
    using type = __nv_fp8x2_e4m3;
};
template<>
struct packed_as<__nv_fp8x2_e4m3, 1> {
    using type = __nv_fp8_e4m3;
};
template<>
struct packed_as<__nv_fp8_e5m2, 2> {
    using type = __nv_fp8x2_e5m2;
};
template<>
struct packed_as<__nv_fp8x2_e5m2, 1> {
    using type = __nv_fp8_e5m2;
};
#endif

inline __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}
inline __device__ float2 operator+(float2 a, float b) {
    return make_float2(a.x + b, a.y + b);
}
inline __device__ float2 operator-(float2 a, float b) {
    return make_float2(a.x - b, a.y - b);
}

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
    return (int8_t)float_to_int8_sat_rn(x);
#else
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t &>(dst);
#endif
}

template<typename T>
inline __device__ T ldg(const T *val) {
    return __ldg(val);
}

#if ENABLE_BF16
#define bf1622float2 __bfloat1622float2
#define float22bf162 __float22bfloat162_rn
#define bf162bf162 __bfloat162bfloat162
inline __device__ int16_t bf1622int16(__nv_bfloat162 val) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(USE_ROCM)
    float2 f_val;
    f_val.x = max(min(__low2float(val), 127.f), -128.f);
    f_val.y = max(min(__high2float(val), 127.f), -128.f);

    union {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = static_cast<int8_t>(static_cast<short>(f_val.x));
    int8[1] = static_cast<int8_t>(static_cast<short>(f_val.y));
    return int16;
#else
    val = __hmin2(val, make_bfloat162(127., 127.));
    val = __hmax2(val, make_bfloat162(-128., -128.));

    union {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = static_cast<int8_t>(static_cast<short>(val.x));
    int8[1] = static_cast<int8_t>(static_cast<short>(val.y));
    return int16;
#endif
}
#endif

#if ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 ldg(const __nv_bfloat162 *val) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(USE_ROCM)
    return val[0];
#else
    return __ldg(val);
#endif
}

template<>
inline __device__ __nv_bfloat16 ldg(const __nv_bfloat16 *val) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(USE_ROCM)
    return val[0];
#else
    return __ldg(val);
#endif
}
#endif // ENABLE_BF16

template<typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
#ifdef USE_ROCM
    // The HIP half and bfloat16 types have several integer constructors, so an implicit conversion is ambiguous.
    if constexpr (std::is_integral_v<T_IN> && !std::is_integral_v<T_OUT>) {
        return T_OUT((float)val);
    } else {
        return val;
    }
#else
    return val;
#endif
}

template<>
__device__ inline float2 cuda_cast<float2, int2>(int2 val) {
    return make_float2(val.x, val.y);
}

template<>
__device__ inline float2 cuda_cast<float2, float>(float val) {
    return make_float2(val, val);
}

template<>
__device__ inline float2 cuda_cast<float2, half2>(half2 val) {
    return __half22float2(val);
}

template<>
__device__ inline half2 cuda_cast<half2, float2>(float2 val) {
    return __float22half2_rn(val);
}

template<>
__device__ inline half2 cuda_cast<half2, float>(float val) {
    return __float2half2_rn(val);
}

template<>
__device__ inline half2 cuda_cast<half2, half>(half val) {
    return __half2half2(val);
}

template<>
__device__ inline int8_t cuda_cast<int8_t, half>(half val) {
    union {
        int8_t int8[2];
        int16_t int16;
    };

    union {
        half fp16;
        int16_t int16_in;
    };

    fp16 = val;
#ifdef USE_ROCM
    int16 = (int8_t)float_to_int8_sat_rn(__half2float(fp16));
#else
    asm volatile("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
#endif
    return int8[0];
}

template<>
__device__ inline int16_t cuda_cast<int16_t, half2>(half2 val) {
    union {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = cuda_cast<int8_t>(val.x);
    int8[1] = cuda_cast<int8_t>(val.y);
    return int16;
}

template<>
__device__ inline int8_t cuda_cast<int8_t, float>(float val) {
    union {
        int8_t int8[2];
        int16_t int16;
    };

#ifdef USE_ROCM
    int16 = (int8_t)float_to_int8_sat_rn(val);
#else
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
#endif
    return int8[0];
}

template<>
__device__ inline int16_t cuda_cast<int16_t, float2>(float2 val) {
    union {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = cuda_cast<int8_t>(val.x);
    int8[1] = cuda_cast<int8_t>(val.y);
    return int16;
}

template<>
__device__ inline half2 cuda_cast<half2, int16_t>(int16_t val) {
    union {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    return make_half2(int8[0], int8[1]);
}

template<>
__device__ inline float2 cuda_cast<float2, int16_t>(int16_t val) {
    union {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    return make_float2(int8[0], int8[1]);
}

#ifdef ENABLE_BF16
template<>
__device__ inline __nv_bfloat16 cuda_cast(int32_t val) {
    return static_cast<float>(val);
}

template<>
__device__ inline __nv_bfloat16 cuda_cast(int8_t val) {
    return static_cast<float>(val);
}

template<>
__device__ inline int8_t cuda_cast(__nv_bfloat16 val) {
    return static_cast<float>(val);
}

template<>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<>
__device__ inline float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val) {
    return bf1622float2(val);
}

template<>
__device__ inline half cuda_cast<half, __nv_bfloat16>(__nv_bfloat16 val) {
    return __float2half(__bfloat162float(val));
}

template<>
__device__ inline int16_t cuda_cast<int16_t, __nv_bfloat162>(__nv_bfloat162 val) {
    return bf1622int16(val);
}

template<>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val) {
    return __float2bfloat16(val);
}

template<>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, half>(half val) {
    return __float2bfloat16(__half2float(val));
}

template<>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_bfloat16>(__nv_bfloat16 val) {
    return bf162bf162(val);
}

template<>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float>(float val) {
#ifdef USE_ROCM
    return __bfloat162bfloat162(__float2bfloat16(val));
#else
    return __float2bfloat162_rn(val);
#endif
}

template<>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float2>(float2 val) {
    return float22bf162(val);
}

template<>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, int16_t>(int16_t val) {
    union {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    __nv_bfloat162 res;
    res.x = cuda_cast<__nv_bfloat16>(int8[0]);
    res.y = cuda_cast<__nv_bfloat16>(int8[1]);
    return res;
}

template<>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, half2>(half2 val) {
    return float22bf162(__half22float2(val));
}

#endif // ENABLE BF16

template<typename f16_t>
__device__ __forceinline__ packed_as<f16_t, 2>::type f162f162(f16_t x);

template<>
__device__ __forceinline__ packed_as<half, 2>::type f162f162<half>(half x) {
    return __half2half2(x);
}

#ifdef ENABLE_BF16
template<>
__device__ __forceinline__ packed_as<__nv_bfloat16, 2>::type f162f162<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
}
#endif

template<typename To, typename Ti>
__device__ inline To cuda_sum(Ti val) {
    return cuda_cast<To>(val);
};

template<typename To>
__device__ inline To cuda_sum(float2 val) {
    return cuda_cast<To>(val.x + val.y);
};

// Unary maximum: compute the max of a vector type
template<typename To, typename Ti>
__device__ inline To cuda_max(Ti val) {
    return cuda_cast<To>(val);
};

template<>
__device__ inline float cuda_max(float2 val) {
    return fmaxf(val.x, val.y);
}

template<>
__device__ inline half cuda_max(half2 val) {
    return __hmax(val.x, val.y);
}

#ifdef ENABLE_BF16
template<>
__device__ inline __nv_bfloat16 cuda_max(__nv_bfloat162 val) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)) || defined(USE_ROCM)
    return __hmax(val.x, val.y);
#else
    assert(false);
    return 0;
#endif
}
#endif

// Binary maximum: compute the max of two scalar types
template<typename T>
__device__ inline T cuda_max(T val1, T val2) {
    return (val1 > val2) ? val1 : val2;
}

template<typename T>
__device__ inline T cuda_abs(T val) {
    assert(false);
    return {};
}

template<>
__device__ inline float cuda_abs(float val) {
    return fabs(val);
}

template<>
__device__ inline float2 cuda_abs(float2 val) {
    return make_float2(fabs(val.x), fabs(val.y));
}

template<>
__device__ inline half cuda_abs(half val) {
    return __habs(val);
}

template<>
__device__ inline half2 cuda_abs(half2 val) {
    return __habs2(val);
}

#ifdef ENABLE_BF16

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template<>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val) {
    return __habs(val);
}

template<>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val) {
    return __habs2(val);
}
#endif

#endif // ENABLE_FP16
