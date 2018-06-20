#pragma once

#include <cuda.h>

#include "constants.h"

namespace curplsh {

////////////////////////////////////////////////////////////////////////////////
// Static Math Functions
////////////////////////////////////////////////////////////////////////////////
template <typename U, typename V>
__host__ __device__ constexpr auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

template <typename U, typename V>
__host__ __device__ constexpr auto roundUp(U a, V b) -> decltype(a + b) {
  return divUp(a, b) * b;
}

template <typename U, typename V>
__host__ __device__ constexpr auto roundDown(U a, V b) -> decltype(a + b) {
  return (a / b) * b;
}

template <typename T>
__host__ __device__ constexpr int log2(T v, int p = 0) {
  return (v <= 1) ? p : log2(v / 2, p + 1);
}

static_assert(log2(2) == 1, "log2");
static_assert(log2(3) == 1, "log2");
static_assert(log2(4) == 2, "log2");
static_assert(log2(8) == 3, "log2");

template <typename T>
__host__ __device__ constexpr bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));  // v > 0 and only 1 bit of v is 1
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(2049), "isPowerOf2");

template <typename T>
__host__ __device__ constexpr T nextPowerOf2(T v) {
  return isPowerOf2(v) ? (T)2 * v : ((T)1 << (log2(v) + 1));
}

static_assert(nextPowerOf2(1) == 2, "nextPowerOf2");
static_assert(nextPowerOf2(2) == 4, "nextPowerOf2");
static_assert(nextPowerOf2(3) == 4, "nextPowerOf2");
static_assert(nextPowerOf2(4) == 8, "nextPowerOf2");

static_assert(nextPowerOf2(15) == 16, "nextPowerOf2");
static_assert(nextPowerOf2(16) == 32, "nextPowerOf2");
static_assert(nextPowerOf2(17) == 32, "nextPowerOf2");

static_assert(nextPowerOf2(1536000000u) == 2147483648u, "nextPowerOf2");
static_assert(nextPowerOf2((size_t)2147483648ULL) == (size_t)4294967296ULL,
              "nextPowerOf2");

////////////////////////////////////////////////////////////////////////////////
// Bit Operations
////////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ unsigned int getBitField(unsigned int val, int pos,
                                                    int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return ret;
}

__forceinline__ __device__ unsigned long getBitField(unsigned long val, int pos,
                                                     int len) {
  unsigned long ret;
  asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
  return ret;
}

__forceinline__ __device__ unsigned int setBitField(unsigned int val,
                                                    unsigned int bits, int pos,
                                                    int len) {
  unsigned int ret;
  asm("bfi.u32 %0, %1, %2, %3, %4;"
      : "=r"(ret)
      : "r"(bits), "r"(val), "r"(pos), "r"(len));
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Lane Utils
////////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ int getLaneId() {
  int laneId;
  asm volatile("mov.s32 %0, %laneid;" : "=r"(laneId));
  return laneId;
}

__forceinline__ __device__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__forceinline__ __device__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

__forceinline__ __device__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__forceinline__ __device__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

////////////////////////////////////////////////////////////////////////////////
/// Memory Barrier
////////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ void warpFence() {
#if __CUDA_ARCH__ >= 700
  __syncwarp();
#else
// For CC < 7x, all threads in mask must execute the same __syncwarp() in
// convergence. So we cannot directly use it but just assume synchronicity.
// Ref: CUDA C Programming, v9.2, p.102-103
#endif
}

__forceinline__ __device__ void namedBarrierWait(int name, int numThreads) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

__forceinline__ __device__ void namedBarrierArrived(int name, int numThreads) {
  asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

////////////////////////////////////////////////////////////////////////////////
// Warp Primitives
////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ inline T shfl(const T val, int srcLane, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(0xffffffffu, val, srcLane, width);
#else
  return __shfl(val, srcLane, width);
#endif
}

template <typename T>
__device__ inline T shfl_up(const T val, int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(0xffffffffu, val, delta, width);
#else
  return __shfl_up(val, delta, width);
#endif
}

template <typename T>
__device__ inline T shfl_down(const T val, int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(0xffffffffu, val, delta, width);
#else
  return __shfl_down(val, delta, width);
#endif
}

template <typename T>
__device__ inline T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffffu, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

__device__ inline int all(const int predicate) {
#if CUDA_VERSION >= 9000
  return __all_sync(0xffffffff, predicate);
#else
  return __all(predicate);
#endif
}

__device__ inline int any(const int predicate) {
#if CUDA_VERSION >= 9000
  return __any_sync(0xffffffff, predicate);
#else
  return __any(predicate);
#endif
}

__device__ inline unsigned ballot(const int predicate) {
#if CUDA_VERSION >= 9000
  return __ballot_sync(0xffffffff, predicate);
#else
  return __ballot(predicate);
#endif
}

__device__ inline unsigned activemask() { return __activemask(); }

}  // namespace curplsh
