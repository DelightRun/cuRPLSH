#pragma once

#include "constants.h"
#include "math_utils.h"

namespace curplsh {

//////////////////////////////////////////////////////////////////////////////
//                              Pointer Traits                              //
//////////////////////////////////////////////////////////////////////////////
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

//////////////////////////////////////////////////////////////////////////////
//                             Math Type Traits                             //
//////////////////////////////////////////////////////////////////////////////
// FIXME: Make vec type constexpr
template <typename T>
struct NumericTraits {
};

// floatV
template <>
struct NumericTraits<float> {
  __host__ __device__ static constexpr inline float zero() { return 0.f; }
  __host__ __device__ static constexpr inline float identity() { return 1.f; }
  __host__ __device__ static constexpr inline float min() { return kFloatMin; }
  __host__ __device__ static constexpr inline float max() { return kFloatMax; }
};

template <>
struct NumericTraits<float2> {
  __host__ __device__ static inline float2 zero() { return make_float2(0.f); }
  __host__ __device__ static inline float2 identity() { return make_float2(1.f); }
  __host__ __device__ static inline float2 min() { return make_float2(kFloatMin); }
  __host__ __device__ static inline float2 max() { return make_float2(kFloatMax); }
};

template <>
struct NumericTraits<float3> {
  __host__ __device__ static inline float3 zero() { return make_float3(0.f); }
  __host__ __device__ static inline float3 identity() { return make_float3(1.f); }
  __host__ __device__ static inline float3 min() { return make_float3(kFloatMin); }
  __host__ __device__ static inline float3 max() { return make_float3(kFloatMax); }
};

template <>
struct NumericTraits<float4> {
  __host__ __device__ static inline float4 zero() { return make_float4(0.f); }
  __host__ __device__ static inline float4 identity() { return make_float4(1.f); }
  __host__ __device__ static inline float4 min() { return make_float4(kFloatMin); }
  __host__ __device__ static inline float4 max() { return make_float4(kFloatMax); }
};

// intV
template <>
struct NumericTraits<int> {
  __host__ __device__ static constexpr inline int zero() { return 0; }
  __host__ __device__ static constexpr inline int identity() { return 1; }
  __host__ __device__ static constexpr inline int min() { return kIntMin; }
  __host__ __device__ static constexpr inline int max() { return kIntMax; }
};

template <>
struct NumericTraits<int2> {
  __host__ __device__ static inline int2 zero() { return make_int2(0); }
  __host__ __device__ static inline int2 identity() { return make_int2(1); }
  __host__ __device__ static inline int2 min() { return make_int2(kIntMin); }
  __host__ __device__ static inline int2 max() { return make_int2(kIntMax); }
};

template <>
struct NumericTraits<int3> {
  __host__ __device__ static inline int3 zero() { return make_int3(0); }
  __host__ __device__ static inline int3 identity() { return make_int3(1); }
  __host__ __device__ static inline int3 min() { return make_int3(kIntMin); }
  __host__ __device__ static inline int3 max() { return make_int3(kIntMax); }
};

template <>
struct NumericTraits<int4> {
  __host__ __device__ static inline int4 zero() { return make_int4(0); }
  __host__ __device__ static inline int4 identity() { return make_int4(1); }
  __host__ __device__ static inline int4 min() { return make_int4(kIntMin); }
  __host__ __device__ static inline int4 max() { return make_int4(kIntMax); }
};

// uintV
template <>
struct NumericTraits<uint> {
  __host__ __device__ static constexpr inline uint zero() { return 0; }
  __host__ __device__ static constexpr inline uint identity() { return 1; }
  __host__ __device__ static constexpr inline uint min() { return kUintMin; }
  __host__ __device__ static constexpr inline uint max() { return kUintMax; }
};

template <>
struct NumericTraits<uint2> {
  __host__ __device__ static inline uint2 zero() { return make_uint2(0); }
  __host__ __device__ static inline uint2 identity() { return make_uint2(1); }
  __host__ __device__ static inline uint2 min() { return make_uint2(kUintMin); }
  __host__ __device__ static inline uint2 max() { return make_uint2(kUintMax); }
};

template <>
struct NumericTraits<uint3> {
  __host__ __device__ static inline uint3 zero() { return make_uint3(0); }
  __host__ __device__ static inline uint3 identity() { return make_uint3(1); }
  __host__ __device__ static inline uint3 min() { return make_uint3(kUintMin); }
  __host__ __device__ static inline uint3 max() { return make_uint3(kUintMax); }
};

template <>
struct NumericTraits<uint4> {
  __host__ __device__ static inline uint4 zero() { return make_uint4(0); }
  __host__ __device__ static inline uint4 identity() { return make_uint4(1); }
  __host__ __device__ static inline uint4 min() { return make_uint4(kUintMin); }
  __host__ __device__ static inline uint4 max() { return make_uint4(kUintMax); }
};

}  // namespace curplsh
