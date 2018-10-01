#pragma once

#include "internal/cuda_utils.h"

#if CUDA_VERSION >= 8000
#define UNIFIED_MEMORY 1
#endif

namespace curplsh {

enum class MemorySpace {
  /// Using host memory, i.e. malloc/free
  Host,
  /// Using manually managed memory, i.e. cudaMalloc/cudaFree
  Device,
  /// Using unified memory, i.e. cudaMallocManaged/cudaFree
  Unified,
};

/// Return memory space fo the given ptr, assume Device for host ptr
template <typename T>
inline MemorySpace getMemorySpace(T* ptr) {
  return getIsManagedForAddress(ptr) == 1 ? MemorySpace::Unified
                                          : MemorySpace::Device;
}

/// Allocates CUDA memory
template <typename T>
inline void allocMemory(T*& ptr, size_t num, MemorySpace space) {
  if (space == MemorySpace::Device) {
    checkCudaErrors(cudaMalloc((void**)&ptr, num * sizeof(T)));
  } else if (space == MemorySpace::Unified) {
#ifdef UNIFIED_MEMORY
    checkCudaErrors(cudaMallocManaged((void**)&ptr, num * sizeof(T)));
#else
    host_assert(false);
#endif
  } else if (space == MemorySpace::Host) {
    ptr = (T*)malloc(num * sizeof(T));
    host_assert(ptr != nullptr);
  } else {
    host_assert(false);
  }
}

template <typename T>
inline void copyMemory(T* dst, T* src, size_t num,
                       cudaMemcpyKind kind = cudaMemcpyDefault,
                       cudaStream_t stream = 0) {
  checkCudaErrors(cudaMemcpyAsync(dst, src, num * sizeof(T), kind, stream));
}

}  // namespace curplsh
