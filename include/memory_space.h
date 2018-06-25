#pragma once

#include "internal/cuda_utils.h"

#if CUDA_VERSION >= 8000
#define UNIFIED_MEMORY 1
#endif

namespace curplsh {

enum class MemorySpace {
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
void allocMemory(void** ptr, size_t size, MemorySpace space);

}  // namespace curplsh
