#include "memory_space.h"

namespace curplsh {

/// Allocates CUDA memory
void allocMemory(void** ptr, size_t size, MemorySpace space) {
  if (space == MemorySpace::Device) {
    checkCudaErrors(cudaMalloc(ptr, size));
  } else if (space == MemorySpace::Unified) {
#ifdef UNIFIED_MEMORY
    checkCudaErrors(cudaMallocManaged(ptr, size));
#else
    host_assert(false);   
#endif
  }
}

}   // namespace curplsh
