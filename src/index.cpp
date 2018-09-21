#include "index.h"

#include "internal/assertions.h"
#include "internal/cuda_utils.h"

namespace curplsh {

Index::Index(DeviceResources* resources, int dimension, IndexConfig config)
    : dimension_(dimension),
      numData_(0),
      resources_(resources),
      memorySpace_(config.memorySpace),
      device_(config.device) {
  // FIXME: exception?
  host_assert(device_ < getNumDevices());
  host_assert(dimension_ > 0);

#ifdef UNIFIED_MEMORY
  host_assert(memorySpace_ == MemorySpace::Device ||
              (memorySpace_ == MemorySpace::Unified &&
               getFullUnifiedMemorySupport(device_)));
#else
  host_assert(memorySpace_ != MemorySpace::Unified);
#endif

  host_assert(resources_);
  resources_->initializeForDevice(device_);
}

}  // namespace curplsh
