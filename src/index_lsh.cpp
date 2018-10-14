#include "index_lsh.h"

#include "internal/build_lsh.h"
#include "internal/constants.h"
#include "internal/matrix_ops.h"
#include "internal/norm.h"
#include "internal/search_lsh.h"

#include "timer.h"

namespace curplsh {

IndexLSH::IndexLSH(DeviceResources* resources, int dim, IndexLSHConfig config)
    : Index(resources, dim, config), config_(config) {}

void IndexLSH::train(int num, const float* data) {
  projMatrix_ =
      DeviceTensor<float, 2>({dimension_, config_.codeLength}, memorySpace_);
  generateProjectionMatrix(resources_, projMatrix_, 0.f, 1.f, 0x1234);  // TODO: Seed
  this->is_trained_ = true;
}

void IndexLSH::reset() {
  DeviceScope scope(device_);

  data_ = DeviceTensor<float, 2>();
  dataTransposed_ = DeviceTensor<float, 2>();
  norms_ = DeviceTensor<float, 1>();
  numData_ = 0;
}

void IndexLSH::add(int num, const float* data) {
  host_assert(numData_ == 0);  // TODO: expandable add

  if (num <= 0) return;

  DeviceScope scope(device_);
  auto stream = resources_->getDefaultStream(device_);

  numData_ += num;
  data_ = DeviceTensor<float, 2>(const_cast<float*>(data), {numData_, dimension_},
                                 stream);

  // Pre-compute Norms
  norms_ = DeviceTensor<float, 1>({numData_}, memorySpace_);
  computeL2Norm(data_, norms_, true, resources_->getDefaultStream(device_));

  // Build Codebook
  codebook_ = DeviceTensor<unsigned, 2>(
      {numData_, config_.codeLength / (sizeof(unsigned) * 8)}, memorySpace_);
  buildCodebook(resources, data_, projMatrix_, codebook_);
}

void IndexLSH::search(int num, const float* queries, int k, int* indices,
                      float* distances, SearchLSHParams params) {
  if (num <= 0) return;

  // For now, we assume all data can be resident on GPU.
  // If GPU memory is not large enough, we should consider using pinned memory
  DeviceScope scope(device_);

  auto stream = resources_->getDefaultStream(device_);

  // toDevice & fromDevice
  auto queries_ =
      DeviceTensor<float, 2>(const_cast<float*>(queries), {num, dimension_}, stream);

  // To avoid unecessary mem copy, create an empty tensor if the given results ptr is
  // resident on another device (i.e. host or other devices)
  auto indices_ = ((memorySpace_ == MemorySpace::Unified) ||
                   (getDeviceForAddress(indices) == device_))
                      ? DeviceTensor<int, 2>(indices, {num, k}, memorySpace_)
                      : DeviceTensor<int, 2>({num, k});
  auto distances_ = ((memorySpace_ == MemorySpace::Unified) ||
                     (getDeviceForAddress(distances) == device_))
                        ? DeviceTensor<float, 2>(distances, {num, k}, memorySpace_)
                        : DeviceTensor<float, 2>({num, k});

  searchHammingDistance(resources_, data_, &norms_, codebook_, projMatrix_, queries_,
                        k, indices_, distances_, params.numCandidates);

  // Copy back results if not using Unified Memory
  if (memorySpace_ != MemorySpace::Unified) {
    if (getDeviceForAddress(distances) == -1) distances_.toHost(distances, stream);
    if (getDeviceForAddress(indices) == -1) indices_.toHost(indices, stream);
  }
}

}  // namespace curplsh
