#include "index_bf.h"

#include "internal/norm.h"
#include "internal/search.h"

namespace curplsh {

IndexBF::IndexBF(DeviceResources* resources, int dim, IndexBFConfig config)
    : Index(resources, dim, config), config_(config) {
  this->isTrained_ = true;
}

void train(int num, const float* data) {
  // Do Nothing
}

void IndexBF::reset() {
  DeviceScope scope(device_);

  data_ = DeviceTensor<float, 2>();
  dataTransposed_ = DeviceTensor<float, 2>();
  norms_ = DeviceTensor<float, 1>();
  numData_ = 0;
}

void IndexBF::add(int num, const float* data) {
  host_assert(numData_ == 0);  // FIXME: expandable add

  if (num <= 0) return;

  DeviceScope scope(device_);
  auto stream = resources_->getDefaultStream(device_);

  numData_ += num;
  data_ = DeviceTensor<float, 2>(const_cast<float*>(data), {numData_, dimension_},
                                 stream);
  if (config_.storeTransposed) {
    dataTransposed_ = DeviceTensor<float, 2>({dimension_, numData_}, memorySpace_);
    // TODO: data transpose
  }

  norms_ = DeviceTensor<float, 1>({numData_}, memorySpace_);
  computeL2Norm(data_, norms_, true, resources_->getDefaultStream(device_));
}

void IndexBF::search(int num, const float* queries, int k, int* indices,
                     float* distances) {
  if (num <= 0) return;

  // For now, we assume all data can be resident on GPU.
  // If GPU memory is not large enough, we should consider using pinned memory
  DeviceScope scope(device_);
  auto stream = resources_->getDefaultStream(device_);

  // toDevice & fromDevice
  auto queries_ = helper::toDevice<float, 2>(const_cast<float*>(queries), {num, dimension_},
                                     device_, stream);

  // To avoid unecessary mem copy, create an empty tensor if the given results ptr is
  // resident on another device (i.e. host or other devices)
  auto indices_ = getDeviceForAddress(indices) == device_
                      ? DeviceTensor<int, 2>(indices, {num, k}, false)
                      : DeviceTensor<int, 2>({num, k});
  auto distances_ = getDeviceForAddress(indices) == device_
                        ? DeviceTensor<float, 2>(distances, {num, k}, false)
                        : DeviceTensor<float, 2>({num, k});

  searchL2Distance(resources_, data_, &norms_, queries_, k, indices_, distances_,
                   true);

  // Copy back results if necessary
  helper::fromDevice<int, 2>(indices, indices_, stream);
  helper::fromDevice<float, 2>(distances, distances_, stream);
}

}  // namespace curplsh
