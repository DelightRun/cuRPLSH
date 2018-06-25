#pragma once

#include "device_tensor.h"

namespace curplsh {

template <typename T, int Dim>
inline DeviceTensor<T, Dim> toDevice(T* src, std::initializer_list<int> sizes,
                                     int device, cudaStream_t stream = 0) {
  static_assert(Dim > 0, "Dim must be positive");
  host_assert(device >= 0);

  return DeviceTensor<T, Dim>(src, sizes, true, stream);
}

template <typename T>
inline void fromDevice(T* dst, T* src, size_t num, cudaStream_t stream = 0) {
  if (src == dst) {
    return;
  }

  auto direction = (getDeviceForAddress(dst) == -1) ? cudaMemcpyDeviceToHost
                                                    : cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpyAsync(dst, src, num * sizeof(T), direction, stream));
}

template <typename T, int Dim>
void fromDevice(T* dst, DeviceTensor<T, Dim>& src, cudaStream_t stream = 0) {
  fromDevice(dst, src.data(), src.getNumElements(), stream);
}

}  // namespace curplsh
