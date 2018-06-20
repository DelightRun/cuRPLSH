#pragma once

#include <cuda_runtime.h>
#include <string>

#include "assertions.h"
#include "cuda_utils.h"

namespace curplsh {

class DeviceMemoryReservation;

class DeviceMemory {
 public:
  virtual ~DeviceMemory();

  /// Returns the device we are managing memory for
  virtual int getDevice() const = 0;

  /// Obtains a temporary memory allocation for our device,
  /// whose usage is ordered with respect to the given stream.
  virtual DeviceMemoryReservation getMemory(cudaStream_t stream, size_t size) = 0;

  /// Returns the current size available without calling cudaMalloc
  virtual size_t getSizeAvailable() const = 0;

  /// Returns a string containing our current memory manager state
  virtual std::string toString() const = 0;

  /// Returns the high-water mark of cudaMalloc allocations for our
  /// device
  virtual size_t getHighWaterCudaMalloc() const = 0;

 protected:
  friend class DeviceMemoryReservation;
  virtual void returnAllocation(DeviceMemoryReservation& m) = 0;
};

class DeviceMemoryReservation {
 public:
  ~DeviceMemoryReservation() {
    if (data_) {
      host_assert(state_);
      state_->returnAllocation(*this);
    }

    data_ = nullptr;
  }

  DeviceMemoryReservation()
      : state_(nullptr), device_(0), data_(nullptr), size_(0), stream_(0) {}

  DeviceMemoryReservation(DeviceMemory* state, int device, void* data, size_t size,
                          cudaStream_t stream)
      : state_(state), device_(device), data_(data), size_(size), stream_(stream) {}

  DeviceMemoryReservation(DeviceMemoryReservation&& other) noexcept {
    this->operator=(std::move(other));
  }

  DeviceMemoryReservation& operator=(DeviceMemoryReservation&& other) {
    if (data_) {
      host_assert(state_);
      state_->returnAllocation(*this);
    }

    state_ = other.state_;
    device_ = other.device_;
    data_ = other.data_;
    size_ = other.size_;
    stream_ = other.stream_;

    other.data_ = nullptr;

    return *this;
  }

  int device() { return device_; }
  void* get() { return data_; }
  size_t size() { return size_; }
  cudaStream_t stream() { return stream_; }

 private:
  DeviceMemory* state_;

  int device_;
  void* data_;
  size_t size_;
  cudaStream_t stream_;
};

}
