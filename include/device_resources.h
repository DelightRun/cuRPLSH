#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "internal/cuda_utils.h"

namespace curplsh {

/// RAII object to select the current device, and restore the previous upon
/// destruction.
class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    prevDevice_ = getCurrentDevice();

    if (prevDevice_ != device) {
      setCurrentDevice(device);
    } else {
      prevDevice_ = -1;
    }
  }
  ~DeviceScope() {
    if (prevDevice_ != -1) {
      setCurrentDevice(prevDevice_);
    }
  }

 private:
  int prevDevice_;
};

/// RAII object to manage cublasHandle_t
class CublasHandleScope {
 public:
  CublasHandleScope() { checkCudaErrors(cublasCreate(&handle_)); }
  ~CublasHandleScope() { checkCudaErrors(cublasDestroy(handle_)); }

  inline cublasHandle_t get() { return handle_; }

 private:
  cublasHandle_t handle_;
};

/// Base class of GPU-side resource provider; hides provision of
/// cuBLAS handles, CUDA streams and a temporary memory manager
class DeviceResources {
 public:
  ~DeviceResources() {
    for (auto& entry : defaultStreams_) {
      DeviceScope scope(entry.first);

      auto iter = userDefaultStreams_.find(entry.first);
      if (iter == userDefaultStreams_.end()) {
        // We create, so we destory
        checkCudaErrors(cudaStreamDestroy(entry.second));
      }
    }

    for (auto& entry : alternateStreams_) {
      DeviceScope scope(entry.first);

      for (auto stream : entry.second) {
        checkCudaErrors(cudaStreamDestroy(stream));
      }
    }

    for (auto& entry : asyncCopyStreams_) {
      DeviceScope scope(entry.first);

      checkCudaErrors(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : blasHandles_) {
      DeviceScope scope(entry.first);

      checkCudaErrors(cublasDestroy(entry.second));
    }
  }

  /// Call to pre-allocate resources for a particular device. If this is
  /// not called, then resources will be allocated at the first time
  /// of demand
  void initializeForDevice(int device) {
    // Use default streams as a marker for whether or not a certain device has been
    // initialized
    if (defaultStreams_.count(device)) {
      return;
    }

    // TODO: pinned memory allocation stuff, refer to faiss

    host_assert(device < getNumDevices());
    DeviceScope scope(device);

    // Cache device properties
    const auto& prop = getDeviceProperties(device);
    host_assert(prop.major >= 6);

    // Create streams
    cudaStream_t defaultStream = 0;
    auto iter = userDefaultStreams_.find(device);
    if (iter != userDefaultStreams_.end()) {
      defaultStream = iter->second;
    } else {
      checkCudaErrors(
          cudaStreamCreateWithFlags(&defaultStream, cudaStreamNonBlocking));
    }
    defaultStreams_[device] = defaultStream;

    cudaStream_t asyncCopyStream = 0;
    checkCudaErrors(
        cudaStreamCreateWithFlags(&asyncCopyStream, cudaStreamNonBlocking));
    asyncCopyStreams_[device] = asyncCopyStream;

    std::vector<cudaStream_t> streams;
    for (int i = 0; i < 2; ++i) {
      cudaStream_t stream = 0;
      checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      streams.push_back(stream);
    }
    alternateStreams_[device] = std::move(streams);

    // Create cuBLAS handle
    cublasHandle_t blasHandle = 0;
    checkCudaErrors(cublasCreate(&blasHandle));
    blasHandles_[device] = blasHandle;

    // TODO: memory manger stuff
  }

  /// Returns the cuBLAS handle that we use for the given device
  inline cublasHandle_t getBlasHandle(int device) {
    initializeForDevice(device);
    return blasHandles_[device];
  }

  /// Returns the stream that we order all computation on for the
  /// given device
  inline cudaStream_t getDefaultStream(int device) {
    initializeForDevice(device);
    return defaultStreams_[device];
  }

  /// Returns the set of alternative streams that we use for the given device
  inline std::vector<cudaStream_t> getAlternateStreams(int device) {
    initializeForDevice(device);
    return alternateStreams_[device];
  }

  /// Returns the temporary memory manager for the given device
  // TODO: DeviceMemory& getMemoryManager(int device) {}

  /// Returns the available CPU pinned memory buffer
  // TODO: std::pair<void*, size_t> getPinnedMemory() {}

  /// Returns the stream on which we perform async CPU <-> GPU copies
  inline cudaStream_t getAsyncCopyStream(int device) {
    return asyncCopyStreams_[device];
  }

  /// Calls getBlasHandle with the current device
  inline cublasHandle_t getBlasHandleCurrentDevice() {
    return getBlasHandle(getCurrentDevice());
  }

  /// Calls getDefaultStream with the current device
  inline cudaStream_t getDefaultStreamCurrentDevice() {
    return getDefaultStream(getCurrentDevice());
  }

  /// Synchronizes the CPU with respect to the default stream for the
  /// given device
  // equivalent to cudaDeviceSynchronize(getDefaultStream(device))
  inline void syncDefaultStream(int device) {
    checkCudaErrors(cudaStreamSynchronize(getDefaultStream(device)));
  }

  /// Calls syncDefaultStream for the current device
  inline void syncDefaultStreamCurrentDevice() {
    syncDefaultStream(getCurrentDevice());
  }

  /// Calls getAlternateStreams for the current device
  inline std::vector<cudaStream_t> getAlternateStreamsCurrentDevice() {
    return getAlternateStreams(getCurrentDevice());
  }

  /// Returns the temporary memory manager for current device
  // TODO: DeviceMemory& geMemoryManagerCurrentDevice() { retturn
  // getMemoryManager(getCurrentDevice()); }

  /// Calls getAsyncCopyStream for the current device
  inline cudaStream_t getAsyncCopyStreamCurrentDevice() {
    return getAsyncCopyStream(getCurrentDevice());
  }

 private:
  /// Default streams for each device
  std::unordered_map<int, cudaStream_t> defaultStreams_;

  /// Default streams set by user, optional
  std::unordered_map<int, cudaStream_t> userDefaultStreams_;

  /// Alternate streams
  std::unordered_map<int, std::vector<cudaStream_t>> alternateStreams_;

  /// Async copy stream to use for GPU <-> CPU pinned memory copies
  std::unordered_map<int, cudaStream_t> asyncCopyStreams_;

  /// cuBLAS handle for each device
  std::unordered_map<int, cublasHandle_t> blasHandles_;

  // TODO: memory management stuff
};

}  // namespace curplsh
