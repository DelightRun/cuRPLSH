#pragma once

#include <chrono>
#include <iostream>

#include "internal/cuda_utils.h"

namespace curplsh {

namespace {
inline void printTime(const char* name, float time) {
  printf("[INFO] %s: %fms elapsed.\n", name, time);
}
}

class HostTimer {
 public:
  HostTimer() : HostTimer("") {}
  HostTimer(const char* name) : name_(name), start_(clock::now()) {}

  virtual ~HostTimer() { printTime(name_, elapsed()); }

  virtual inline float elapsed() const {
    float time =
        std::chrono::duration_cast<Millisecond>(clock::now() - start_).count();

    return time;
  }

 protected:
  typedef std::chrono::duration<float, std::ratio<1, 1000>> Millisecond;
  typedef std::chrono::high_resolution_clock::time_point time_point;
  typedef std::chrono::high_resolution_clock clock;

  const char* name_;
  time_point start_;
};

class DeviceTimer : public HostTimer {
 public:
   DeviceTimer() : HostTimer() {}
   DeviceTimer(const char* name) : HostTimer(name) {}

  inline float elapsed() const override {
    cudaDeviceSynchronize();
    return HostTimer::elapsed();
  }
};

class CUDATimer {
 public:
  CUDATimer() : CUDATimer("") {}
  CUDATimer(const char* name)
      : name_(name), startEvent_(0), stopEvent_(0), stream_(0) {
    checkCudaErrors(cudaEventCreate(&startEvent_));
    checkCudaErrors(cudaEventCreate(&stopEvent_));

    checkCudaErrors(cudaEventRecord(startEvent_, stream_));
  }

  ~CUDATimer() {
    elapsed<true>();

    checkCudaErrors(cudaEventDestroy(startEvent_));
    checkCudaErrors(cudaEventDestroy(stopEvent_));
  }

  inline void start() { checkCudaErrors(cudaEventRecord(startEvent_, stream_)); }

  template <bool Verbose = false>
  inline float elapsed() const {
    checkCudaErrors(cudaEventRecord(stopEvent_, stream_));
    checkCudaErrors(cudaEventSynchronize(stopEvent_));

    float time = 0.f;
    checkCudaErrors(cudaEventElapsedTime(&time, startEvent_, stopEvent_));

    if (Verbose) {
      printTime(name_, time);
    }

    return time;
  }

 private:
  const char* name_;

  cudaEvent_t startEvent_;
  cudaEvent_t stopEvent_;
  cudaStream_t stream_;  // TODO Currently we only use default stream
};
}
