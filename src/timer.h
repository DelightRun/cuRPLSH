#pragma once

#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

#include "cuda_utils.h"

namespace curplsh {

namespace {
inline void printTime(float time) {
  printf("[INFO] %fms elapsed.\n", time);
}
}

class HostTimer {
 public:
  HostTimer() : start_(clock::now()) {}

  ~HostTimer() { elapsed<true>(); }

  template <bool Verbose = false>
  inline float elapsed() const {
    float time =
        std::chrono::duration_cast<Millisecond>(clock::now() - start_).count();

    if (Verbose) {
      printTime(time);
    }

    return time;
  }

 private:
  typedef std::chrono::duration<float, std::ratio<1, 1000>> Millisecond;
  typedef std::chrono::high_resolution_clock::time_point time_point;
  typedef std::chrono::high_resolution_clock clock;

  time_point start_;
};

class DeviceTimer {
 public:
  DeviceTimer() : startEvent_(0), stopEvent_(0), stream_(0) {
    checkCudaErrors(cudaEventCreate(&startEvent_));
    checkCudaErrors(cudaEventCreate(&stopEvent_));

    checkCudaErrors(cudaEventRecord(startEvent_, stream_));
  }

  ~DeviceTimer() {
    elapsed<true>();

    checkCudaErrors(cudaEventDestroy(startEvent_));
    checkCudaErrors(cudaEventDestroy(stopEvent_));
  }

  inline void start() {
    checkCudaErrors(cudaEventRecord(startEvent_, stream_));
  }

  template <bool Verbose = false>
  inline float elapsed() const {
    checkCudaErrors(cudaEventRecord(stopEvent_, stream_));
    checkCudaErrors(cudaEventSynchronize(stopEvent_));

    float time = 0.f;
    checkCudaErrors(cudaEventElapsedTime(&time, startEvent_, stopEvent_));

    if (Verbose) {
      printTime(time);
    }

    return time;
  }

 private:
  cudaEvent_t startEvent_;
  cudaEvent_t stopEvent_;
  cudaStream_t stream_;  // TODO Currently we only use default stream
};
}
