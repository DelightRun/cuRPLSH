#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include "device_resources.h"
#include "device_tensor.h"
#include "memory_space.h"

#include "internal/select.h"

using namespace curplsh;

int main(int argc, const char **argv) {
  constexpr int kNum = 10000;
  constexpr int k = 100;

  int device = getCudaDevice(argc, argv);
  DeviceResources resources(device);
  DeviceScope scope(device);

  auto stream = resources.getDefaultStreamCurrentDevice();

  std::cout << "Alloc Memory" << std::endl;

  DeviceTensor<unsigned, 2> data({1, kNum}, MemorySpace::Device);
  DeviceTensor<unsigned, 2> sorted({1, k}, MemorySpace::Device);
  DeviceTensor<int, 2> indices({1, k}, MemorySpace::Device);

  std::cout << "Initialize Data" << std::endl;

  thrust::sequence(thrust::cuda::par.on(stream), data.data(), data.data() + kNum,
                   kNum - 1, -1);

  std::cout << "Begin" << std::endl;

  radixSelect(data, sorted, indices, k, false, stream);

  std::cout << "After" << std::endl;

  unsigned *hsorted = new unsigned[sorted.getNumElements()];
  int *hindices = new int[sorted.getNumElements()];

  sorted.toHost(hsorted, stream);
  indices.toHost(hindices, stream);

  if (argc > 1) {
    for (int i = 0; i < k; i++) {
      std::cout << "#" << i << "\t" << hsorted[i] << " - " << hindices[i]
                << std::endl;
    }
  }
}
