#include <initializer_list>
#include <iostream>

#include "device_tensor.h"
#include "constants.h"
#include "timer.h"

#include "norm.h"

int main(int argc, const char** argv) {
  int devId = getCudaDevice(argc, argv);

  int num = atoi(argv[1]);
  int dim = atoi(argv[2]);

  curplsh::DeviceTensor<float, 2> vectors({num, dim}, curplsh::MemorySpace::Unified);
  curplsh::DeviceTensor<float, 1> norms({num}, curplsh::MemorySpace::Unified);

  for (int i = 0; i < vectors.getSize(0); ++i) {
    for (int d = 0; d < vectors.getSize(1); ++d) {
      vectors[i][d] = 1.f;
    }
  }

  cudaMemPrefetchAsync(vectors.data(), vectors.getDataMemSize(), devId, 0);

  curplsh::computeL2Norm(vectors, norms, true);

  for (int i = 0; i < 3; ++i) {
    curplsh::DeviceTimer timer;
    curplsh::computeL2Norm(vectors, norms, true);
  }

  for (int i = 0; i < norms.getSize(0); i++) {
    std::cout << "Norm #" << i << " = " << norms[i] << std::endl;
  }

  return 0;
}
