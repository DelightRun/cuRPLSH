#include <initializer_list>
#include <iostream>

#include "device_resources.h"
#include "device_tensor.h"
#include "norm.h"
#include "search.h"
#include "constants.h"
#include "timer.h"

int main(int argc, const char** argv) {
  int device = getCudaDevice(argc, argv);
  curplsh::DeviceScope scope(device);
  curplsh::DeviceResources resources;

  const int dim = 128;
  const int numBases = 1000;
  const int numQueries = 100;
  const int k = 10;

  curplsh::DeviceTensor<float, 2> bases({numBases, dim},
                                        curplsh::MemorySpace::Unified);
  curplsh::DeviceTensor<float, 1> baseNorms({numBases},
                                            curplsh::MemorySpace::Unified);
  curplsh::DeviceTensor<float, 2> queries({numQueries, dim},
                                          curplsh::MemorySpace::Unified);
  curplsh::DeviceTensor<int, 2> indices({numQueries, k},
                                        curplsh::MemorySpace::Unified);
  curplsh::DeviceTensor<float, 2> distances({numQueries, k},
                                            curplsh::MemorySpace::Unified);

  for (int d = 0; d < dim; ++d) {
    for (int i = 0; i < queries.getSize(0); ++i) queries[i][d] = 1.f;
    for (int i = 0; i < bases.getSize(0); ++i) bases[i][d] = min((float)i, 10.f);
  }

  cudaMemPrefetchAsync(bases.data(), bases.getDataMemSize(), device, 0);
  cudaMemPrefetchAsync(queries.data(), queries.getDataMemSize(), device, 0);
  cudaMemPrefetchAsync(indices.data(), indices.getDataMemSize(), device, 0);
  cudaMemPrefetchAsync(distances.data(), distances.getDataMemSize(), device, 0);

  {
    curplsh::DeviceTimer timer;
    curplsh::computeL2Norm(bases, baseNorms, true);
  }

  {
    curplsh::DeviceTimer timer;
    resources.initializeForDevice(device);
  }

  for (int i = 0; i < 3; ++i) {
    curplsh::DeviceTimer timer;
    curplsh::searchL2Distance(&resources, bases, &baseNorms, queries, k, indices,
                              distances, false);
  }

  for (int i = 0; i < indices.getSize(0); ++i) {
    for (int j = 0; j < indices.getSize(1); ++j) {
      std::cout << '(' << indices[i][j] << ',' << distances[i][j] << ')' << '\t';
    }
    std::cout << std::endl;
  }

  return 0;
}
