#include <algorithm>
#include <initializer_list>
#include <iostream>

#include "dataset.h"
#include "device_resources.h"
#include "device_tensor.h"
#include "timer.h"

#include "internal/build_lsh.h"
#include "internal/constants.h"
#include "internal/matrix_ops.h"
#include "internal/norm.h"
#include "internal/search_lsh.h"

int main(int argc, const char **argv) {
  int device = getCudaDevice(argc, argv);
  curplsh::DeviceResources resources(device);
  curplsh::DeviceScope scope(device);

  curplsh::DatasetSIFT sift("/home/changxu/Datasets/sift/",
                            curplsh::MemorySpace::Device);

  const int dim = sift.getDimension();
  const int k = sift.getGroundTruthK();
  const int numBases = sift.getNumBase();
  const int numQueries = sift.getNumQuery();

  curplsh::DeviceTensor<float, 2> bases(const_cast<float *>(sift.getBase()),
                                        {numBases, dim}, sift.getMemorySpace());
  curplsh::DeviceTensor<float, 2> queries(const_cast<float *>(sift.getQuery()),
                                          {numQueries, dim}, sift.getMemorySpace());

  curplsh::DeviceTensor<float, 1> basesNorm({numBases}, sift.getMemorySpace());
  curplsh::DeviceTensor<int, 2> indices({numQueries, k}, sift.getMemorySpace());
  curplsh::DeviceTensor<float, 2> distances({numQueries, k}, sift.getMemorySpace());

  std::cout << "Data Loaded" << std::endl;

  {
    curplsh::DeviceTimer timer("Compute Norm");
    curplsh::computeL2Norm(bases, basesNorm, true);
    cudaDeviceSynchronize();
  }

  int numTables = 32;
  int numCandidates = argc > 1 ? atoi(argv[1]) : 1024;

  curplsh::DeviceTensor<float, 2> matrix({dim, numTables * 32},
                                         sift.getMemorySpace());
  curplsh::DeviceTensor<unsigned, 2> codebook({numBases, numTables},
                                              sift.getMemorySpace());
  {
    curplsh::DeviceTimer timer("Build Codebook");
    curplsh::generateProjectionMatrix(&resources, matrix, 0.f, 1.f, 0x1234);
    curplsh::buildCodebook(&resources, bases, matrix, codebook);
  }

  {
    curplsh::DeviceTimer timer("Search");
    curplsh::searchHammingDistance(&resources, bases, &basesNorm, codebook, matrix,
                                   queries, k, indices, distances, numCandidates);
  }

  if (numQueries == sift.getNumQuery()) {
    std::cout << sift.evaluate(indices.data()) << std::endl;
  }

  return 0;
}
