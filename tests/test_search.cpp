#include <initializer_list>
#include <iostream>

#include "dataset.h"
#include "device_resources.h"
#include "device_tensor.h"
#include "index_bf.h"
#include "timer.h"

#include "internal/search.h"

int main(int argc, const char **argv) {
  int device = getCudaDevice(argc, argv);

  curplsh::DeviceScope scope(device);
  curplsh::DeviceResources resources;

  resources.initializeForDevice(device);

  curplsh::DatasetSIFT sift("/home/changxu/Datasets/sift",
                            curplsh::MemorySpace::Device);

  const int dim = sift.getDimension();
  const int k = sift.getGroundTruthK();
  const int numBases = sift.getNumBase();
  const int numQueries = sift.getNumQuery();
  // const int numQueries = 1;

  curplsh::DeviceTensor<float, 2> bases(const_cast<float *>(sift.getBase()),
                                        {numBases, dim}, sift.getMemorySpace());
  curplsh::DeviceTensor<float, 2> queries(const_cast<float *>(sift.getQuery()),
                                          {numQueries, dim}, sift.getMemorySpace());
  curplsh::DeviceTensor<int, 2> indices({numQueries, k}, sift.getMemorySpace());
  curplsh::DeviceTensor<float, 2> distances({numQueries, k}, sift.getMemorySpace());

  curplsh::IndexBF index(&resources, dim);

  {
    curplsh::DeviceTimer timer("Add");
    index.add(numBases, bases.data());
  }

  {
    curplsh::DeviceTimer timer("Query");
    index.search(numQueries, queries.data(), k, indices.data(), distances.data());
  }

  if (numQueries == sift.getNumQuery()) {
    std::cout << "Recall: " << sift.evaluate(indices.data()) << std::endl;
  }

  return 0;
}
