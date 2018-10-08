#pragma once

#include "device_resources.h"
#include "memory_space.h"
#include "tensor.h"

namespace curplsh {

struct IndexConfig {
  inline IndexConfig() : device(0), memorySpace(MemorySpace::Device) {}

  /// GPU device on which the index is resident
  int device;

  /// What memory space to use for primary storage.
  MemorySpace memorySpace;
};

class Index {
 public:
  Index(DeviceResources* resources, int dimension, IndexConfig config);

  virtual ~Index() = default;

  /** Perform training on representative set of vectors.
   *
   * @param num     number of training data
   * @param data    training data, size num * dim
   */
  virtual void train(int num, const float* data) = 0;

  /** Add num data of dimension dim to the index.
   *
   * Data are imlicitly assigned sequential labels.
   *
   * @param num     number of added data
   * @param data    added data, size num * dim
   */
  virtual void add(int num, const float* data) = 0;

  /** Perfor searching on the given query data
   *
   * return at most K nearest neighbor of each query.
   * If there are not enough results for a query,
   * the output will be padded with -1.
   *
   * @param num         number of queries
   * @param queries     query data, size num * dim
   * @param k           the number of returned nearest neighbors
   * @param indices     output indices of the k NNs, size num * k
   * @param distances   output pairwise distances, size num * k
  virtual void search(int num, const float* queries, int k, int* indices,
          float* distances, SearchParams params = SearchParams{}) {
  };
   */

  /// Remove all data from the index.
  virtual void reset() = 0;

  inline int getDimension() const { return dimension_; }

  inline int getNumData() const { return numData_; }

  inline DeviceResources* getResources() { return resources_; }

  inline int getDevice() const { return device_; }

 protected:
  const int dimension_;  ///< data dimension
  int numData_;          ///< data number
  bool verbose_;         ///< verbose flag

  /// true if the index doesn't require training, or the training is done
  bool isTrained_;

  /// Device resources such as streams, cuBLAS handles, etc.
  DeviceResources* resources_;

  /// The memory space of our primary storage on the GPU
  const MemorySpace memorySpace_;

  /// The GPU device we are resident on
  const int device_;
};

}  // namespace curplsh
