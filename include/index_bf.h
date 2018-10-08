#pragma once

#include "device_tensor.h"
#include "index.h"

namespace curplsh {

struct IndexBFConfig : public IndexConfig {
  IndexBFConfig(bool storeTrans = false) : storeTransposed(storeTrans) {}

  /// Whether or not data is stored (transparently) in a transposed layout.
  /// This will speedup GEMM (~10% faster), but substantially will slow down
  /// add() and increase storage requirements
  bool storeTransposed;
};

struct SearchBFParams {
  SearchBFParams(bool computeExactDist = false)
      : computeExactDistances(computeExactDist) {}

  /// Whether to compute exact distances, we only compute relative distance
  /// (-2qb + ||b||) by default
  bool computeExactDistances;
};

class IndexBF : public Index {
 public:
  IndexBF(DeviceResources* resources, int dim,
          IndexBFConfig config = IndexBFConfig());

  ~IndexBF() override = default;

  void train(int num, const float* data) override;

  void add(int num, const float* data) override;

  void search(int num, const float* queries, int k, int* indices, float* distances,
              SearchBFParams params = SearchBFParams{});

  void reset() override;

 protected:
  const IndexBFConfig config_;

  /// Data
  DeviceTensor<float, 2> data_;
  DeviceTensor<float, 2> dataTransposed_;

  /// Precomputed L2 norms
  DeviceTensor<float, 1> norms_;
};
}  // namespace curplsh
