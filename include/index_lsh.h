#pragma once

#include "device_tensor.h"
#include "index.h"

namespace curplsh {

struct IndexLSHConfig : public IndexConfig {
  IndexLSHConfig(int codeLen = 1024, bool computeExactDist = false)
      : storeTransposed(storeTrans), computeExactDistances(computeExactDist) {}

  /// Code Lengths
  int codeLength;

  /// Whether to compute exact distances, we only compute relative distance
  /// (-2qb + ||b||) by default
  bool computeExactDistances;
};

struct SearchLSHParams {
  SearchLSHParams(int numCands = 1024) : numCandidates(numCands) {}

  /// Number of Candidates
  int numCandidates;
};

class IndexLSH : public Index {
 public:
  IndexLSH(DeviceResources* resources, int dim,
           IndexLSHConfig config = IndexLSHConfig());

  ~IndexLSH() override = default;

  void train(int num, const float* data) override;

  void add(int num, const float* data) override;

  void search(int num, const float* queries, int k, int* indices, float* distances,
              SearchLSHParams params);

  void reset() override;

 protected:
  const IndexLSHConfig config_;

  /// Data
  DeviceTensor<float, 2> data_;

  /// Codebook
  DeviceTensor<unsigned, 2> codebook_;

  /// Project Matrix
  DeviceTensor<float, 2> projMatrix_;

  /// Precomputed L2 norms
  DeviceTensor<float, 1> norms_;
};
}  // namespace curplsh
