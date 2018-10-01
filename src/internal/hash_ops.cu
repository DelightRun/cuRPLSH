#include "internal/hash_ops.h"

#include "internal/kernel_utils.cuh"

namespace curplsh {

namespace {

/// NOTE: Futher optimization such as vectorization is not necessary, since the
/// binarization is not very time-consuming.
template <typename T, typename IndexT, int BatchSize = 8>
__global__ void kernelBinarize(const Tensor<T, 2, IndexT> projections,
                               Tensor<unsigned, 2, IndexT> codes) {
  IndexT threadId = threadIdx.x;
  IndexT warpId = threadId / kWarpSize;
  IndexT laneId = getLaneId();

  IndexT idxOffset = blockIdx.x * BatchSize;

  bool isLastBatch = ((projections.getSize(0) - idxOffset) < BatchSize);

  T proj[BatchSize];
  unsigned code[BatchSize];

  if (isLastBatch) {
    IndexT lastBatchSize = projections.getSize(0) - idxOffset;

    for (int i = 0; i < lastBatchSize; ++i) {
      proj[i] = projections[idxOffset + i][threadId];
    }

    for (int i = 0; i < lastBatchSize; ++i) {
      bool vote = proj[i] > 0;
      code[i] = ballot(vote);
    }

    if (laneId == 0) {
      for (int i = 0; i < lastBatchSize; ++i) {
        codes[idxOffset + i][warpId] = code[i];
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      proj[i] = projections[idxOffset + i][threadId];
    }

#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      bool vote = proj[i] > 0;
      code[i] = ballot(vote);
    }

    if (laneId == 0) {
#pragma unroll
      for (int i = 0; i < BatchSize; ++i) {
        codes[idxOffset + i][warpId] = code[i];
      }
    }
  }
}

}  // namespace

template <typename T, typename CodeT, typename IndexT = int, int BatchSize = 8>
void binarize(const Tensor<T, 2, IndexT>& projections,
              Tensor<CodeT, 2, IndexT>& codes, cudaStream_t stream) {
  host_assert(projections.getSize(0) == codes.getSize(0));
  host_assert(projections.getSize(1) == codes.getSize(1) * sizeof(CodeT) * 8);
  host_assert(codes.template isCastable<unsigned>());

  auto grid = dim3(divUp(projections.getSize(0), BatchSize));
  auto block = dim3(projections.getSize(1));
  kernelBinarize<<<grid, block, 0, stream>>>(projections,
                                             codes.template cast<unsigned>());
}

void binarize(const Tensor<float, 2>& projections, Tensor<unsigned, 2>& codes,
              cudaStream_t stream) {
  binarize<float, unsigned>(projections, codes, stream);
}

}  // namespace curplsh
