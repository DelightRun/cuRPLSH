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

template <typename CodeT, typename DistT, typename IndexT, int TileSize = 32>
__global__ void kernelHammingDistance(Tensor<CodeT, 2, IndexT> queriesCode,
                                      Tensor<CodeT, 2, IndexT> basesCode,
                                      Tensor<DistT, 2, IndexT> distances) {
  // TODO: arbitrary code length, use template specilization
  __shared__ CodeT tileA[TileSize][TileSize];
  __shared__ CodeT tileB[TileSize][TileSize];

  IndexT row = blockIdx.y * blockDim.y + threadIdx.y;
  IndexT col = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: multiple tiles
  if (row < distances.getSize(0)) {
    tileA[threadIdx.y][threadIdx.x] = queriesCode[row][threadIdx.x];
  }
  if (col < distances.getSize(1)) {
    tileB[threadIdx.y][threadIdx.x] = basesCode[col][threadIdx.y];
  }

  __syncthreads();

  if ((row >= distances.getSize(0)) || (col >= distances.getSize(1))) return;

  DistT dist = 0;
  for (int i = 0; i < blockDim.x; ++i) {
    DistT val = popcnt(tileA[threadIdx.y][i] ^ tileB[i][threadIdx.x]);
    dist += val;
  }
  distances[row][col] = dist;
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

template <typename CodeT, typename DistT, typename IndexT, int BatchSize = 8>
void computeHammingDistance(const Tensor<CodeT, 2, IndexT>& queriesCode,
                            const Tensor<CodeT, 2, IndexT>& basesCode,
                            Tensor<DistT, 2, IndexT>& distances,
                            cudaStream_t stream) {
  host_assert(distances.getSize(0) == queriesCode.getSize(0));
  host_assert(distances.getSize(1) == basesCode.getSize(0));
  host_assert(queriesCode.getSize(1) == basesCode.getSize(1));
  // host_assert(basesCode.template isCastable<unsigned>());

  // TODO: 128 - 4096 bits
  // auto basesCodeCasted = basesCode.template cast<unsigned>();
  // auto block = dim3(basesCodeCasted.getSize(1),
  //                  kMaxThreadsPerBlock / basesCodeCasted.getSize(1));
  auto block =
      dim3(basesCode.getSize(1), kMaxThreadsPerBlock / basesCode.getSize(1));
  auto grid = dim3(divUp(basesCode.getSize(0), block.y),
                   divUp(queriesCode.getSize(0), block.y));

  kernelHammingDistance<<<grid, block, 0, stream>>>(queriesCode, basesCode,
                                                    distances);
}

void binarize(const Tensor<float, 2>& projections, Tensor<unsigned, 2>& codes,
              cudaStream_t stream) {
  binarize<float, unsigned>(projections, codes, stream);
}

void computeHammingDistance(const Tensor<unsigned, 2>& queriesCode,
                            const Tensor<unsigned, 2> basesCode,
                            Tensor<unsigned, 2>& distances, cudaStream_t stream) {
  computeHammingDistance<unsigned, unsigned, int>(queriesCode, basesCode, distances,
                                                  stream);
}

}  // namespace curplsh
