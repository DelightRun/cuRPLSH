#include "internal/norm.h"

#include <type_traits>

#include "internal/constants.h"
#include "internal/cuda_utils.h"
#include "internal/kernel_utils.cuh"
#include "internal/math_utils.h"
#include "internal/reductions.cuh"

namespace curplsh {

namespace {

template <typename T, typename TVec, typename IndexT, int BatchSize, int DimLevel,
          bool Squared>
__global__ void kernelL2Norm(const Tensor<TVec, 2, IndexT> vectors,
                             Tensor<T, 1, IndexT> norms) {
  extern __shared__ char smemByte[];
  T* smem = (T*)smemByte;

  IndexT numWarps = divUp(blockDim.x, kWarpSize);
  IndexT numVecsPerIter = max(DimLevel, 1);
  IndexT numWarpsPerVec = numWarps / numVecsPerIter;

  IndexT laneId = getLaneId();
  IndexT warpId = threadIdx.x / kWarpSize;  // Warp ID in block

  IndexT dim = threadIdx.x % (numWarpsPerVec * kWarpSize);
  IndexT idxOffset = blockIdx.x * BatchSize * numVecsPerIter;

  bool isLastBatch = (blockIdx.x == (gridDim.x - 1));

  T batchNorms[BatchSize];

  if (isLastBatch) {
    int lastBatchSize = divUp(vectors.getSize(0) - idxOffset, numVecsPerIter);

    for (int idx = 0; idx < lastBatchSize; ++idx) {
      batchNorms[idx] = NumericTraits<T>::zero();
    }

    do {
      for (int idx = 0; idx < lastBatchSize; ++idx) {
        IndexT realIdx = idxOffset + idx * numVecsPerIter + warpId / numWarpsPerVec;
        if (realIdx < vectors.getSize(0)) {
          TVec tmp = vectors[realIdx][dim];
          batchNorms[idx] += dot(tmp, tmp);
        }
      }
    } while ((DimLevel < 0) && ((dim += blockDim.x) < vectors.getSize(1)));

    for (int idx = 0; idx < lastBatchSize; ++idx) {
      batchNorms[idx] = warpReduceSum<T>(batchNorms[idx]);
    }

    if (laneId == 0) {
      for (int idx = 0; idx < lastBatchSize; ++idx) {
        smem[idx * numWarps + warpId] = batchNorms[idx];
      }
    }
  } else {
    TVec tmp[BatchSize];

#pragma unroll
    for (int idx = 0; idx < BatchSize; ++idx) {
      batchNorms[idx] = NumericTraits<T>::zero();
    }

    do {
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        tmp[idx] =
            vectors[idxOffset + idx * numVecsPerIter + warpId / numWarpsPerVec][dim];
      }
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        tmp[idx] *= tmp[idx];
      }
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        batchNorms[idx] += sumc(tmp[idx]);
      }
    } while ((DimLevel < 0) && ((dim += blockDim.x) < vectors.getSize(1)));

#pragma unroll
    for (int idx = 0; idx < BatchSize; ++idx) {
      batchNorms[idx] = warpReduceSum<T>(batchNorms[idx]);
    }

    if (laneId == 0) {
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        smem[idx * numWarps + warpId] = batchNorms[idx];
      }
    }
  }

  __syncthreads();

  if ((warpId % numWarpsPerVec) == 0) {
#pragma unroll
    for (int idx = 0; idx < BatchSize; ++idx) {
      // batchNorms[idx] = laneId < numWarps ? smem[idx * numWarps + laneId]
      batchNorms[idx] = laneId < numWarpsPerVec
                            ? smem[idx * numWarps + warpId + laneId]
                            : NumericTraits<T>::zero();
    }

#pragma unroll
    for (int idx = 0; idx < BatchSize; ++idx) {
      batchNorms[idx] = warpReduceSum<T>(batchNorms[idx]);
    }

    if (laneId == 0) {  // write out results
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        IndexT realIdx = idxOffset + idx * numVecsPerIter + warpId / numWarpsPerVec;
        if (!isLastBatch || realIdx < norms.getSize(0)) {
          norms[realIdx] = Squared ? batchNorms[idx] : sqrt(batchNorms[idx]);
        }
      }
    }
  }
}

}  // namespace

template <bool Squared, typename T, typename TVec, typename IndexT = int,
          int BatchSize = 8>
inline void computeL2Norm(const Tensor<TVec, 2, IndexT>& vectors,
                          Tensor<T, 1, IndexT>& norms, cudaStream_t stream) {
  constexpr IndexT kSuggestThreads = 256;  // empirical value

  IndexT dim = vectors.getSize(1);

#define EXECUTE_L2_NORM_SPECIAL(DIM)                                      \
  do {                                                                    \
    constexpr IndexT dimLevel = kSuggestThreads / DIM;                    \
    constexpr int batchSize = BatchSize / dimLevel;                       \
    auto grid = dim3(divUp(vectors.getSize(0), batchSize * dimLevel));    \
    auto block = dim3(kSuggestThreads);                                   \
    auto smem = sizeof(T) * batchSize * dimLevel;                         \
    kernelL2Norm<T, TVec, IndexT, batchSize, dimLevel,                    \
                 Squared><<<grid, block, smem, stream>>>(vectors, norms); \
  } while (0)

#define EXECUTE_L2_NORM(NUM_THREAD, DIM_LEVEL)                            \
  do {                                                                    \
    auto grid = dim3(divUp(vectors.getSize(0), BatchSize));               \
    auto block = dim3(NUM_THREAD);                                        \
    auto smem = sizeof(T) * BatchSize * (NUM_THREAD / kWarpSize);         \
    kernelL2Norm<T, TVec, IndexT, BatchSize, DIM_LEVEL,                   \
                 Squared><<<grid, block, smem, stream>>>(vectors, norms); \
  } while (0)

  if (dim == 32) {
    EXECUTE_L2_NORM_SPECIAL(32);
  } else if (dim == 64) {
    EXECUTE_L2_NORM_SPECIAL(64);
  } else if (dim == 128) {
    EXECUTE_L2_NORM_SPECIAL(128);
  } else if (dim > kMaxThreadsPerBlock) {
    EXECUTE_L2_NORM(kMaxThreadsPerBlock, -1);
  } else {
    EXECUTE_L2_NORM(dim, 1);
  }
}

template <typename T, typename TVec>
void computeL2Norm(const Tensor<T, 2>& vectors, Tensor<T, 1>& norms, bool squared,
                   cudaStream_t stream) {
  runtime_assert(vectors.getSize(0) == norms.getSize(0));

  if (!std::is_same<T, TVec>::value &&
      vectors.template isCastable<TVec>()) {  // Vectorize
    if (squared) {
      computeL2Norm<true, T, TVec>(vectors.template cast<TVec>(), norms, stream);
    } else {
      computeL2Norm<false, T, TVec>(vectors.template cast<TVec>(), norms, stream);
    }
  } else {  // Direct
    if (squared) {
      computeL2Norm<true, T, T>(vectors, norms, stream);
    } else {
      computeL2Norm<false, T, T>(vectors, norms, stream);
    }
  }
}

void computeL2Norm(const Tensor<float, 2>& vectors, Tensor<float, 1>& norms,
                   bool squared, cudaStream_t stream) {
  auto dim = vectors.getSize(1);

  if (dim % (4 * kWarpSize) == 0) {
    computeL2Norm<float, float4>(vectors, norms, squared, stream);
  } else if (dim % (2 * kWarpSize)) {
    computeL2Norm<float, float2>(vectors, norms, squared, stream);
  } else {
    computeL2Norm<float, float>(vectors, norms, squared, stream);
  }
}

}  // namespace curplsh
