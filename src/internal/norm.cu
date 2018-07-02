#include "internal/norm.h"

#include <type_traits>

#include "internal/constants.h"
#include "internal/cuda_utils.h"
#include "internal/kernel_utils.cuh"
#include "internal/math_utils.h"
#include "internal/reductions.cuh"

namespace curplsh {

namespace {

template <typename T, typename TVec, typename IndexT, int BatchSize, bool VeryHighDim,
          bool Squared>
__global__ void kernelL2Norm(const Tensor<TVec, 2, IndexT> vectors,
                             Tensor<T, 1, IndexT> norms) {
  extern __shared__ char smemByte[];
  T* smem = (T*)smemByte;

  IndexT numWarps = divUp(blockDim.x * blockDim.y, kWarpSize);
  IndexT numWarpsPerVec = divUp(blockDim.x, kWarpSize);

  IndexT laneId = getLaneId();
  IndexT warpId = threadIdx.x / kWarpSize;

  IndexT dim = threadIdx.x;
  IndexT idxOffset = blockIdx.x * blockDim.y * BatchSize;

  bool isLastBatch = ((vectors.getSize(0) - idxOffset) < BatchSize);

  T batchNorms[BatchSize];

  if (isLastBatch) {
    int lastBatchSize = divUp(vectors.getSize(0) - idxOffset, blockDim.y);

    for (int idx = 0; idx < lastBatchSize; ++idx) {
      batchNorms[idx] = NumericTraits<T>::zero();
    }

    do {
      for (int idx = 0; idx < lastBatchSize; ++idx) {
        IndexT realIdx = idxOffset + idx * blockDim.y + threadIdx.y;
        if (realIdx < vectors.getSize(0)) {
          TVec tmp = vectors[realIdx][dim];
          batchNorms[idx] += dot(tmp, tmp);
        }
      }
    } while (VeryHighDim && ((dim += blockDim.x) < vectors.getSize(1)));

    for (int idx = 0; idx < lastBatchSize; ++idx) {
      batchNorms[idx] = warpReduceSum<T>(batchNorms[idx]);
    }

    if (laneId == 0) {
      for (int idx = 0; idx < lastBatchSize; ++idx) {
        smem[idx * numWarps + threadIdx.y * numWarpsPerVec + warpId] =
            batchNorms[idx];
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
        tmp[idx] = vectors[idxOffset + idx * blockDim.y + threadIdx.y][dim];
      }
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        tmp[idx] *= tmp[idx];
      }
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        batchNorms[idx] += sumc(tmp[idx]);
      }
    } while (VeryHighDim && ((dim += blockDim.x) < vectors.getSize(1)));

#pragma unroll
    for (int idx = 0; idx < BatchSize; ++idx) {
      batchNorms[idx] = warpReduceSum<T>(batchNorms[idx]);
    }

    if (laneId == 0) {
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        smem[idx * numWarps + threadIdx.y * numWarpsPerVec + warpId] =
            batchNorms[idx];
      }
    }
  }

  __syncthreads();

  if ((warpId % numWarpsPerVec) == 0) {
#pragma unroll
    for (int idx = 0; idx < BatchSize; ++idx) {
      batchNorms[idx] =
          laneId < numWarpsPerVec
              ? smem[idx * numWarps + threadIdx.y * numWarpsPerVec + warpId + laneId]
              : NumericTraits<T>::zero();
    }

#pragma unroll
    for (int idx = 0; idx < BatchSize; ++idx) {
      batchNorms[idx] = warpReduceSum<T>(batchNorms[idx]);
    }

    if (laneId == 0) {  // write out results
#pragma unroll
      for (int idx = 0; idx < BatchSize; ++idx) {
        IndexT realIdx = idxOffset + idx * blockDim.y + threadIdx.y;
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
    constexpr IndexT vecsPerBlock = kSuggestThreads / DIM;                    \
    constexpr int batchSize = BatchSize / vecsPerBlock;                       \
    auto grid = dim3(divUp(vectors.getSize(0), batchSize * vecsPerBlock));    \
    auto block = dim3(DIM, vecsPerBlock);                                     \
    auto smem = sizeof(T) * batchSize * vecsPerBlock;                         \
    kernelL2Norm<T, TVec, IndexT, batchSize, false,                    \
                 Squared><<<grid, block, smem, stream>>>(vectors, norms); \
  } while (0)

#define EXECUTE_L2_NORM(DIM, VERY_HIGH_DIM)                                   \
  do {                                                                    \
    auto grid = dim3(divUp(vectors.getSize(0), BatchSize));               \
    auto block = dim3(DIM);                                               \
    auto smem = sizeof(T) * BatchSize * (DIM / kWarpSize);                \
    kernelL2Norm<T, TVec, IndexT, BatchSize, VERY_HIGH_DIM,                   \
                 Squared><<<grid, block, smem, stream>>>(vectors, norms); \
  } while (0)

  if (dim == 32) {
    EXECUTE_L2_NORM_SPECIAL(32);
  } else if (dim == 64) {
    EXECUTE_L2_NORM_SPECIAL(64);
  } else if (dim == 128) {
    EXECUTE_L2_NORM_SPECIAL(128);
  } else if (dim > kMaxThreadsPerBlock) {
    EXECUTE_L2_NORM(kMaxThreadsPerBlock, true);
  } else {
    EXECUTE_L2_NORM(dim, false);
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
