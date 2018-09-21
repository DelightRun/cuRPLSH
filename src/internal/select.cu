#include "internal/select.h"

#include "internal/heap.cuh"
#include "internal/kernel_utils.cuh"
#include "internal/pair.cuh"
#include "internal/radix.cuh"
#include "internal/reductions.cuh"

namespace curplsh {

namespace {

// BlockSelect without indices/labels, use sequential indices implicitly
template <typename T, typename IndexT, bool SelectMax, int NumWarpQ, int NumThreadQ,
          int ThreadsPerBlock>
__global__ void kernelBlockSelect(Tensor<T, 2, IndexT> inK,
                                  Tensor<T, 2, IndexT> outK,
                                  Tensor<IndexT, 2, IndexT> outV, T initK,
                                  IndexT initV, IndexT k) {
  constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

  __shared__ T smemK[kNumWarps * NumWarpQ];
  __shared__ IndexT smemV[kNumWarps * NumWarpQ];

  BlockHeap<T, IndexT, SelectMax, Comparator<T>, NumWarpQ, NumThreadQ,
            ThreadsPerBlock>
      heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  IndexT row = blockIdx.x;

  IndexT i = threadIdx.x;
  T* inStart = inK[row][i].data();

  IndexT limit = roundDown(inK.getSize(1), kWarpSize);

  for (; i < limit; i += ThreadsPerBlock) {
    heap.add(*inStart, i);
    inStart += ThreadsPerBlock;
  }

  // Handle last remainder fraction of a warp of elements
  if (i < inK.getSize(1)) {
    heap.addThreadQ(*inStart, i);
  }

  heap.reduce();

  // Write out results
  for (IndexT i = threadIdx.x; i < k; i += ThreadsPerBlock) {
    outK[row][i] = smemK[i];
    outV[row][i] = smemV[i];
  }
}

// BlockSelect with given indices/labels
template <typename T, typename IndexT, bool SelectMax, int NumWarpQ, int NumThreadQ,
          int ThreadsPerBlock>
__global__ void kernelBlockSelect(Tensor<T, 2, IndexT> inK,
                                  Tensor<IndexT, 2, IndexT> inV,
                                  Tensor<T, 2, IndexT> outK,
                                  Tensor<IndexT, 2, IndexT> outV, T initK,
                                  IndexT initV, IndexT k) {
  constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

  __shared__ T smemK[kNumWarps * NumWarpQ];
  __shared__ IndexT smemV[kNumWarps * NumWarpQ];

  BlockHeap<T, IndexT, SelectMax, Comparator<T>, NumWarpQ, NumThreadQ,
            ThreadsPerBlock>
      heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  IndexT row = blockIdx.x;

  IndexT i = threadIdx.x;
  T* inKStart = inK[row][i].data();
  IndexT* inVStart = inV[row][i].data();

  IndexT limit = roundDown(inK.getSize(1), kWarpSize);

  for (; i < limit; i += ThreadsPerBlock) {
    heap.add(*inKStart, *inVStart);
    inKStart += ThreadsPerBlock;
    inVStart += ThreadsPerBlock;
  }

  // Handle last remainder fraction of a warp of elements
  if (i < inK.getSize(1)) {
    heap.addThreadQ(*inKStart, *inVStart);
  }

  heap.reduce();

  // Write out results
  for (IndexT i = threadIdx.x; i < k; i += ThreadsPerBlock) {
    outK[row][i] = smemK[i];
    outV[row][i] = smemV[i];
  }
}

// RadixSelect without given indices/labels
template <typename T, typename IndexT, bool SelectMax, int RadixBits>
__global__ void kernelRadixSelect(Tensor<T, 2, IndexT> inK,
                                  Tensor<T, 2, IndexT> outK,
                                  Tensor<IndexT, 2, IndexT> outV, IndexT k) {
  constexpr T Identity =
      SelectMax ? NumericTraits<T>::min() : NumericTraits<T>::max();

  __shared__ int smem[kWarpSize];
  __shared__ IndexT idxOffset;

  if (threadIdx.x == 0) {
    idxOffset = 0;
  }
  __syncthreads();

  IndexT row = blockIdx.x;

  // Step 1. Select k-th small/large value
  T kthValue = radixSelectKthElement<T, IndexT, SelectMax, RadixBits>(
      inK[row].data(), inK.getSize(1), k, smem);

  IndexT numRoundedUp = roundUp(inK.getSize(1), (IndexT)blockDim.x);

  // Step 2. Filter all data, output those greater/less than k-th
  for (IndexT i = threadIdx.x; i < numRoundedUp; i += blockDim.x) {
    bool isInRange = i < inK.getSize(1);
    T value = isInRange ? inK[row][i].ldg() : Identity;

    bool hasTopK;
    if (SelectMax) {
      hasTopK = isInRange && (value > kthValue);
    } else {
      hasTopK = isInRange && (value < kthValue);
    }

    /*
    int index, carry;
    // FIXME: Exclusive Scan vs atomicAdd
    */

    if (hasTopK) {
      IndexT idx = atomicAdd(&idxOffset, 1);
      outK[row][idx] = value;
      outV[row][idx] = i;
    }

    __syncthreads();
  }

  // Step 3. Filter all data, output those are equal to k-th
  for (IndexT i = threadIdx.x; i < numRoundedUp; i += blockDim.x) {
    bool isInRange = i < inK.getSize(1);
    T value = isInRange ? inK[row][i].ldg() : NumericTraits<T>::zero();
    bool hasTopK = isInRange && (value == kthValue);

    if (hasTopK) {
      IndexT idx = atomicAdd(&idxOffset, 1);
      if (idx >= k) break;
      outK[row][idx] = value;
      outV[row][idx] = i;
    }

    __syncthreads();
  }
}

template <typename T, typename IndexT, bool SelectMax, int RadixBits>
__global__ void kernelRadixSelect(Tensor<T, 2, IndexT> inK,
                                  Tensor<IndexT, 2, IndexT> inV,
                                  Tensor<T, 2, IndexT> outK,
                                  Tensor<IndexT, 2, IndexT> outV, IndexT k) {
  __shared__ int smem[kWarpSize];
  __shared__ IndexT idxOffset;

  if (threadIdx.x == 0) {
    idxOffset = 0;
  }
  __syncthreads();

  IndexT row = blockIdx.x;

  // Step 1. Select k-th small/large value
  T kthValue = radixSelectKthElement<T, IndexT, SelectMax, RadixBits>(
      inK[row].data(), inK.getSize(1), k, smem);

  IndexT numRoundedUp = roundUp(inK.getSize(1), (IndexT)blockDim.x);

  // Step 2. Filter all data, output those greater/less than k-th
  for (IndexT i = threadIdx.x; i < numRoundedUp; i += blockDim.x) {
    bool isInRange = i < inK.getSize(1);

    T value = isInRange ? inK[row][i].ldg() : NumericTraits<T>::zero();
    IndexT index = isInRange ? inV[row][i].ldg() : (IndexT)-1;

    bool hasTopK;
    if (SelectMax) {
      hasTopK = isInRange && (value > kthValue);
    } else {
      hasTopK = isInRange && (value < kthValue);
    }

    /*
    int index, carry;
    // FIXME: Exclusive Scan vs atomicAdd
    */

    if (hasTopK) {
      IndexT idx = atomicAdd(&idxOffset, 1);
      outK[row][idx] = value;
      outV[row][idx] = index;
    }
    __syncthreads();
  }

  // Step 3. Filter all data, output those are equal to k-th
  runtime_assert(outV.getSize(1) >= idxOffset);
  for (IndexT i = threadIdx.x; i < numRoundedUp; i += blockDim.x) {
    bool isInRange = i < inK.getSize(1);

    T value = isInRange ? inK[row][i].ldg() : NumericTraits<T>::zero();
    IndexT index = isInRange ? inV[row][i].ldg() : (IndexT)-1;

    bool hasTopK = isInRange && (value == kthValue);

    if (hasTopK) {
      IndexT idx = atomicAdd(&idxOffset, 1);
      if (idx >= k) break;
      outK[row][idx] = value;
      outV[row][idx] = index;
    }
    __syncthreads();
  }
}

// Speical K-Selection implementation for L2 distance with k == 1
template <typename T, typename IndexT, int kRowsPerBlock, int kBlockSize>
__global__ void kernelL2Select1(Tensor<T, 2, IndexT> productDistances,
                                Tensor<T, 1, IndexT> baseNorms,
                                Tensor<T, 2, IndexT> distances,
                                Tensor<IndexT, 2, IndexT> indices) {
  // Each block handles kRowsPerBlock rows of the distances (results)
  __shared__ Pair<T, IndexT> blockMin[kRowsPerBlock * (kBlockSize / kWarpSize)];

  Pair<T, IndexT> threadMin[kRowsPerBlock];

  T distance[kRowsPerBlock];

#pragma unroll
  for (int i = 0; i < kRowsPerBlock; ++i) {
    threadMin[i].k = NumericTraits<T>::max();
    threadMin[i].v = -1;
  }

  // blockIdx.x: which chunk of rows we are responsible for updating
  IndexT rowStart = blockIdx.x * kRowsPerBlock;

  // FIXME: if we have exact multiples, don't need this
  bool endRow = (blockIdx.x == gridDim.x - 1) &&
                (productDistances.getSize(0) % kRowsPerBlock == 0);

  if (endRow) {
    for (IndexT row = rowStart; row < productDistances.getSize(0); ++row) {
      for (IndexT col = threadIdx.x; col < productDistances.getSize(1);
           col += blockDim.x) {
        distance[0] = baseNorms[col] + productDistances[row][col];

        if (distance[0] < threadMin[0].k) {
          threadMin[0].k = distance[0];
          threadMin[0].v = col;
        }
      }  // end for col

      // Reduce within the block
      threadMin[0] = blockReduce<Pair<T, IndexT>, Min, false>(
          threadMin[0], Min<Pair<T, IndexT>>(), blockMin);

      if (threadIdx.x == 0) {
        distances[row][0] = threadMin[0].k;
        indices[row][0] = threadMin[0].v;
      }

      // so we can use the shared memory again
      __syncthreads();

      threadMin[0].k = NumericTraits<T>::max();
      threadMin[0].v = -1;
    }
  } else {
    for (IndexT col = threadIdx.x; col < productDistances.getSize(1);
         col += blockDim.x) {
      T baseNorm =
          baseNorms[col];  // NOTE: load into register for better performance

#pragma unroll
      for (IndexT row = 0; row < kRowsPerBlock; ++row) {
        distance[row] = productDistances[rowStart + row][col];
      }

#pragma unroll
      for (IndexT row = 0; row < kRowsPerBlock; ++row) {
        distance[row] = distance[row] + baseNorm;
      }

#pragma unroll
      for (IndexT row = 0; row < kRowsPerBlock; ++row) {
        if (distance[row] < threadMin[row].k) {
          threadMin[row].k = distance[row];
          threadMin[row].v = col;
        }
      }
    }  // end for col

    // Reduce within the block
    blockReduce<Pair<T, IndexT>, kRowsPerBlock, Min, false>(
        threadMin, Min<Pair<T, IndexT>>(), blockMin);

    if (threadIdx.x == 0) {
#pragma unroll
      for (IndexT row = 0; row < kRowsPerBlock; ++row) {
        distances[rowStart + row][0] = threadMin[row].k;
        indices[rowStart + row][0] = threadMin[row].v;
      }
    }
  }
}

// Speical K-Selection implementation for L2 distance with arbitrary k
template <typename T, typename IndexT, int NumWarpQ, int NumThreadQ,
          int ThreadsPerBlock>
__global__ void kernelL2SelectK(Tensor<T, 2, IndexT> productDistances,
                                Tensor<T, 1, IndexT> baseNorms,
                                Tensor<T, 2, IndexT> distances,
                                Tensor<IndexT, 2, IndexT> indices, IndexT k,
                                T initK) {
  // Each block handles a single row of the distances (results)
  constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

  __shared__ T smemK[kNumWarps * NumWarpQ];
  __shared__ IndexT smemV[kNumWarps * NumWarpQ];

  BlockHeap<T, IndexT, false, Comparator<T>, NumWarpQ, NumThreadQ, ThreadsPerBlock>
      heap(initK, -1, smemK, smemV, k);

  IndexT row = blockIdx.x;

  // Whole warps must participate in the selection
  IndexT limit = roundDown(productDistances.getSize(1), kWarpSize);
  IndexT i = threadIdx.x;

  for (; i < limit; i += blockDim.x) {
    T v = baseNorms[i] + productDistances[row][i];
    heap.add(v, i);
  }

  if (i < productDistances.getSize(1)) {
    T v = baseNorms[i] + productDistances[row][i];
    heap.addThreadQ(v, i);
  }

  heap.reduce();
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    distances[row][i] = smemK[i];
    indices[row][i] = smemV[i];
  }
}

}  // namespace

template <typename T, typename IndexT>
void blockSelect(const Tensor<T, 2, IndexT>& in, Tensor<T, 2, IndexT>& outK,
                 Tensor<IndexT, 2, IndexT>& outV, IndexT k, bool selecMax,
                 cudaStream_t stream) {
  constexpr int kBlockSelectNumThreads = 128;

  host_assert(outK.isSameSizes(outV));
  host_assert(outK.getSize(1) == k);
  host_assert(in.getSize(0) == outK.getSize(0));

  auto grid = dim3(in.getSize(0));
  auto block = dim3(kBlockSelectNumThreads);

  auto initK = selecMax ? NumericTraits<T>::min() : NumericTraits<T>::max();
  auto initV = -1;

#define EXECUTE_BLOCK_SELECT(WARP_Q, NUM_NUM_THREAD_Q)                   \
  do {                                                                   \
    if (selecMax) {                                                      \
      kernelBlockSelect<T, IndexT, true, WARP_Q, NUM_NUM_THREAD_Q,       \
                        kBlockSelectNumThreads>                          \
          <<<grid, block, 0, stream>>>(in, outK, outV, initK, initV, k); \
    } else {                                                             \
      kernelBlockSelect<T, IndexT, false, WARP_Q, NUM_NUM_THREAD_Q,      \
                        kBlockSelectNumThreads>                          \
          <<<grid, block, 0, stream>>>(in, outK, outV, initK, initV, k); \
    }                                                                    \
    getLastCudaError("Check CUDA error");                                \
  } while (0)

  if (k == 1) {
    EXECUTE_BLOCK_SELECT(1, 1);
  } else if (k <= 32) {
    EXECUTE_BLOCK_SELECT(32, 2);
  } else if (k <= 64) {
    EXECUTE_BLOCK_SELECT(64, 3);
  } else if (k <= 128) {
    EXECUTE_BLOCK_SELECT(128, 3);
  } else if (k <= 256) {
    EXECUTE_BLOCK_SELECT(256, 4);
  } else if (k <= 512) {
    EXECUTE_BLOCK_SELECT(512, 8);
  } else if (k <= 1024) {
    EXECUTE_BLOCK_SELECT(1024, 8);
  }
#undef EXECUTE_BLOCK_SELECT
}

template <typename T, typename IndexT>
void blockSelect(const Tensor<T, 2, IndexT>& inK,
                 const Tensor<IndexT, 2, IndexT>& inV, Tensor<T, 2, IndexT>& outK,
                 Tensor<IndexT, 2, IndexT>& outV, IndexT k, bool selecMax,
                 cudaStream_t stream) {
  constexpr int kBlockSelectNumThreads = 128;

  host_assert(inK.isSameSizes(inV));
  host_assert(outK.isSameSizes(outV));
  host_assert(outK.getSize(1) == k);
  host_assert(inK.getSize(0) == outK.getSize(0));

  auto grid = dim3(inK.getSize(0));
  auto block = dim3(kBlockSelectNumThreads);

  auto initK = selecMax ? NumericTraits<T>::min() : NumericTraits<T>::max();
  auto initV = -1;

#define EXECUTE_BLOCK_SELECT(NUM_WARP_Q, NUM_THREAD_Q)                         \
  do {                                                                         \
    if (selecMax) {                                                            \
      kernelBlockSelect<T, IndexT, true, NUM_WARP_Q, NUM_THREAD_Q,             \
                        kBlockSelectNumThreads>                                \
          <<<grid, block, 0, stream>>>(inK, inV, outK, outV, initK, initV, k); \
    } else {                                                                   \
      kernelBlockSelect<T, IndexT, false, NUM_WARP_Q, NUM_THREAD_Q,            \
                        kBlockSelectNumThreads>                                \
          <<<grid, block, 0, stream>>>(inK, inV, outK, outV, initK, initV, k); \
    }                                                                          \
    getLastCudaError("Check CUDA error");                                      \
  } while (0)

  if (k == 1) {
    EXECUTE_BLOCK_SELECT(1, 1);
  } else if (k <= 32) {
    EXECUTE_BLOCK_SELECT(32, 2);
  } else if (k <= 64) {
    EXECUTE_BLOCK_SELECT(64, 3);
  } else if (k <= 128) {
    EXECUTE_BLOCK_SELECT(128, 3);
  } else if (k <= 256) {
    EXECUTE_BLOCK_SELECT(256, 4);
  } else if (k <= 512) {
    EXECUTE_BLOCK_SELECT(512, 8);
  } else if (k <= 1024) {
    EXECUTE_BLOCK_SELECT(1024, 8);
  }
#undef EXECUTE_BLOCK_SELECT
}

template <typename T, typename IndexT>
void radixSelect(const Tensor<T, 2, IndexT> inK, Tensor<T, 2, IndexT> outK,
                 Tensor<IndexT, 2, IndexT> outV, IndexT k, bool selectMax,
                 cudaStream_t stream) {
  host_assert(inK.getSize(0) == outK.getSize(0));
  host_assert(outK.getSize(0) == outV.getSize(0));
  host_assert(outK.getSize(1) == outV.getSize(1));
  host_assert(outK.getSize(1) == k);

  // FIXME: special case - k == 1
  int numThreads = roundUp(k, kWarpSize);

  auto block = dim3(std::min(numThreads, kMaxThreadsPerBlock));
  auto grid = dim3(inK.getSize(0));

#define EXECUTE_RADIX_SELECT(RADIX_BITS)                  \
  if (selectMax) {                                        \
    kernelRadixSelect<T, IndexT, true, 2>                 \
        <<<grid, block, 0, stream>>>(inK, outK, outV, k); \
  } else {                                                \
    kernelRadixSelect<T, IndexT, false, 2>                \
        <<<grid, block, 0, stream>>>(inK, outK, outV, k); \
  }

  EXECUTE_RADIX_SELECT(2);
#undef EXECUTE_RADIX_SELECT
}

template <typename T, typename IndexT>
void radixSelect(const Tensor<T, 2, IndexT> inK, const Tensor<IndexT, 2, IndexT> inV,
                 Tensor<T, 2, IndexT> outK, Tensor<IndexT, 2, IndexT> outV, IndexT k,
                 bool selectMax, cudaStream_t stream) {
  host_assert(inK.getSize(0) == inV.getSize(0));
  host_assert(inK.getSize(1) == inV.getSize(1));
  host_assert(inK.getSize(0) == outK.getSize(0));
  host_assert(outK.getSize(0) == outV.getSize(0));
  host_assert(outK.getSize(1) == outV.getSize(1));
  host_assert(outV.getSize(1) == k);

  // FIXME: special case - k == 1
  int numThreads = roundUp(k, kWarpSize);

  auto block = dim3(std::min(numThreads, kMaxThreadsPerBlock));
  auto grid = dim3(inK.getSize(0));

#define EXECUTE_RADIX_SELECT(RADIX_BITS)                       \
  if (selectMax) {                                             \
    kernelRadixSelect<T, IndexT, true, 2>                      \
        <<<grid, block, 0, stream>>>(inK, inV, outK, outV, k); \
  } else {                                                     \
    kernelRadixSelect<T, IndexT, false, 2>                     \
        <<<grid, block, 0, stream>>>(inK, inV, outK, outV, k); \
  }

  EXECUTE_RADIX_SELECT(2);
#undef EXECUTE_RADIX_SELECT
}

// FIXME: specialization for TVec types
template <typename T, typename IndexT>
void l2Select(const Tensor<T, 2, IndexT>& productDistances,
              const Tensor<T, 1, IndexT>& baseNorms, Tensor<T, 2, IndexT>& distances,
              Tensor<IndexT, 2, IndexT>& indices, IndexT k, cudaStream_t stream) {
  host_assert(productDistances.getSize(0) == distances.getSize(0));
  host_assert(productDistances.getSize(0) == indices.getSize(0));
  host_assert(productDistances.getSize(1) == baseNorms.getSize(0));
  host_assert(distances.getSize(1) == k);
  host_assert(indices.getSize(1) == k);
  host_assert(k <= 1024);  // WarpSelect cannot handle large k

  if (k == 1) {
    constexpr int kThreadsPerBlock = 256;
    constexpr int kRowsPerBlock = 8;

    auto block = dim3(kThreadsPerBlock);
    auto grid = dim3(divUp(distances.getSize(0), kRowsPerBlock));

    kernelL2Select1<T, IndexT, kRowsPerBlock, kThreadsPerBlock>
        <<<grid, block, 0, stream>>>(productDistances, baseNorms, distances,
                                     indices);
  } else {
    constexpr int kThreadsPerBlock = 128;

    auto block = dim3(kThreadsPerBlock);
    auto grid = dim3(distances.getSize(0));

#define EXECUTE_L2_SELECT(NUM_WARP_Q, NUM_THREAD_Q)                          \
  do {                                                                       \
    kernelL2SelectK<T, IndexT, NUM_WARP_Q, NUM_THREAD_Q, kThreadsPerBlock>   \
        <<<grid, block, 0, stream>>>(productDistances, baseNorms, distances, \
                                     indices, k, NumericTraits<T>::max());   \
    getLastCudaError("Check CUDA error");                                    \
  } while (0)

    if (k <= 32) {
      EXECUTE_L2_SELECT(32, 2);
    } else if (k <= 64) {
      EXECUTE_L2_SELECT(64, 3);
    } else if (k <= 128) {
      EXECUTE_L2_SELECT(128, 3);
    } else if (k <= 256) {
      EXECUTE_L2_SELECT(256, 4);
    } else if (k <= 512) {
      EXECUTE_L2_SELECT(512, 8);
    } else if (k <= 1024) {
      EXECUTE_L2_SELECT(1024, 8);
    } else {
      host_assert(false);
    }
  }
#undef EXECUTE_L2_SELECT
}

void blockSelect(const Tensor<float, 2>& inK,  //
                 Tensor<float, 2>& outK,       //
                 Tensor<int, 2>& outV,         //
                 int k,                        //
                 bool selectMax,               //
                 cudaStream_t stream) {
  blockSelect<float, int>(inK, outK, outV, k, selectMax, stream);
}

void blockSelect(const Tensor<float, 2>& inK,  //
                 const Tensor<int, 2>& inV,    //
                 Tensor<float, 2>& outK,       //
                 Tensor<int, 2>& outV,         //
                 int k,                        //
                 bool selectMax,               //
                 cudaStream_t stream) {
  blockSelect<float, int>(inK, inV, outK, outV, k, selectMax, stream);
}

void radixSelect(const Tensor<unsigned, 2>& inK,  //
                 Tensor<unsigned, 2>& outK,       //
                 Tensor<int, 2>& outV,            //
                 int k,                           //
                 bool selectMax,                  //
                 cudaStream_t stream) {
  radixSelect<unsigned, int>(inK, outK, outV, k, selectMax, stream);
}

void radixSelect(const Tensor<unsigned, 2>& inK,  //
                 const Tensor<int, 2>& inV,       //
                 Tensor<unsigned, 2>& outK,       //
                 Tensor<int, 2>& outV,            //
                 int k,                           //
                 bool selectMax,                  //
                 cudaStream_t stream) {
  radixSelect<unsigned, int>(inK, inV, outK, outV, k, selectMax, stream);
}

void l2Select(const Tensor<float, 2>& productDistances,  //
              const Tensor<float, 1>& baseNorms,         //
              Tensor<float, 2>& distances,               //
              Tensor<int, 2>& indices,                   //
              int k,                                     //
              cudaStream_t stream) {
  l2Select<float, int>(productDistances, baseNorms, distances, indices, k, stream);
}

}  // namespace curplsh
