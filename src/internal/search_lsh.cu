#include "internal/search_lsh.h"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>

#include "timer.h"

#include "internal/assertions.h"
#include "internal/cuda_utils.h"
#include "internal/hash_ops.h"
#include "internal/kernel_utils.cuh"
#include "internal/matrix_ops.h"
#include "internal/norm.h"
#include "internal/reductions.cuh"
#include "internal/select.h"
#include "internal/traits.h"

namespace curplsh {

namespace {

template <typename IndexT>
__global__ void kernelAdjustIndex(Tensor<IndexT, 2, IndexT> indices, int chunk,
                                  int increment) {
  for (IndexT i = threadIdx.x; i < chunk; i += blockDim.x) {
    // For each chunk of k indices, increase the indices by chunk * increment
    indices[blockIdx.y][blockIdx.x * chunk + i] += blockIdx.x * increment;
  }
}

// Used to adjust result indices since we use tiled distance computation algorithm
template <typename IndexT>
void adjustIndices(Tensor<IndexT, 2, IndexT>& indices, int chunk, int increment,
                   cudaStream_t stream) {
  host_assert(indices.getSize(1) % chunk == 0);

  auto grid = dim3(indices.getSize(1) / chunk, indices.getSize(0));
  auto block = dim3(min(chunk, kMaxThreadsPerBlock / 2));

  kernelAdjustIndex<<<grid, block, 0, stream>>>(indices, chunk, increment);

  cudaDeviceSynchronize();
}

template <typename T, typename TVec, typename IndexT>
__global__ void kernelL2Distance(Tensor<TVec, 2, IndexT> queries,
                                 Tensor<TVec, 2, IndexT> bases,
                                 Tensor<T, 1, IndexT> basesNorm,
                                 Tensor<IndexT, 2, IndexT> idsLists,
                                 Tensor<T, 2, IndexT> distances) {
  extern __shared__ char smemByte[];
  T* smem = (T*)smemByte;

  int numWarps = blockDim.x / kWarpSize;

  int laneId = getLaneId();
  int warpId = threadIdx.x / kWarpSize;

  IndexT dimension = queries.getSize(1);  // TODO: VeryHighDim
  IndexT numCandidates = idsLists.getSize(1);

  IndexT queryId = blockIdx.x * blockDim.x + threadIdx.y;
  IndexT currentDim = threadIdx.x;

  if (queryId >= queries.getSize(0)) return;

  // TODO: unroll
  TVec qval = queries[queryId][currentDim];

  for (IndexT i = 0; i < numCandidates; ++i) {
    IndexT baseId = idsLists[queryId][i];  // TODO: can speedUp use cache?

    __syncthreads();

    T norm = basesNorm[baseId];
    T prod = -2 * sumc(qval * bases[baseId][currentDim]);

    prod = warpReduceSum<T>(prod);

    if (laneId == 0) {
      smem[threadIdx.y * numWarps + warpId] = prod;
    }

    __syncthreads();

    if (warpId == 0) {
      prod = laneId < numWarps ? smem[threadIdx.y * numWarps + laneId]
                               : NumericTraits<T>::zero();
      prod = warpReduceSum<T>(prod);

      if (laneId == 0) {
        distances[queryId][i] = prod + norm;
      }
    }
  }
}

template <typename T, typename IndexT>
void computeL2Distance(const Tensor<T, 2, IndexT>& queries,
                       const Tensor<T, 2, IndexT>& bases,
                       const Tensor<T, 1, IndexT>& basesNorm,
                       const Tensor<IndexT, 2, IndexT>& idsLists,
                       Tensor<T, 2, IndexT>& products, cudaStream_t stream) {
  host_assert(queries.getSize(1) == bases.getSize(1));
  host_assert(queries.getSize(0) == idsLists.getSize(0));
  host_assert(queries.getSize(0) == products.getSize(0));
  host_assert(idsLists.getSize(1) == products.getSize(1));

#define EXECUTE_L2_DISTANCE(TVec, QUERIES, BASES)                                 \
  do {                                                                            \
    int numQueryPerBlock = kMaxThreadsPerBlock / QUERIES.getSize(1);              \
    auto block =                                                                  \
        dim3(QUERIES.getSize(1), std::min(QUERIES.getSize(0), numQueryPerBlock)); \
    auto grid = dim3(divUp(QUERIES.getSize(0), block.y));                         \
    auto smem = sizeof(T) * block.x * block.y / kWarpSize;                        \
    kernelL2Distance<T, TVec, IndexT><<<grid, block, smem, stream>>>(             \
        QUERIES, BASES, basesNorm, idsLists, products);                           \
    cudaDeviceSynchronize();                                                      \
  } while (0)

  if (std::is_same<T, float>::value) {
    auto dim = queries.getSize(1);
    if (dim % (4 * kWarpSize) == 0) {
      EXECUTE_L2_DISTANCE(float4, queries.template cast<float4>(),
                          bases.template cast<float4>());
    } else if (dim % (2 * kWarpSize) == 0) {
      EXECUTE_L2_DISTANCE(float2, queries.template cast<float2>(),
                          bases.template cast<float2>());
    }
  } else {
    EXECUTE_L2_DISTANCE(T, queries, bases);
  }
}

template <typename IndexT>
void chooseTileSize(const IndexT numQueries, const IndexT numBases,
                    size_t elementSize, IndexT& queryTileSize,
                    IndexT& baseTileSize) {
  auto globalMem = getCurrentDeviceProperties().totalGlobalMem;

  auto targetUsage = globalMem;
  if (globalMem <= (static_cast<size_t>(4)) * 1024 * 1024 * 1024) {
    targetUsage = static_cast<size_t>(2) * 1024 * 1024 * 1024;
  } else if (globalMem <= (static_cast<size_t>(8)) * 1024 * 1024 * 1024) {
    targetUsage = 768 * 1024 * 1024;
  } else {
    targetUsage = 1024 * 1024 * 1024;
  }

  targetUsage /= 2 * elementSize;

  IndexT preferredTileQueries = 1024;
  IndexT preferredTileBases = targetUsage / preferredTileQueries;

  queryTileSize = std::min(preferredTileQueries, numQueries);
  baseTileSize = std::min(preferredTileBases, numBases);
}

}  // namespace

template <typename T, typename CodeT, typename IndexT>
void searchHammingDistance(DeviceResources* resources,
                           const Tensor<T, 2, IndexT>& bases,
                           const Tensor<T, 1, IndexT>* basesNorm,
                           const Tensor<CodeT, 2, IndexT>& basesCode,
                           const Tensor<T, 2, IndexT>& projMatrix,
                           const Tensor<T, 2, IndexT>& queries, IndexT k,
                           Tensor<IndexT, 2, IndexT>& indices,
                           Tensor<T, 2, IndexT>& distances, IndexT numCandidates) {
  host_assert(bases.getSize(0) == basesCode.getSize(0));
  host_assert(indices.getSize(0) == queries.getSize(0));
  host_assert(indices.getSize(0) == distances.getSize(0));
  host_assert(indices.getSize(1) == distances.getSize(1));
  host_assert(indices.getSize(1) == k);
  host_assert(numCandidates >= k);

  cudaStream_t defaultStream = resources->getDefaultStreamCurrentDevice();

  if (bases.getSize(0) == 0) {
    thrust::fill(thrust::cuda::par.on(defaultStream), indices.data(), indices.end(),
                 (IndexT)-1);
    return;
  }

  IndexT numQueries = queries.getSize(0);
  IndexT numBases = bases.getSize(0);
  IndexT numCodes = projMatrix.getSize(1);
  IndexT numTables = basesCode.getSize(1);  // == numCodes * sizeof(CodeT) * 8

  // Compute bases' norm if necessary
  DeviceTensor<T, 1, IndexT> bNorms;
  if (!basesNorm) {
    bNorms = std::move(DeviceTensor<T, 1, IndexT>({bases.getSize(0)}));
    computeL2Norm(bases, bNorms, true, defaultStream);
    basesNorm = &bNorms;
  }

  numCandidates = std::max(numCandidates, k);

  IndexT queryTileSize, baseTileSize;
  chooseTileSize(queries.getSize(0), bases.getSize(0), sizeof(CodeT), queryTileSize,
                 baseTileSize);

  IndexT numBaseTiles = divUp(bases.getSize(0), baseTileSize);

  IndexT numCandidatesAll = numBaseTiles * numCandidates;

  DeviceTensor<T, 2, IndexT> tileProjectionBuf1({queryTileSize, numCodes});
  DeviceTensor<T, 2, IndexT> tileProjectionBuf2({queryTileSize, numCodes});
  DeviceTensor<T, 2, IndexT>* tileProjectionBufs[2] = {&tileProjectionBuf1,
                                                       &tileProjectionBuf2};

  DeviceTensor<CodeT, 2, IndexT> tileQueriesCodeBuf1({queryTileSize, numTables});
  DeviceTensor<CodeT, 2, IndexT> tileQueriesCodeBuf2({queryTileSize, numTables});
  DeviceTensor<CodeT, 2, IndexT>* tileQueriesCodeBufs[2] = {&tileQueriesCodeBuf1,
                                                            &tileQueriesCodeBuf2};

  DeviceTensor<CodeT, 2, IndexT> tileHammingBuf1({queryTileSize, baseTileSize});
  DeviceTensor<CodeT, 2, IndexT> tileHammingBuf2({queryTileSize, baseTileSize});
  DeviceTensor<CodeT, 2, IndexT>* tileHammingBufs[2] = {&tileHammingBuf1,
                                                        &tileHammingBuf2};

  DeviceTensor<CodeT, 2, IndexT> tileHammingSortBuf1(
      {queryTileSize, numCandidatesAll}, MemorySpace::Unified);
  DeviceTensor<CodeT, 2, IndexT> tileHammingSortBuf2(
      {queryTileSize, numCandidatesAll}, MemorySpace::Unified);
  DeviceTensor<CodeT, 2, IndexT>* tileHammingSortBufs[2] = {&tileHammingSortBuf1,
                                                            &tileHammingSortBuf2};

  DeviceTensor<IndexT, 2, IndexT> tileIndicesSortBuf1(
      {queryTileSize, numCandidatesAll}, MemorySpace::Unified);
  DeviceTensor<IndexT, 2, IndexT> tileIndicesSortBuf2(
      {queryTileSize, numCandidatesAll}, MemorySpace::Unified);
  DeviceTensor<IndexT, 2, IndexT>* tileIndicesSortBufs[2] = {&tileIndicesSortBuf1,
                                                             &tileIndicesSortBuf2};

  DeviceTensor<CodeT, 2, IndexT> tileCandidateDistBuf1(
      {queryTileSize, numCandidates}, MemorySpace::Unified);
  DeviceTensor<CodeT, 2, IndexT> tileCandidateDistBuf2(
      {queryTileSize, numCandidates}, MemorySpace::Unified);
  DeviceTensor<CodeT, 2, IndexT>* tileCandidateDistBufs[2] = {
      &tileCandidateDistBuf1, &tileCandidateDistBuf2};

  DeviceTensor<IndexT, 2, IndexT> tileCandidateIndicesBuf1(
      {queryTileSize, numCandidates}, MemorySpace::Unified);
  DeviceTensor<IndexT, 2, IndexT> tileCandidateIndicesBuf2(
      {queryTileSize, numCandidates}, MemorySpace::Unified);
  DeviceTensor<IndexT, 2, IndexT>* tileCandidateIndicesBufs[2] = {
      &tileCandidateIndicesBuf1, &tileCandidateIndicesBuf2};

  DeviceTensor<T, 2, IndexT> tileL2DistanceBuf1({queryTileSize, numCandidates});
  DeviceTensor<T, 2, IndexT> tileL2DistanceBuf2({queryTileSize, numCandidates});
  DeviceTensor<T, 2, IndexT>* tileL2DistanceBufs[2] = {&tileL2DistanceBuf1,
                                                       &tileL2DistanceBuf2};

  auto streams = resources->getAlternateStreamsCurrentDevice();
  // cudaStream_t streams[2] = {defaultStream, defaultStream};
  streamWait(streams, {defaultStream});

  int currentStream = 0;

  for (IndexT i = 0; i < queries.getSize(0); i += queryTileSize) {
    IndexT currentNumQueries = std::min(queryTileSize, queries.getSize(0) - i);

    auto queriesView = queries.narrow(0, i, currentNumQueries);

    auto indicesView = indices.narrow(0, i, currentNumQueries);

    auto distancesView = distances.narrow(0, i, currentNumQueries);

    auto tileProjectionBufView =
        tileProjectionBufs[currentStream]->narrow(0, 0, currentNumQueries);

    auto tileQueriesCodeBufView =
        tileQueriesCodeBufs[currentStream]->narrow(0, 0, currentNumQueries);

    auto tileHammingSortBufView =
        tileHammingSortBufs[currentStream]->narrow(0, 0, currentNumQueries);

    auto tileIndicesSortBufView =
        tileIndicesSortBufs[currentStream]->narrow(0, 0, currentNumQueries);

    matrixMultiply(tileProjectionBufView, false, queriesView, false, projMatrix,
                   false, 1.0f, 0.0f, resources->getBlasHandleCurrentDevice(),
                   streams[currentStream]);
    binarize(tileProjectionBufView, tileQueriesCodeBufView, streams[currentStream]);

    for (IndexT j = 0; j < bases.getSize(0); j += baseTileSize) {
      IndexT currentNumBases = std::min(baseTileSize, bases.getSize(0) - j);

      IndexT currentBaseTile = j / baseTileSize;

      auto basesCodeView = basesCode.narrow(0, j, currentNumBases);

      auto tileHammingBufView = tileHammingBufs[currentStream]
                                    ->narrow(0, 0, currentNumQueries)
                                    .narrow(1, 0, currentNumBases);

      {
        // DeviceTimer timer("Hamming Distance");
        computeHammingDistance(tileQueriesCodeBufView, basesCodeView,
                               tileHammingBufView, streams[currentStream]);
        // cudaDeviceSynchronize();
      }

      auto tileHammingSortDistBufView = tileHammingSortBufView.narrow(
          1, numCandidates * currentBaseTile, numCandidates);
      auto tileIndicesSortDistBufView = tileIndicesSortBufView.narrow(
          1, numCandidates * currentBaseTile, numCandidates);

      {
        // DeviceTimer timer("Select");
        if (numCandidates <= 1024) {
          blockSelect(tileHammingBufView, tileHammingSortDistBufView,
                      tileIndicesSortDistBufView, numCandidates, false,
                      streams[currentStream]);
        } else {
          radixSelect(tileHammingBufView, tileHammingSortDistBufView,
                      tileIndicesSortDistBufView, numCandidates, false,
                      streams[currentStream]);
        }
        // cudaDeviceSynchronize();
      }
    }

    auto tileCandidateDistBufView =
        tileCandidateDistBufs[currentStream]->narrow(0, 0, currentNumQueries);
    auto tileCandidateIndicesBufView =
        tileCandidateIndicesBufs[currentStream]->narrow(0, 0, currentNumQueries);

    // Final Sort Hamming Distance
    adjustIndices(tileIndicesSortBufView, numCandidates, baseTileSize,
                  streams[currentStream]);

    if (numCandidates <= 1024) {
      blockSelect(tileHammingSortBufView, tileIndicesSortBufView,
                  tileCandidateDistBufView, tileCandidateIndicesBufView,
                  numCandidates, false, streams[currentStream]);
    } else {
      radixSelect(tileHammingSortBufView, tileIndicesSortBufView,
                  tileCandidateDistBufView, tileCandidateIndicesBufView,
                  numCandidates, false, streams[currentStream]);
    }

    auto tileL2DistanceBufView =
        tileL2DistanceBufs[currentStream]->narrow(0, 0, currentNumQueries);

    // Step 1. L2 Distance
    {
      DeviceTimer timer("L2 Distance");
      computeL2Distance(queriesView, bases, *basesNorm, tileCandidateIndicesBufView,
                        tileL2DistanceBufView, streams[currentStream]);
      cudaDeviceSynchronize();
    }

    // Step 2. K-Select
    blockSelect(tileL2DistanceBufView, tileCandidateIndicesBufView, distancesView,
                indicesView, k, false, streams[currentStream]);

    // TODO: Compute actual distances if necessary

    currentStream = (currentStream + 1) % 2;
  }

  streamWait({defaultStream}, streams);
}  // namespace curplsh

void searchHammingDistance(DeviceResources* resources,
                           const Tensor<float, 2, int>& bases,
                           const Tensor<float, 1, int>* basesNorm,
                           const Tensor<unsigned, 2, int>& basesCode,
                           const Tensor<float, 2, int>& projMatrix,
                           const Tensor<float, 2, int>& queries, int k,
                           Tensor<int, 2, int>& indices,
                           Tensor<float, 2, int>& distances, int numCandidates) {
  searchHammingDistance<float, unsigned, int>(resources, bases, basesNorm, basesCode,
                                              projMatrix, queries, k, indices,
                                              distances, numCandidates);
}

}  // namespace curplsh
