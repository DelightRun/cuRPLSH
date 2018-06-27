#include "internal/search.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>

#include "internal/assertions.h"
#include "internal/broadcast.h"
#include "internal/cuda_utils.h"
#include "internal/kernel_utils.cuh"
#include "internal/matrix_multiply.h"
#include "internal/norm.h"
#include "internal/select.h"
#include "internal/traits.h"

namespace curplsh {

namespace {

template <typename IndexT>
__global__ void kernelAdjustIndex(Tensor<IndexT, 2> indices, int k, int increment) {
  for (IndexT i = threadIdx.x; i < k; i += blockDim.x) {
    // For each chunk of k indices, increase the indices by chunk * increment
    indices[blockIdx.y][blockIdx.x * k + i] += blockIdx.x * increment;
  }
}

// Used to adjust result indices since we use tiled distance computation algorithm
template <typename IndexT>
void adjustIndices(Tensor<IndexT, 2>& indices, int k, int increment,
                   cudaStream_t stream) {
  host_assert(indices.getSize(1) % k == 0);

  auto grid = dim3(indices.getSize(1) / k, indices.getSize(0));
  auto block = dim3(min(k, 512));

  kernelAdjustIndex<<<grid, block, 0, stream>>>(indices, k, increment);

  cudaDeviceSynchronize();
}

template <typename T, typename IndexT, bool Transposed = false>
inline Tensor<T, 2> sliceBases(const Tensor<T, 2>& data, IndexT start, IndexT num) {
  int axis = Transposed ? 1 : 0;
  if (start == 0 && num == data.getSize(axis)) {
    return data;
  } else {
    return data.narrow(axis, start, num);
  }
}

template <typename IndexT>
void chooseTileSize(const IndexT numQueries, const IndexT numBases, const IndexT dim,
                    size_t elementSize, IndexT& queryTileSize,
                    IndexT& baseTileSize) {
  auto globalMem = getCurrentDeviceProperties().totalGlobalMem;

  int targetUsage = 0;
  if (globalMem <= (static_cast<size_t>(4)) * 1024 * 1024 * 1024) {
    targetUsage = 512 * 1024 * 1024;
  } else if (globalMem <= (static_cast<size_t>(8)) * 1024 * 1024 * 1024) {
    targetUsage = 768 * 1024 * 1024;
  } else {
    targetUsage = 1024 * 1024 * 1024;
  }

  targetUsage /= 2 * elementSize;

  IndexT preferredTileQueries = dim <= 32 ? 1024 : 512;

  queryTileSize = std::min(preferredTileQueries, numQueries);
  baseTileSize = std::min(targetUsage / preferredTileQueries, numBases);
}
}  // namespace

template <typename T, typename IndexT>
void searchL2Distance(DeviceResources* resources,
                      const Tensor<T, 2>& bases,  // TODO basesTranspose
                      Tensor<T, 1>* basesNorm, const Tensor<T, 2>& queries, IndexT k,
                      Tensor<IndexT, 2>& indices, Tensor<T, 2>& distances,
                      bool computeExactDistances) {
  runtime_assert(distances.getSize(0) == queries.getSize(0));
  runtime_assert(indices.getSize(0) == queries.getSize(0));
  runtime_assert(distances.getSize(1) == k);
  runtime_assert(indices.getSize(1) == k);

  cudaStream_t defaultStream = resources->getDefaultStreamCurrentDevice();

  // Special Case
  if (bases.getNumElements() == 0) {
    thrust::fill(thrust::cuda::par.on(defaultStream), distances.data(),
                 distances.end(), NumericTraits<T>::max());
    thrust::fill(thrust::cuda::par.on(defaultStream), indices.data(), indices.end(),
                 -1);
    return;
  }

  // Compute bases' norm if necessary
  DeviceTensor<T, 1> bNorms;
  if (!basesNorm) {
    bNorms = std::move(DeviceTensor<T, 1>({bases.getSize(0)}));
    computeL2Norm(bases, bNorms, true, defaultStream);
    basesNorm = &bNorms;
  }

  // Compute queries' norm
  DeviceTensor<T, 1> queriesNorm({queries.getSize(0)});  // TODO stream and mem
  computeL2Norm(queries, queriesNorm, true, defaultStream);

  IndexT queryTileSize;
  IndexT baseTileSize;
  chooseTileSize(queries.getSize(0), bases.getSize(0), queries.getSize(1), sizeof(T),
                 queryTileSize, baseTileSize);

  IndexT numBaseTiles = divUp(bases.getSize(0), baseTileSize);

  host_assert(k <= bases.getSize(0));
  host_assert(k <= 1024);  // Select Limitation

  // Tempory Output Memory
  DeviceTensor<T, 2> tileDistanceBuf1({queryTileSize, baseTileSize});
  DeviceTensor<T, 2> tileDistanceBuf2({queryTileSize, baseTileSize});
  DeviceTensor<T, 2>* tileDistanceBufs[2] = {&tileDistanceBuf1, &tileDistanceBuf2};

  DeviceTensor<T, 2> distancesBuf1({queryTileSize, numBaseTiles * k});
  DeviceTensor<T, 2> distancesBuf2({queryTileSize, numBaseTiles * k});
  DeviceTensor<T, 2>* distancesBufs[2] = {&distancesBuf1, &distancesBuf2};

  DeviceTensor<IndexT, 2> indicesBuf1({queryTileSize, numBaseTiles * k});
  DeviceTensor<IndexT, 2> indicesBuf2({queryTileSize, numBaseTiles * k});
  DeviceTensor<IndexT, 2>* indicesBufs[2] = {&indicesBuf1, &indicesBuf2};

  auto streams = resources->getAlternateStreamsCurrentDevice();
  streamWait(streams, {defaultStream});

  int currentStream = 0;

  // Tile over the queries
  for (IndexT i = 0; i < queries.getSize(0); i += queryTileSize) {
    IndexT currentNumQueries = min(queryTileSize, queries.getSize(0) - i);

    auto distancesView = distances.narrow(0, i, currentNumQueries);
    auto indicesView = indices.narrow(0, i, currentNumQueries);

    auto queriesView = queries.narrow(0, i, currentNumQueries);
    auto queriesNormView = queriesNorm.narrow(0, i, currentNumQueries);

    auto distancesBufQueryView =
        distancesBufs[currentStream]->narrow(0, 0, currentNumQueries);
    auto indicesBufQueryView =
        indicesBufs[currentStream]->narrow(0, 0, currentNumQueries);

    // Tile over the bases
    for (IndexT j = 0; j < bases.getSize(0); j += baseTileSize) {
      IndexT currentNumBases = min(baseTileSize, bases.getSize(0) - j);

      IndexT currentBaseTile = j / baseTileSize;  // index of current tile

      auto basesView = sliceBases(bases, j, currentNumBases);  // TODO transpose base

      auto tileDistanceBufView = tileDistanceBufs[currentStream]
                                     ->narrow(0, 0, currentNumQueries)
                                     .narrow(1, 0, currentNumBases);

      auto distancesBufBaseView =
          distancesBufQueryView.narrow(1, k * currentBaseTile, k);
      auto indicesBufBaseView =
          indicesBufQueryView.narrow(1, k * currentBaseTile, k);

      // Matrix multiply
      matrixMultiply(tileDistanceBufView, false, queriesView, false, basesView, true,
                     -2.0f, 0.0f, resources->getBlasHandleCurrentDevice(),
                     streams[currentStream]);

      if (baseTileSize == bases.getSize(0)) {
        // If there's only 1 tile (i.e. number of bases less than tile size),
        // we directly write into final output
        l2Select(tileDistanceBufView, *basesNorm, distancesView, indicesView, k,
                 streams[currentStream]);
        if (computeExactDistances) {
          // add queries'norm to get real distances
          broadcastAlongRowsSum(queriesNormView, distancesView,
                                streams[currentStream]);
        }
      } else {
        auto basesNormView = basesNorm->narrow(0, j, currentNumBases);
        // Else we need to write into our intermediate output first
        l2Select(tileDistanceBufView, basesNormView, distancesBufBaseView,
                 indicesBufBaseView, k, streams[currentStream]);

        if (computeExactDistances) {
          broadcastAlongRowsSum(queriesNormView, distancesBufBaseView,
                                streams[currentStream]);
        }
      }
    }

    // Perform the final k-selection
    if (baseTileSize != bases.getSize(0)) {
      adjustIndices(indicesBufQueryView, k, baseTileSize, streams[currentStream]);
      blockSelect(distancesBufQueryView, indicesBufQueryView, distancesView,
                  indicesView, k, false, streams[currentStream]);
    }

    // Switch stream
    currentStream = (currentStream + 1) % 2;
  }

  // Have the desired ordering stream waiting on the multi-stream
  streamWait({defaultStream}, streams);
}

void searchL2Distance(DeviceResources* resources, const Tensor<float, 2>& bases,
                      Tensor<float, 1>* basesNorm, const Tensor<float, 2>& queries,
                      int k, Tensor<int, 2>& indices, Tensor<float, 2>& distances,
                      bool computeExactDistances) {
  searchL2Distance<float, int>(resources, bases, basesNorm, queries, k, indices,
                               distances, computeExactDistances);
}

}  // namespace curplsh
