#include "internal/build_lsh.h"

#include "internal/kernel_utils.cuh"
#include "internal/matrix_ops.h"
#include "internal/hash_ops.h"

namespace curplsh {

namespace {

template <typename IndexT>
IndexT chooseTileSize(const IndexT num, const IndexT dim, const IndexT ntables,
                      size_t elementSize) {
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
  return std::min(targetUsage / 1024, num);
}
}

template <typename T, typename IndexT>
inline void generateProjectionMatrix(DeviceResources* resources,
                                     Tensor<T, 2, IndexT> matrix, float mean,
                                     float stddev, unsigned long long seed) {
  matrixRandomInitialize(matrix, mean, stddev,
                         resources->getRandGeneratorCurrentDevice(), seed);
}

template <typename T, typename IndexT, typename CodeT>
void buildCodebook(DeviceResources* resources, const Tensor<T, 2, IndexT>& data,
                   Tensor<T, 2, IndexT>& projMatrix,
                   Tensor<CodeT, 2, IndexT>& codebook) {
  runtime_assert(data.getSize(1) == projMatrix.getSize(0));
  runtime_assert(data.getSize(0) == codebook.getSize(0));
  runtime_assert(projMatrix.getSize(1) == codebook.getSize(1) * sizeof(CodeT) * 8);

  if (data.getSize(0) == 0) return;

  IndexT tileSize = chooseTileSize(data.getSize(0), data.getSize(1),
                                   codebook.getSize(1), sizeof(T));
  IndexT numTiles = divUp(data.getSize(0), tileSize);

  // Tempory Output Memory
  DeviceTensor<T, 2, IndexT> tileProjectionBuf1({tileSize, projMatrix.getSize(1)});
  DeviceTensor<T, 2, IndexT> tileProjectionBuf2({tileSize, projMatrix.getSize(1)});
  DeviceTensor<T, 2, IndexT>* tileProjectionBufs[2] = {&tileProjectionBuf1,
                                                       &tileProjectionBuf2};

  auto defaultStream = resources->getDefaultStreamCurrentDevice();
  auto streams = resources->getAlternateStreamsCurrentDevice();
  streamWait(streams, {defaultStream});

  int currentStream = 0;

  // Tile over data
  for (IndexT i = 0; i < data.getSize(0); i += tileSize) {
    IndexT currentNumData = min(tileSize, data.getSize(0) - i);

    auto dataView = data.narrow(0, i, currentNumData);
    auto codebookView = codebook.narrow(0, i, currentNumData);

    auto tileProjectionBufView =
        tileProjectionBufs[currentStream]->narrow(0, 0, currentNumData);

    // Do projection
    matrixMultiply(tileProjectionBufView, false, dataView, false, projMatrix, false,
                   1.0f, 0.0f, resources->getBlasHandleCurrentDevice(),
                   streams[currentStream]);

    // Perform binarization
    binarize(tileProjectionBufView, codebookView, streams[currentStream]);

    // Switch stream
    currentStream = (currentStream + 1) % 2;
  }

  // TODO: maybe we should transpose codebook
}

void generateProjectionMatrix(DeviceResources* resources,
                              Tensor<float, 2, int>& matrix, float mean,
                              float stddev, unsigned long long seed) {
  generateProjectionMatrix<float, int>(resources, matrix, mean, stddev, seed);
}

void generateProjectionMatrix(DeviceResources* resources,
                              Tensor<float, 2, int>& matrix,
                              unsigned long long seed) {
  generateProjectionMatrix(resources, matrix, 0.f, 1.f, seed);
}

void buildCodebook(DeviceResources* resources, const Tensor<float, 2, int>& data,
                   Tensor<float, 2, int>& projMatrix,
                   Tensor<unsigned, 2, int> codebook) {
  buildCodebook<float, int, unsigned>(resources, data, projMatrix, codebook);
}

}  // namespace curplsh
