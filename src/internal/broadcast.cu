#include "internal/broadcast.h"

#include <type_traits>

#include "internal/assertions.h"
#include "internal/constants.h"
#include "internal/math_utils.h"
#include "internal/kernel_utils.cuh"
#include "internal/reduction_ops.h"

namespace curplsh {

namespace {

template <typename T, typename TVec>
__global__ void kernelBroadcastAlongRows(Tensor<T, 1> input,
                                         Tensor<TVec, 2> output) {
  __shared__ T sval;

  int row = blockIdx.x;
  if (threadIdx.x == 0) {
    sval = input[row];
  }

  __syncthreads();

  T val = sval;

  for (int i = threadIdx.x; i < output.getSize(1); i += blockDim.x) {
    // FIXME: speed up use atomicAdd?
    /*
    TVec out = output[row][i];
    out += val;
    output[row][i] = out;
    */
    atomicAddT(&output[row][i], val);
  }
}
}

template <typename T, typename TVec>
void broadcastAlongRows(Tensor<T, 1>& input, Tensor<T, 2>& output,
                        cudaStream_t stream) {
  host_assert(input.getSize(0) == output.getSize(0));

  if (!std::is_same<T, TVec>::value && output.template isCastable<TVec>()) {
    auto outputV = output.template cast<TVec>();

    int numThreads = std::min(outputV.getSize(1), kMaxThreadsPerBlock);
    auto grid = dim3(input.getSize(0));
    auto block = dim3(numThreads);

    kernelBroadcastAlongRows<T, TVec><<<grid, block, 0, stream>>>(input, outputV);
  } else {
    int numThreads = std::min(output.getSize(1), kMaxThreadsPerBlock);
    auto grid = dim3(input.getSize(0));
    auto block = dim3(numThreads);

    kernelBroadcastAlongRows<T, T><<<grid, block, 0, stream>>>(input, output);
  }
}

void  broadcastAlongRowsSum(Tensor<float, 1>& input, Tensor<float, 2>& output, cudaStream_t stream) {
  auto dim = output.getSize(1);

  if (dim % (4 * kWarpSize) == 0) {
    broadcastAlongRows<float, float4>(input, output, stream);
  } else if (dim % (2 * kWarpSize) == 0) {
    broadcastAlongRows<float, float2>(input, output, stream);
  } else {
    broadcastAlongRows<float, float>(input, output, stream);
  }

}

}   // namespace curplsh
