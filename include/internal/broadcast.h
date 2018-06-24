#pragma once

#include "tensor.h"

namespace curplsh {

void broadcastAlongRowsSum(Tensor<float, 1>& input, Tensor<float, 2>& output,
                           cudaStream_t stream);

}  // namespace curplsh
