#pragma once

#include "tensor.h"

namespace curplsh {

void computeL2Norm(const Tensor<float, 2>& vectors, Tensor<float, 1>& norms,
                   bool squared, cudaStream_t stream = 0);

}   // namespace curplsh
