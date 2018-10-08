#pragma once

#include "tensor.h"

namespace curplsh {

void binarize(const Tensor<float, 2>& projections, Tensor<unsigned, 2>& codes,
              cudaStream_t stream = 0);

void computeHammingDistance(const Tensor<unsigned, 2>& queriesCode,
                            const Tensor<unsigned, 2> basesCode,
                            Tensor<unsigned, 2>& distances, cudaStream_t stream = 0);

}  // namespace curplsh
