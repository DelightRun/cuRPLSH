#pragma once

#include "device_resources.h"
#include "device_tensor.h"

namespace curplsh {

void searchHammingDistance(DeviceResources* resources,
                           const Tensor<float, 2, int>& bases,
                           const Tensor<float, 1, int>* basesNorm,
                           const Tensor<unsigned, 2, int>& basesCode,
                           const Tensor<float, 2, int>& projMatrix,
                           const Tensor<float, 2, int>& queries, int k,
                           Tensor<int, 2, int>& indices,
                           Tensor<float, 2, int>& distances, int numCandidates);

}  // namespace curplsh
