#pragma once

#include "device_resources.h"
#include "device_tensor.h"

namespace curplsh {

void searchL2Distance(DeviceResources* resources, const Tensor<float, 2>& bases,
                      Tensor<float, 1>* basesNorm, const Tensor<float, 2>& queries,
                      int k, Tensor<int, 2>& indices, Tensor<float, 2>& distances,
                      bool needRealDistances);

}   // namespace curplsh
