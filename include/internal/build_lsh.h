#pragma once

#include "device_resources.h"
#include "device_tensor.h"

namespace curplsh {

void generateProjectionMatrix(DeviceResources* resources,
                              Tensor<float, 2, int>& matrix, float mean,
                              float stddev, unsigned long long seed);

void generateProjectionMatrix(DeviceResources* resources,
                              Tensor<float, 2, int>& matrix,
                              unsigned long long seed);

void buildCodebook(DeviceResources* resources, const Tensor<float, 2, int>& data,
                   Tensor<float, 2, int>& projMatrix,
                   Tensor<unsigned, 2, int> codebook);
}
