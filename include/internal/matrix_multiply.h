#pragma once

#include <cublas_v2.h>

#include "tensor.h"

namespace curplsh {

/// C = alpha * A + beta * C
/// Expects row major layout
void matrixMultiply(Tensor<float, 2>& c, bool transC,  // Matrix C
                    Tensor<float, 2>& a, bool transA,  // Matrix A
                    Tensor<float, 2>& b, bool transB,  // Matrix B
                    float alpha, float beta,           // Coefficient alpha & beta
                    cublasHandle_t handle, cudaStream_t stream);

}
