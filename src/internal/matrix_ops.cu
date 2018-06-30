#include "internal/matrix_ops.h"

#include "internal/assertions.h"
#include "internal/cuda_utils.h"

namespace curplsh {

template <typename T>
void matrixMultiply(Tensor<T, 2>& c, bool transC,        // Matrix C
                    const Tensor<T, 2>& a, bool transA,  // Matrix A
                    const Tensor<T, 2>& b, bool transB,  // Matrix B
                    float alpha, float beta,             // Coefficient alpha & beta
                    cublasHandle_t handle, cudaStream_t stream) {
  cublasSetStream(handle, stream);

  // Check that we have (m x k) * (k x n) = (m x n)
  int mA = a.getSize(transA ? 1 : 0);
  int kA = a.getSize(transA ? 0 : 1);

  int kB = b.getSize(transB ? 1 : 0);
  int nB = b.getSize(transB ? 0 : 1);

  int mC = c.getSize(transC ? 1 : 0);
  int nC = c.getSize(transC ? 0 : 1);

  host_assert(mA == mC);
  host_assert(kA == kB);
  host_assert(nB == nC);

  host_assert(a.getStride(1) == 1);
  host_assert(b.getStride(1) == 1);
  host_assert(c.getStride(1) == 1);

  // Now, we have to transpose the matrix into column-major layout
  T* pA = transC ? a.data() : b.data();
  T* pB = transC ? b.data() : a.data();
  T* pC = c.data();

  int m = c.getSize(1);  // stride 1 size
  int n = c.getSize(0);
  int k = transA ? a.getSize(0) : a.getSize(1);

  int lda = transC ? a.getStride(0) : b.getStride(0);
  int ldb = transC ? b.getStride(0) : a.getStride(0);
  int ldc = c.getStride(0);

  cublasOperation_t gemmTrA, gemmTrB;
  if (transC) {
    gemmTrA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;
    gemmTrB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
  } else {
    gemmTrA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    gemmTrB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

  checkCudaErrors(cublasSgemm(handle, gemmTrA, gemmTrB,  //
                              m, n, k, &alpha,           //
                              pA, lda, pB, ldb, &beta, pC, ldc));
  getLastCudaError("Check CUDA error.");
}

void matrixMultiply(Tensor<float, 2>& c, bool transC,        // Matrix C
                    const Tensor<float, 2>& a, bool transA,  // Matrix A
                    const Tensor<float, 2>& b, bool transB,  // Matrix B
                    float alpha, float beta,  // Coefficient alpha & beta
                    cublasHandle_t handle, cudaStream_t stream) {
  return matrixMultiply<float>(c, transC, a, transA, b, transB,  //
                               alpha, beta, handle, stream);
}

void matrixRandomInit(Tensor<float, 2>& matrix, float mean, float stddev,
                      curandGenerator_t generator, unsigned long long seed) {
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(generator, seed));
  checkCudaErrors(curandGenerateNormal(generator, matrix.data(),
                                       matrix.getNumElements(), mean, stddev));
}

}  // namespace curplsh
