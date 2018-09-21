#pragma once

#include "tensor.h"

namespace curplsh {

/// Blockwise K-Selection without given indices/labels.
/// Use sequential indices as labels implicitly.
void blockSelect(const Tensor<float, 2>& inK,  //
                 Tensor<float, 2>& outK,       //
                 Tensor<int, 2>& outV,         //
                 int k,                        // K
                 bool selectMax,               // select largest/smallest value
                 cudaStream_t stream);

/// Blockwise K-Selection with given indices/labels.
/// Use the given indices/labels to produce results.
void blockSelect(const Tensor<float, 2>& inK,  // input values
                 const Tensor<int, 2>& inV,    // input indices
                 Tensor<float, 2>& outK,       // output values
                 Tensor<int, 2>& outV,         // output indices
                 int k,                        // K
                 bool selectMax,               // select largest/smallest value
                 cudaStream_t stream);

/// Blockwise Radix Selection without given indices/labels.
/// Use sequential indices as labels implicitly.
void radixSelect(const Tensor<unsigned, 2>& inK,  //
                 Tensor<unsigned, 2>& outK,       //
                 Tensor<int, 2>& outV,            //
                 int k,                           // K
                 bool selectMax,                  // select largest/smallest value
                 cudaStream_t stream);

/// Blockwise Radix Selection with given indices/labels.
void radixSelect(const Tensor<unsigned, 2>& inK,  //
                 const Tensor<int, 2>& inV,       //
                 Tensor<unsigned, 2>& outK,       //
                 Tensor<int, 2>& outV,            //
                 int k,                           // K
                 bool selectMax,                  // select largest/smallest value
                 cudaStream_t stream);

/// Special K-Selection implementation for L2 distance
/// This implementation combine partial distance computation and K-Selection
/// into one kernel.
void l2Select(const Tensor<float, 2>& productDistances,  // inner products
              const Tensor<float, 1>& baseNorms,         // norms of bases
              Tensor<float, 2>& distances,               // result - distances
              Tensor<int, 2>& indices,                   // result - indices
              int k,                                     // K
              cudaStream_t stream);

}  // namespace curplsh
