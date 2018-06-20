#pragma once

#include "math_utils.h"
#include "traits.h"

namespace curplsh {

template <typename T>
struct Sum {
  __device__ inline T operator()(T a, T b) const { return a + b; }

  __device__ inline T identity() const { return NumericTraits<T>::zero(); }
};

template <typename T>
struct Min {
  __device__ inline T operator()(T a, T b) const { return a < b ? a : b; }

  __device__ inline T identity() const { return NumericTraits<T>::max(); }
};

template <typename T>
struct Max {
  __device__ inline T operator()(T a, T b) const { return a > b ? a : b; }

  __device__ inline T identity() const { return NumericTraits<T>::min(); }
};

}  // namespace curplsh
