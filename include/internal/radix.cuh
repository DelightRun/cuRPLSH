#pragma once

#include "internal/assertions.h"
#include "internal/kernel_utils.cuh"
#include "internal/traits.h"

namespace curplsh {

namespace {

// Counts the distribution of `digitPos` radix of input values.
// Only those statisifing `((val & desiredMask) == desired)` will be counted.
// As you can see, the filter above checks if the given positions (indicated by
// `desiredMask`) is equal to the desired value (indicated by `desired`)
template <typename T, typename IndexT, int RadixBits, int RadixSize>
__device__ void countRadixDesired(T* data, IndexT num, unsigned int desired,
                                  unsigned int desiredMask, int digitPos,
                                  IndexT counts[RadixSize], IndexT* smem) {
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();

  for (IndexT i = threadIdx.x; i < num; i += blockDim.x) {
    T val = ldg(&data[i]);

    bool isDesired = ((val & desiredMask) == desired);
    unsigned int digit = getBitField(val, digitPos, RadixBits);

#pragma unroll
    for (unsigned int j = 0; j < RadixSize; ++j) {
      bool vote = isDesired && (digit == j);
      counts[j] += popcnt(ballot(vote));
    }
  }

  if (getLaneId() == 0) {
#pragma unroll
    for (int i = 0; i < RadixSize; ++i) {
      // FIXME: atomicAdd or reduce
      atomicAdd(&smem[i], counts[i]);
    }
  }

  __syncthreads();

#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }

  __syncthreads();
}

// Finds the unique value `v` that matches the pattern
// `((v & desiredMask) == desired)`
template <typename T, typename IndexT>
__device__ T findDesired(T* data, IndexT num, unsigned int desired,
                         unsigned int desiredMask, T* smem) {
  if (threadIdx.x < 32) {
    smem[threadIdx.x] = NumericTraits<T>::zero();
  }
  __syncthreads();

  IndexT numRoundedUp = roundUp(num, blockDim.x);
  for (IndexT i = threadIdx.x; i < numRoundedUp; i += blockDim.x) {
    bool isInRange = (i < num);
    T value = isInRange ? ldg(&data[i]) : NumericTraits<T>::zero();

    // FIXME: Maybe can use primitives instead of shared memory
    if (isInRange && ((value & desiredMask) == desired)) {
      smem[0] = NumericTraits<T>::identity();
      smem[1] = value;
    }

    __syncthreads();

    T found = smem[0];
    T val = smem[1];

    __syncthreads();

    if (found != NumericTraits<T>::zero()) {
      return val;
    }
  }

  // Not found, should not happend
  runtime_assert(false);
  return NumericTraits<T>::zero();
}

}  // namespace

// Select the k-th small/large data of the given data
template <typename T, typename IndexT, bool SelectMax, int RadixBits>
__device__ T radixSelectKthElement(T* data, IndexT num, IndexT k, int* smem) {
  static_assert(RadixBits > 0);

  constexpr unsigned int RadixSize = 1 << RadixBits;
  constexpr unsigned int RadixMask = RadixSize - 1;

  // Per-thread buckets into which we accumulate digit counts in our radix
  int counts[RadixSize];

  unsigned int desired = 0;
  unsigned int desiredMask = 0;

  int kToFind = k;

  // MSB Radix Sort
#pragma unroll
  for (int digitPos = sizeof(T) * 8 - RadixBits; digitPos >= 0;
       digitPos -= RadixBits) {
    countRadixDesired<T, IndexT, RadixBits, RadixSize>(
        data, num, desired, desiredMask, digitPos, counts, smem);

// TODO: Comments
#define CHECK_RADIX(i)                                                        \
  int count = counts[i];                                                      \
  if (count == 1 && kToFind == 1) {                                           \
    desired = setBitField(desired, i, digitPos, RadixBits);     \
    desiredMask = setBitField(desiredMask, RadixMask, digitPos, RadixBits);   \
    return findDesired<T, IndexT>(data, num, desired, desiredMask, (T*)smem); \
  }                                                                           \
  if (count >= kToFind) {                                                     \
    desired = setBitField(desired, i, digitPos, RadixBits);     \
    desiredMask = setBitField(desiredMask, RadixMask, digitPos, RadixBits);   \
    break;                                                                    \
  }                                                                           \
  kToFind -= count;

    if (SelectMax) {
#pragma unroll
      for (int i = RadixSize - 1; i >= 0; --i) {
        CHECK_RADIX(i);
      }
    } else {
#pragma unroll
      for (int i = 0; i < RadixSize; ++i) {
        CHECK_RADIX(i);
      }
    }
#undef CHECK_RADIX
  }

  // There is no unique but many result
  return desired;
}

}  // namespace curplsh
