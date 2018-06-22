#pragma once

#include "traits.h"

namespace curplsh {

template <typename K, typename V>
struct Pair {
  __device__ constexpr inline Pair() {}
  __device__ constexpr inline Pair(K key, V value) : k(key), v(value) {}

  __device__ inline bool operator==(const Pair<K, V>& rhs) const {
    return k == rhs.k && v == rhs.v;
  }
  __device__ inline bool operator!=(const Pair<K, V>& rhs) const {
    return !this->operator==(rhs);
  }

  __device__ inline bool operator<(const Pair<K, V>& rhs) const {
    return (k < rhs.k) || ((k == rhs.k) && (v < rhs.v));
  }

  __device__ inline bool operator>(const Pair<K, V>& rhs) const {
    return (k > rhs.k) || ((k == rhs.k) && (v > rhs.v));
  }

  K k;
  V v;
};

// specialization for shfl_up & shfl_xor
template <typename K, typename V>
__device__ inline Pair<K, V> shfl_up(Pair<K, V>& pair, int delta,
                                     int width = kWarpSize) {
  return Pair<K, V>(shfl_up(pair.k, delta, width), shfl_up(pair.v, delta, width));
}

template <typename K, typename V>
__device__ inline Pair<K, V> shfl_xor(const Pair<K, V>& pair, int mask,
                                      int width = kWarpSize) {
  return Pair<K, V>(shfl_xor(pair.k, mask, width), shfl_xor(pair.v, mask, width));
}

// specialization for NumericTraits
template <typename K, typename V>
struct NumericTraits<Pair<K, V>> {
  __host__ __device__ static inline Pair<K, V> min() {
    return Pair<K, V>(NumericTraits<K>::min(), NumericTraits<V>::min());
  }

  __host__ __device__ static inline Pair<K, V> max() {
    return Pair<K, V>(NumericTraits<K>::max(), NumericTraits<V>::max());
  }
};

}  // namespace curplsh
