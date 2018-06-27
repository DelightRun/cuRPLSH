#include "internal/algorithms.h"

#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include "internal/cuda_utils.h"

namespace curplsh {

template <typename T, bool sort1, bool sort2>
T intersect(T* start1, T* end1, T* start2, T* end2, T* out) {
  int device1 = getDeviceForAddress(start1);
  int device2 = getDeviceForAddress(start2);
  host_assert(device1 == device2);

  T* end = nullptr;
  if (device1 == -1) {
    if (sort1) thrust::sort(thrust::host, start1, end1);
    if (sort2) thrust::sort(thrust::host, start2, end2);
    end = thrust::set_intersection(thrust::host, start1, end1, start2, end2, out);
  } else {
    if (sort1) thrust::sort(thrust::device, start1, end1);
    if (sort2) thrust::sort(thrust::device, start2, end2);
    end = thrust::set_intersection(thrust::device, start1, end1, start2, end2, out);
  }
  host_assert(end != nullptr);
  return end - out;
}

template <typename T>
T intersect(T* start1, T* end1, T* start2, T* end2, T* out, bool sort1, bool sort2) {
  if (sort1) {
    if (sort2) {
      return intersect<T, true, true>(start1, end1, start2, end2, out);
    } else {
      return intersect<T, true, false>(start1, end1, start2, end2, out);
    }
  } else {
    if (sort2) {
      return intersect<T, false, true>(start1, end1, start2, end2, out);
    } else {
      return intersect<T, false, false>(start1, end1, start2, end2, out);
    }
  }
}

int intersect(int* start1, int* end1, int* start2, int* end2, int* out, bool sort1,
              bool sort2) {
  return intersect<int>(start1, end1, start2, end2, out, sort1, sort2);
}

unsigned intersect(unsigned* start1, unsigned* end1, unsigned* start2,
                   unsigned* end2, unsigned* out, bool sort1, bool sort2) {
  return intersect<unsigned>(start1, end1, start2, end2, out, sort1, sort2);
}

long intersect(long* start1, long* end1, long* start2, long* end2, long* out,
               bool sort1, bool sort2) {
  return intersect<long>(start1, end1, start2, end2, out, sort1, sort2);
}

size_t intersect(size_t* start1, size_t* end1, size_t* start2, size_t* end2,
                 size_t* out, bool sort1, bool sort2) {
  return intersect<size_t>(start1, end1, start2, end2, out, sort1, sort2);
}

}  // namespace curplsh
