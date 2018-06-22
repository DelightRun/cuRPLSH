#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#define host_assert(X)                                              \
  do {                                                              \
    if (!(X)) {                                                     \
      fprintf(stderr, "Assertion '%s' failed in %s at %s:%d\n", #X, \
              __PRETTY_FUNCTION__, __FILE__, __LINE__);             \
      abort();                                                      \
    }                                                               \
  } while (false)

#ifdef __CUDA_ARCH__
#define runtime_assert(X) assert(X)
#else
#define runtime_assert(X) host_assert(X)
#endif
