#pragma once

#include <cstddef>

namespace curplsh {

int intersect(int* start1, int* end1, int* start2, int* end2, int* out,
              bool sort1 = true, bool sort2 = true);
unsigned intersect(unsigned* start1, unsigned* end1, unsigned* start2,
                   unsigned* end2, unsigned* out, bool sort1 = true,
                   bool sort2 = true);
long intersect(long* start1, long* end1, long* start2, long* end2, long* out,
               bool sort1 = true, bool sort2 = true);
size_t intersect(size_t* start1, size_t* end1, size_t* start2, size_t* end2,
                 size_t* out, bool sort1 = true, bool sort2 = true);

}  // namespace curplsh
