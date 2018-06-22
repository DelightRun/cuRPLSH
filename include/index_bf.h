#pragma once

#include "index.h"

namespace curplsh {

struct IndexBruteForceConfig : public IndexConfig {};

class IndexBruteForce : public Index {
 public:
  IndexBruteForce(int dim, IndexBruteForceConfig);

  inline int getNum() const { return num_; }

 private:
  float* data_;
  float* norms_;

  int num_;
};

}
