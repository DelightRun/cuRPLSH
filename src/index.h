#pragma once

#include "constants.h"

namespace curplsh {

struct IndexConfig {};

class Index {
 public:
  Index(int dim, IndexConfig);

  virtual void add(idx_t num, const float* data) = 0;

  virtual void search(idx_t num, const float* queries, idx_t k, idx_t* indices,
                      float* distances) = 0;

  inline int getDim() const { return dim_; }

 private:
  int dim_;
};

struct IndexBruteForceConfig : public IndexConfig {};

class IndexBruteForce : public Index {
 public:
  IndexBruteForce(int dim, IndexBruteForceConfig);

  inline idx_t getNum() const { return num_; }

 private:
  float* data_;
  float* norms_;

  idx_t num_;
};
}
