#pragma once

namespace curplsh {

struct IndexConfig {};

class Index {
 public:
  Index(int dim, IndexConfig);

  virtual void add(int num, const float* data) = 0;

  virtual void search(int num, const float* queries, int k, int* indices,
                      float* distances) = 0;

  inline int getDim() const { return dim_; }

 private:
  int dim_;
};

}   // namespace curplsh
