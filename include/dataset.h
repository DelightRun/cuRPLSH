#pragma once

#include <fstream>
#include <string>

#include "memory_space.h"

#include "internal/algorithms.h"

namespace curplsh {

namespace helper {

template <typename T, typename IndexT>
T* loadXvecsData(const char* filename, IndexT& num, IndexT& dim,
                 MemorySpace space = MemorySpace::Host) {
  std::ifstream file(filename, std::ios::binary);
  host_assert(file.is_open());

  // Read dimension
  file.read((char*)&dim, sizeof(IndexT));

  // Get file size and compute data number
  file.seekg(0, std::ios::end);
  size_t filesize = (size_t)file.tellg();
  num = (IndexT)(filesize / (dim * sizeof(T) + sizeof(IndexT)));
  file.seekg(0, std::ios::beg);

  // Alloc memory and read
  size_t numElems = num * dim;
  T *data, *hostData;
  allocMemory(data, numElems, space);
  hostData = (space == MemorySpace::Device) ? new T[numElems] : data;
  host_assert(hostData);

  for (IndexT i = 0; i < num; ++i) {
    file.seekg(sizeof(IndexT), std::ios::cur);
    file.read((char*)(hostData + i * dim), dim * sizeof(T));
  }
  file.close();

  if (space == MemorySpace::Device) {
    copyMemory(data, hostData, numElems, cudaMemcpyHostToDevice);
    delete[] hostData;
  }
  return data;
}

// TODO: saveXvecsData
}  // namespace helper

template <typename T, typename IndexT>
class Dataset {
 public:
  typedef T DataType;
  typedef IndexT IndexType;

  Dataset()
      : Dataset(0, 0, 0, 0, nullptr, nullptr, nullptr, "", MemorySpace::Device) {}

  Dataset(IndexT dimension, IndexT gtK, IndexT numBase, IndexT numQuery, T* base,
          T* query, IndexT* gt, const char* name, MemorySpace space)
      : dimension_(dimension),
        gtK_(gtK),
        numBase_(numBase),
        numQuery_(numQuery),
        base_(base),
        query_(query),
        gt_(gt),
        name_(name),
        space_(space),
        isMemOwner(false) {}

  Dataset(const Dataset<T, IndexT>& other) { this->operator=(other); }

  Dataset(Dataset<T, IndexT>&& other) { this->operator=(std::move(other)); }

  Dataset<T, IndexT>& operator=(const Dataset<T, IndexT>& other) {
    isMemOwner = false;

    dimension_ = other.dimension_;
    gtK_ = other.gtK_;
    numBase_ = other.numBase_;
    numQuery_ = other.numQuery_;
    base_ = other.base_;
    query_ = other.query_;
    gt_ = other.gt_;
    name_ = other.name_;
    space_ = other.space_;

    return *this;
  }

  Dataset<T, IndexT>& operator=(Dataset<T, IndexT>&& other) {
    dimension_ = other.dimension_;
    gtK_ = other.gtK_;
    numBase_ = other.numBase_;
    numQuery_ = other.numQuery_;
    base_ = other.base_;
    query_ = other.query_;
    gt_ = other.gt_;
    isMemOwner = other.isMemOwner;
    name_ = other.name_;
    space_ = other.space_;

    other.isMemOwner = false;
    other.dimension_ = 0;
    other.numBase_ = 0;
    other.numQuery_ = 0;
    other.base_ = nullptr;
    other.query_ = nullptr;
    other.gt_ = nullptr;

    return *this;
  }

  virtual ~Dataset() {
    if (isMemOwner) {
      cudaFree(base_);
      cudaFree(query_);
      cudaFree(gt_);
    }
  }

  IndexT getDimension() const { return dimension_; }
  IndexT getGroundTruthK() const { return gtK_; }
  IndexT getNumBase() const { return numBase_; }
  IndexT getNumQuery() const { return numQuery_; }

  const T* getBase() const { return base_; }
  const T* getQuery() const { return query_; }
  const IndexT* getGroundTruth() const { return gt_; }

  std::string getName() const { return name_; }
  MemorySpace getMemorySpace() const { return space_; }

  virtual float evaluate(IndexT* indices, IndexT* diff = nullptr) const {
    IndexT* hindices = new IndexT[numQuery_ * gtK_];
    copyMemory(hindices, indices, numQuery_ * gtK_);

    float recall = 0.f;
    for (IndexT i = 0; i < numQuery_; ++i) {
      IndexT* result = hindices + i * gtK_;
      IndexT* groundtruth = gt_ + i * gtK_;
      int num = 0;
      for (int p = 0; p < gtK_; ++p)
        for (int q = 0; q < gtK_; ++q)
          if (result[p] == groundtruth[q]) ++num;
      recall += (num / (float)gtK_);
      if (diff != nullptr) diff[i] = gtK_ - num;
    }
    recall /= numQuery_;

    delete[] hindices;
    return recall;
  }

 protected:
  IndexT dimension_;
  IndexT gtK_;
  IndexT numBase_;
  IndexT numQuery_;

  T* base_;
  T* query_;
  IndexT* gt_;

  std::string name_;
  MemorySpace space_;
  bool isMemOwner;
};

template <typename IndexT>
class DatasetIrisa : public Dataset<float, IndexT> {
 public:
  DatasetIrisa(const char* name, const char* datadir,
               MemorySpace space = MemorySpace::Device) {
    this->name_ = name;
    this->space_ = space;

    std::string basedir(datadir);
    if (*basedir.rbegin() != '/') {
      basedir += '/';
    }

    std::string basefile(basedir + name + "_base.fvecs");
    this->base_ = helper::loadXvecsData<float, IndexT>(
        basefile.c_str(), this->numBase_, this->dimension_, space);
    std::string queryfile(basedir + name + "_query.fvecs");
    this->query_ = helper::loadXvecsData<float, IndexT>(
        queryfile.c_str(), this->numQuery_, this->dimension_, space);
    std::string gtfile(basedir + name + "_groundtruth.ivecs");
    this->gt_ = helper::loadXvecsData<IndexT, IndexT>(gtfile.c_str(),
                                                      this->numQuery_, this->gtK_);
  }
};

class DatasetSIFT : public DatasetIrisa<int> {
 public:
  DatasetSIFT(const char* basedir, MemorySpace space = MemorySpace::Device)
      : DatasetIrisa<int>("sift", basedir, space) {}
};

class DatasetGIST : public DatasetIrisa<int> {
 public:
  DatasetGIST(const char* basedir, MemorySpace space = MemorySpace::Device)
      : DatasetIrisa<int>("gist", basedir, space) {}
};

}  // namespace curplsh
