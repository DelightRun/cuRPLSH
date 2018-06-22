#pragma once

#include "tensor.h"

namespace curplsh {

template <typename T, int Dim, typename IndexT = int,
          template <typename U> class PtrTraits = DefaultPtrTraits>
class HostTensor : public Tensor<T, Dim, IndexT, PtrTraits> {
 public:
  typedef T DataType;
  typedef IndexT IndexType;
  typedef typename PtrTraits<T>::PtrType DataPtrType;

  /// Default Constructor
  __host__ HostTensor()
      : Tensor<T, Dim, IndexT, PtrTraits>(), allocState_(AllocState::NotOwner) {}

  /// Destructor
  __host__ ~HostTensor() {
    if (allocState_ == AllocState::Owner) {
      runtime_assert(this->data_ != nullptr);
      delete[] this->data_;
      this->data_ = nullptr;
    }
  }

  /// Move Constructor
  __host__ HostTensor(HostTensor<T, Dim, IndexT, PtrTraits>&& other)
      : Tensor<T, Dim, IndexT, PtrTraits>(), allocState_(AllocState::NotOwner) {
    this->operator=(std::move(other));
  }

  /// Move Assignment
  __host__ HostTensor<T, Dim, IndexT, PtrTraits>& operator=(
      HostTensor<T, Dim, IndexT, PtrTraits>&& other) {
    if ((this->allocState_ == AllocState::Owner) && (this->data_ != nullptr)) {
      delete[] this->data_;
    }

    this->Tensor<T, Dim, IndexT, PtrTraits>::operator=(std::move(other));

    this->allocState_ = other.allocState;
    other.allocState_ = AllocState::NotOwner;

    return *this;
  }

  /// Constructs a tensor of given size, allocating memory locally
  __host__ HostTensor(const IndexType sizes[Dim])
      : Tensor<T, Dim, IndexT, PtrTraits>(nullptr, sizes),
        allocState_(AllocState::Owner) {
    this->data_ = new T[this->getNumElements()];
    runtime_assert(this->data_ != nullptr);
  }

  /// Constructs a tensor of given size, allocating memory locally
  __host__ HostTensor(std::initializer_list<IndexType> sizes)
      : Tensor<T, Dim, IndexT, PtrTraits>(nullptr, sizes),
        allocState_(AllocState::Owner) {
    this->data_ = new T[this->getNumElements()];
    runtime_assert(this->data_ != nullptr);
  }

  /// Constructs a tensor of given size, referencing the given memory
  __host__ HostTensor(DataPtrType data, const IndexType sizes[Dim])
      : Tensor<T, Dim, IndexT, PtrTraits>(data, sizes),
        allocState_(AllocState::NotOwner) {}

  /// Constructs a tensor of given size, referencing the given memory
  __host__ HostTensor(DataPtrType data, std::initializer_list<IndexType> sizes)
      : Tensor<T, Dim, IndexT, PtrTraits>(data, sizes),
        allocState_(AllocState::NotOwner) {}

  /// Constructs a tensor of given size and strides, referencing the given
  /// memory
  __host__ HostTensor(DataPtrType data, const IndexType sizes[Dim],
                      const IndexType strides[Dim])
      : Tensor<T, Dim, IndexT, PtrTraits>(data, sizes, strides),
        allocState_(AllocState::NotOwner) {}

  /// Deep Copy Constructor
  __host__ HostTensor(Tensor<T, Dim, IndexT, PtrTraits>& other, bool /*deep*/)
      : Tensor<T, Dim, IndexT, PtrTraits>(nullptr, other.sizes(), other.strides()),
        allocState_(AllocState::Owner) {
    runtime_assert(other.isContiguous());

    this->data_ = new T[this->getNumElements()];
    runtime_assert(this->data_ != nullptr);

    this->copyFrom(other);
  }

  /// Call to zero out memory
  __host__ HostTensor<T, Dim, IndexT, PtrTraits>& zero() {
    if (this->data_ != nullptr) {
      runtime_assert(this->isContiguous());
      memset(this->data_, 0, this->getDataMemSize());
    }

    return *this;
  }

 private:
  enum class AllocState {
    /// This tensor owns the memory, which must be freed manually
    Owner,
    /// This tensor is not the owner of the memory, so nothing to free
    NotOwner,
  };

  AllocState allocState_;
};
}
