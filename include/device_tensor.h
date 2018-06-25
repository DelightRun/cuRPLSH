#pragma once

#include "memory_space.h"
#include "tensor.h"

namespace curplsh {

template <typename T, int Dim, typename IndexT = int,
          template <typename U> class PtrTraits = DefaultPtrTraits>
class DeviceTensor : public Tensor<T, Dim, IndexT, PtrTraits> {
 public:
  typedef T DataType;
  typedef IndexT IndexType;
  typedef typename PtrTraits<T>::PtrType DataPtrType;

  /// Destructor
  __host__ ~DeviceTensor() {
    if (allocState_ == AllocState::Owner) {
      runtime_assert(this->data_ || (this->getDataMemSize() == 0));
      checkCudaErrors(cudaFree(this->data_));
      this->data_ = nullptr;
    }
  }

  /// Default Constructor
  __host__ DeviceTensor()
      : Tensor<T, Dim, IndexT, PtrTraits>(),
        allocState_(AllocState::NotOwner),
        space_(MemorySpace::Device) {}

  /// Move Constructor
  __host__ DeviceTensor(DeviceTensor<T, Dim, IndexT, PtrTraits>&& other)
      : Tensor<T, Dim, IndexT, PtrTraits>(),
        allocState_(AllocState::NotOwner),
        space_(MemorySpace::Device) {
    this->operator=(std::move(other));
  }

  /// Move Assignment
  __host__ DeviceTensor<T, Dim, IndexT, PtrTraits>& operator=(
      DeviceTensor<T, Dim, IndexT, PtrTraits>&& other) {
    if (this->allocState_ == AllocState::Owner) {
      checkCudaErrors(cudaFree(this->data_));
    }

    this->Tensor<T, Dim, IndexT, PtrTraits>::operator=(std::move(other));

    this->allocState_ = other.allocState_;
    this->space_ = other.space_;
    other.allocState_ = AllocState::NotOwner;

    return *this;
  }

  /// Constructs a tensor of given size, allocating memory locally
  __host__ DeviceTensor(const IndexType sizes[Dim],
                        MemorySpace space = MemorySpace::Device)
      : Tensor<T, Dim, IndexT, PtrTraits>(nullptr, sizes),
        allocState_(AllocState::Owner),
        space_(space) {
    allocate_();
  }

  /// Constructs a tensor of given size, allocating memory locally
  __host__ DeviceTensor(std::initializer_list<IndexType> sizes,
                        MemorySpace space = MemorySpace::Device)
      : Tensor<T, Dim, IndexT, PtrTraits>(nullptr, sizes),
        allocState_(AllocState::Owner),
        space_(space) {
    allocate_();
  }

  /// Constructs a tensor of given size, referencing the given memory.
  /// If tryCopy is true, memory locate on host and we don't use unified memory,
  /// then we alloc memory and copy from given data
  __host__ DeviceTensor(DataPtrType data, const IndexType sizes[Dim],
                        bool tryCopy = true, cudaStream_t stream = 0)
      : Tensor<T, Dim, IndexT, PtrTraits>(data, sizes),
        allocState_(AllocState::NotOwner),
        space_{getMemorySpace(data)} {
    if (tryCopy) {
      ctorAllocCopy_(data, stream);
    }
  }

  /// Constructs a tensor of given size, referencing the given memory.
  /// If tryCopy is true, memory locate on host and we don't use unified memory,
  /// then we alloc memory and copy from given data
  __host__ DeviceTensor(DataPtrType data, std::initializer_list<IndexType> sizes,
                        bool tryCopy = true, cudaStream_t stream = 0)
      : Tensor<T, Dim, IndexT, PtrTraits>(data, sizes),
        allocState_(AllocState::NotOwner),
        space_(getMemorySpace(data)) {
    if (tryCopy) {
      ctorAllocCopy_(data, stream);
    }
  }

  /// Constructs a tensor of given size, referencing the given memory.
  /// If tryCopy is true, memory locate on host and we don't use unified memory,
  /// then we alloc memory and copy from given data
  __host__ DeviceTensor(DataPtrType data, const IndexType sizes[Dim],
                        const IndexType strides[Dim], bool tryCopy = true,
                        cudaStream_t stream = 0)
      : Tensor<T, Dim, IndexT, PtrTraits>(data, sizes, strides),
        allocState_(AllocState::NotOwner),
        space_(getMemorySpace(data)) {
    if (tryCopy) {
      ctorAllocCopy_(data, stream);
    }
  }

  /// Deep Copy Constructor
  __host__ DeviceTensor(Tensor<T, Dim, IndexT, PtrTraits>& other, MemorySpace space)
      : Tensor<T, Dim, IndexT, PtrTraits>(nullptr, other.sizes(), other.strides()),
        allocState_(AllocState::Owner),
        space_(space) {
    host_assert(other.isContiguous());
    allocate_();
    this->copyFrom(other);
  }

  /// Get device of current tensor data
  __host__ int device() const { return getDeviceForAddress(this->data()); }

  /// Fill tensor with 0
  __host__ DeviceTensor<T, Dim, IndexT, PtrTraits>& zero(cudaStream_t stream = 0) {
    if (this->data_ != nullptr) {
      host_assert(this->isContiguous());
      checkCudaErrors(
          cudaMemsetAsync(this->data_, 0, this->getDataMemSize(), stream));
    }

    return *this;
  }

 private:
  enum class AllocState {
    Owner,     // This tensor owns the memory, which must be freed manually
    NotOwner,  // This tensor is not the owner of the memory, so nothing to free
  };

  AllocState allocState_;
  MemorySpace space_;

  // WARNING Only can be used in ctor
  __host__ inline void allocate_() {
    allocMemory((void**)&this->data_, this->getDataMemSize(), space_);
    host_assert(this->data_ || (this->getDataMemSize() == 0));
    allocState_ = AllocState::Owner;
  }

  __host__ inline void ctorAllocCopy_(DataPtrType data, cudaStream_t stream) {
    if (space_ == MemorySpace::Device) {
      int currentDevice = getCurrentDevice();
      int srcDevice = getDeviceForAddress(data);

      if (srcDevice != currentDevice) {
        allocate_();
        checkCudaErrors(cudaMemcpyAsync(
            this->data_, data, this->getDataMemSize(),
            srcDevice == -1 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice,
            stream));
      }
    }
  }
};
}  // namespace curplsh
