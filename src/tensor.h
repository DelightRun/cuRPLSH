#pragma once

#include <cstddef>
#include <initializer_list>

#include <cuda_runtime.h>

#include "assertions.h"
#include "cuda_utils.h"
#include "constants.h"
#include "traits.h"

namespace curplsh {

template <typename TensorType, int SubDim, template <typename U> class PtrTraits>
class TensorSlice;

template <typename T, int Dim, typename IndexT = int,
          template <typename U> class PtrTraits = DefaultPtrTraits>
class Tensor {
 public:
  typedef T DataType;
  typedef IndexT IndexType;
  typedef typename PtrTraits<T>::PtrType DataPtrType;
  typedef Tensor<T, Dim, IndexT, PtrTraits> TensorType;

  /// Default Constructor
  __host__ __device__ Tensor() : data_(nullptr) {
    static_assert(Dim > 0, "must have at least 1 dimension");

    for (int i = 0; i < Dim; ++i) {
      sizes_[i] = 0;
      strides_[i] = (IndexType)1;
    }
  }

  /// Copy Constructor
  __host__ __device__ Tensor(const Tensor<T, Dim, IndexT, PtrTraits>& other) {
    this->operator=(other);
  }

  /// Move Constructor
  __host__ __device__ Tensor(Tensor<T, Dim, IndexT, PtrTraits>&& other) {
    this->operator=(std::move(other));
  }

  /// Copy Assignment
  __host__ __device__ Tensor<T, Dim, IndexT, PtrTraits>& operator=(
      const Tensor<T, Dim, IndexT, PtrTraits>& other) {
    data_ = other.data_;
    for (int i = 0; i < Dim; ++i) {
      sizes_[i] = other.sizes_[i];
      strides_[i] = other.strides_[i];
    }

    return *this;
  }

  /// Move Assignment
  __host__ __device__ Tensor<T, Dim, IndexT, PtrTraits>& operator=(
      Tensor<T, Dim, IndexT, PtrTraits>&& other) {
    data_ = other.data_;
    other.data_ = nullptr;
    for (int i = 0; i < Dim; ++i) {
      sizes_[i] = other.sizes_[i];
      other.sizes_[i] = 0;
      strides_[i] = other.strides_[i];
      other.strides_[i] = 0;
    }

    return *this;
  }

  /// Contructs from given data and size with no padding
  __host__ __device__ Tensor(DataPtrType data, const IndexType sizes[Dim])
      : data_(data) {
    static_assert(Dim > 0, "must have at least 1 dimension");

    for (int i = 0; i < Dim; ++i) {
      sizes_[i] = sizes[i];
    }

    strides_[Dim - 1] = (IndexType)1;
    for (int i = Dim - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * sizes[i + 1];
    }
  }

  /// Contructs from given data and size with no padding
  __host__ __device__ Tensor(DataPtrType data,
                             std::initializer_list<IndexType> sizes)
      : data_(data) {
    static_assert(Dim > 0, "must have at least 1 dimension");

    int i = 0;
    for (auto size : sizes) {
      sizes_[i++] = size;
    }

    strides_[Dim - 1] = (IndexType)1;
    for (int i = Dim - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * sizes_[i + 1];
    }
  }

  /// Contructs from given data and size & strides
  __host__ __device__ Tensor(DataPtrType data, const IndexType sizes[Dim],
                             const IndexType strides[Dim])
      : data_(data) {
    static_assert(Dim > 0, "must have at least 1 dimension");

    for (int i = 0; i < Dim; ++i) {
      sizes_[i] = sizes[i];
      strides_[i] = strides[i];
    }
  }

  /// Copy from src to dst, static version
  __host__ static inline void copy(Tensor<T, Dim, IndexT, PtrTraits>& src,
                                   Tensor<T, Dim, IndexT, PtrTraits>& dst) {
    runtime_assert(src.isContiguous());
    runtime_assert(dst.isContiguous());

    runtime_assert(src.getNumElements() == dst.getNumElements());

    if (dst.getNumElements() <= 0) return;

    runtime_assert(src.data());
    runtime_assert(dst.data());

    int srcDev = getDeviceForAddress(src.data());
    int dstDev = getDeviceForAddress(dst.data());

    cudaMemcpyKind kind;
    if (srcDev == -1) {
      kind = dstDev == -1 ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice;
    } else {
      kind = srcDev == -1 ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
    }
    checkCudaErrors(cudaMemcpy(src.data(), dst.data(), src.getDataMemSize(), kind));
  }

  /// Copy data from another tensor, sizes must match
  __host__ inline void copyFrom(Tensor<T, Dim, IndexT, PtrTraits>& src) {
    copy(src, *this);
  }

  /// Copy data to another tensor, sizes must match
  __host__ inline void copyTo(Tensor<T, Dim, IndexT, PtrTraits>& dst) {
    copy(*this, dst);
  }

  /// Copy data to the specified address
  __host__ inline void copyTo(T* dst) {
    T* src = this->data();
    if (src == dst) {
      return;
    }

    int srcDev = getDeviceForAddress(src);
    int dstDev = getDeviceForAddress(dst);

    cudaMemcpyKind kind;
    if (srcDev == -1) {
      kind = dstDev == -1 ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice;
    } else {
      kind = srcDev == -1 ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
    }
    checkCudaErrors(cudaMemcpy(src, dst, this->getDataMemSize(), kind));
  }

  /// Returns true if another tensor has exactly same dimensions, sizes and
  /// strides.
  template <typename OtherT, int OtherDim>
  __host__ __device__ bool isSame(
      const Tensor<OtherT, OtherDim, IndexT, PtrTraits>& other) const {
    if (Dim != OtherDim) {
      return false;
    }

    for (int i = 0; i < Dim; ++i) {
      if (this->getSize(i) != other.getSize(i)) {
        return false;
      }
      if (this->getStride(i) != other.getStride(i)) {
        return false;
      }
    }

    return true;
  }

  /// Returns true if another tensor has same dimensions and sizes
  template <typename OtherT, int OtherDim>
  __host__ __device__ bool isSameSizes(
      const Tensor<OtherT, OtherDim, IndexT, PtrTraits>& other) const {
    if (Dim != OtherDim) {
      return false;
    }

    for (int i = 0; i < Dim; ++i) {
      if (this->getSize(i) != other.getSize(i)) {
        return false;
      }
    }

    return true;
  }

  /// Returns the size array
  __host__ __device__ inline const IndexType* sizes() const { return sizes_; }
  /// Returns the stride array
  __host__ __device__ inline const IndexType* strides() const { return strides_; }

  /// Returns the size of the given dimension(`[0, Dim - 1]`)
  __host__ __device__ inline IndexType getSize(int i) const {
    return i >= 0 ? sizes_[i] : sizes_[Dim + i];
  }
  /// Returns the stride of the given dimension(`[0, Dim - 1]`)
  __host__ __device__ inline IndexType getStride(int i) const {
    return i >= 0 ? strides_[i] : strides_[Dim + i];
  }

  /// Returns the total number of data's elements
  __host__ __device__ size_t getNumElements() const {
    size_t size = (size_t)getSize(0);

    for (int i = 1; i < Dim; ++i) {
      size *= (size_t)getSize(i);
    }

    return size;
  }

  /// Returns the total size (in bytes) of data (assuming no padding / is
  /// contiguous)
  __host__ __device__ size_t getDataMemSize() const {
    return getNumElements() * sizeof(T);
  }

  /// operator[], return a slice to access data
  __host__ __device__ inline TensorSlice<TensorType, Dim - 1, PtrTraits> operator[](
      IndexType index) {
    return TensorSlice<TensorType, Dim - 1, PtrTraits>(
        TensorSlice<TensorType, Dim, PtrTraits>(*this)[index]);
  }

  /// operator[], return a slice to access data, const version
  __host__ __device__ inline const TensorSlice<TensorType, Dim - 1, PtrTraits>
  operator[](IndexType index) const {
    return TensorSlice<TensorType, Dim - 1, PtrTraits>(
        TensorSlice<TensorType, Dim, PtrTraits>(
            const_cast<TensorType&>(*this))[index]);
  }

  /// Returns a raw pointer to the start of data
  __host__ __device__ inline DataPtrType data() { return data_; }
  /// Returns a raw pointer to the start of const data
  __host__ __device__ inline const DataPtrType data() const { return data_; }

  /// Returns a raw pointer to the end of data (assuming no padding)
  __host__ __device__ inline DataPtrType end() { return data() + getNumElements(); }
  /// Returns a raw pointer to the end of const data (assuming no padding)
  __host__ __device__ inline const DataPtrType end() const {
    return data() + getNumElements();
  }

  /// Returns true if we can cast this tensor to the new type
  template <typename U>
  __host__ __device__ bool isCastable() const {
    static_assert(sizeof(U) >= sizeof(T), "new type must has greater size.");

    if (sizeof(U) != sizeof(T)) {
      constexpr int kMultiple = sizeof(U) / sizeof(T);

      if ((sizeof(U) % sizeof(T)) != 0) return false;

      if ((reinterpret_cast<uintptr_t>(data_) % sizeof(U)) != 0) return false;

      if (sizes_[Dim - 1] % kMultiple != 0) return false;

      if (strides_[Dim - 1] != 1) return false;

      for (int i = 0; i < Dim - 1; ++i) {
        if (strides_[i] % kMultiple != 0) return false;
      }
    }

    return true;
  }

  /// Cast to a tensor of a different type which is potentially a
  /// different size than our type T. Tensor must be aligned and the
  /// innermost dimension must be a size that is a multiple of
  /// sizeof(U) / sizeof(T), and the stride of the innermost dimension
  /// must be contiguous. The stride of all outer dimensions must be a
  /// multiple of sizeof(U) / sizeof(T) as well.
  template <typename U>
  __host__ __device__ Tensor<U, Dim, IndexT, PtrTraits> cast() {
    runtime_assert(isCastable<U>());

    constexpr int kMultiple = sizeof(U) / sizeof(T);

    if (sizeof(U) == sizeof(T)) {
      return Tensor<U, Dim, IndexT, PtrTraits>(
          reinterpret_cast<typename PtrTraits<U>::PtrType>(data_), sizes_, strides_);
    } else {
      IndexType newSizes[Dim];
      IndexType newStrides[Dim];

      for (int i = 0; i < Dim - 1; ++i) {
        newSizes[i] = sizes_[i];
        newStrides[i] = strides_[i] / kMultiple;
      }

      newSizes[Dim - 1] = sizes_[Dim - 1] / kMultiple;
      newStrides[Dim - 1] = 1;

      return Tensor<U, Dim, IndexT, PtrTraits>(
          reinterpret_cast<typename PtrTraits<U>::PtrType>(data_), newSizes,
          newStrides);
    }
  }

  /// cast(), const version
  template <typename U>
  __host__ __device__ const Tensor<U, Dim, IndexT, PtrTraits> cast() const {
    return const_cast<Tensor<T, Dim, IndexT, PtrTraits>*>(this)->cast<U>();
  }

  /// Cast data to another type
  template <typename U>
  __host__ __device__ inline typename PtrTraits<U>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<U>::PtrType>(data_);
  }
  /// Cast data to another type, const version
  template <typename U>
  __host__ __device__ inline const typename PtrTraits<U>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<U>::PtrType>(data_);
  }

  /// Returns true if the entire data is contiguous (no padding and re-ordering
  /// of dimensions)
  __host__ __device__ bool isContiguous() const {
    /*
    size_t prevSize = 1;

    for (int i = Dim - 1; i >= 0; --i) {
      if (getSize(i) != (IndexType)1) {
        continue;
      }
      if (getStride(i) != prevSize) {
        return false;
      }
      prevSize *= getSize(i);
    }
    */

    // TODO currently always contiguous
    return true;
  }
  /// Return true if data is contiguous (no padding) in the given dimension
  __host__ __device__ bool isContiguous(int i) const {
    /*
    return (i == Dim - 1) ||
           ((i < Dim - 1) && ((getStride(i) / getStride(i + 1)) == getSize(i + 1)));
     */

    // TODO currently always contiguous
    return true;
  }

  // TODO what's this?
  // __host__ __device__ bool isConsistentlySized() const;
  // __host__ __device__ bool isConsistentlySized(int i) const;

  // TODO is transpose necessary?
  // __host__ __device__ TensorType transpose(int dim1, int dim2) const;

  /// Returns a tensor of the same dimension that is a view of the
  /// original tensor with the specified dimension restricted to the
  /// elements in the range [start, start + size).
  __host__ __device__ Tensor<T, Dim, IndexT, PtrTraits> narrow(int dim, IndexT start,
                                                               IndexT size) const {
    DataPtrType data = data_;

    runtime_assert(start >= 0 && start < sizes_[dim] &&
                   (start + size) <= sizes_[dim]);

    if (start > 0) {
      data += (size_t)start * strides_[dim];
    }

    IndexT sizes[Dim];
    for (int i = 0; i < Dim; ++i) {
      if (i == dim) {
        runtime_assert(start + size <= sizes_[dim]);
        sizes[i] = size;
      } else {
        sizes[i] = sizes_[i];
      }
    }

    return Tensor<T, Dim, IndexT, PtrTraits>(data, sizes, strides_);
  }

  template <int NewDim>
  __host__ __device__ Tensor<T, NewDim, IndexT, PtrTraits> view(
      std::initializer_list<IndexT> sizes) {
    runtime_assert(this->isContiguous());
    runtime_assert(sizes.size() == NewDim);

    size_t numElems = 1;
    for (auto s : sizes) {
      numElems *= s;
    }

    runtime_assert(numElems == getNumElements());
    return Tensor<T, NewDim, IndexT, PtrTraits>(data_, sizes);
  }

 protected:
  DataPtrType data_;
  IndexType sizes_[Dim];
  IndexType strides_[Dim];
};

/// We use a lightwight class to represent a slice of tensor instead of using
/// the
/// original tensor type because is too heavy.
template <typename TensorType, int SubDim, template <typename U> class PtrTraits>
class TensorSlice {
 public:
  typedef typename TensorType::DataType DataType;
  typedef typename TensorType::IndexType IndexType;
  typedef typename TensorType::DataPtrType DataPtrType;

  /// Constructor
  __host__ __device__ TensorSlice(TensorType& tensor)
      : tensor_(tensor), data_(tensor.data()) {}

  /// Constructor
  __host__ __device__ TensorSlice(TensorType& tensor, DataPtrType data)
      : tensor_(tensor), data_(data) {}

  /// operator[]
  __host__ __device__ inline TensorSlice<TensorType, SubDim - 1, PtrTraits>
  operator[](IndexType index) {
    return TensorSlice<TensorType, SubDim - 1, PtrTraits>(
        tensor_, data_ + index * tensor_.getStride(-SubDim));
  }

  /// operator[], const version
  __host__ __device__ inline const TensorSlice<TensorType, SubDim - 1, PtrTraits>
  operator[](IndexType index) const {
    return TensorSlice<TensorType, SubDim - 1, PtrTraits>(
        tensor_, data_ + index * tensor_.getStride(-SubDim));
  }

  /// operator&, return T*
  __host__ __device__ DataType* operator&() { return data_; }

  /// operator&, return const T*
  __host__ __device__ const DataType* operator&() const { return data_; }

  /// Return raw pointer to our data slice
  __host__ __device__ inline DataPtrType data() { return data_; }

  /// Return raw pointer to our data slice, const version
  __host__ __device__ inline const DataPtrType data() const { return data_; }

  /// View data pointer as another type pointer
  template <typename U>
  __host__ __device__ inline typename PtrTraits<U>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<U>::PtrType>(data_);
  }

  /// View data pointer as another type pointer, const version
  template <typename U>
  __host__ __device__ inline typename PtrTraits<const U>::PtrType dataAs() const {
    return reinterpret_cast<typename PtrTraits<const U>::PtrType>(data_);
  }

  /// Use the texture cache for reads
  __device__ inline DataType ldg() const {
#if __CUDA_ARCH__ >= 350
    return __ldg(data_);
#else
    return *data_;
#endif
  }

  /// Use the texture cache for reads & cast to another type
  template <typename U>
  __device__ inline U ldgAs() const {
#if __CUDA_ARCH__ >= 350
    return __ldg(dataAs<U>());
#else
    return *dataAs<U>();
#endif
  }

  /* TODO implement if necessary
  Tensor< DataType, SubDim, typename IndexType,
  PtrTraits>
    asTensor() {
      return tensor_.template view<SubDim>(data_);
    }
  */
 protected:
  // TODO friend if necessary

  TensorType& tensor_;
  DataPtrType data_;
};

/// Specialization for scalar (dim == 0)
template <typename TensorType, template <typename U> class PtrTraits>
class TensorSlice<TensorType, 0, PtrTraits> {
 public:
  typedef typename TensorType::DataType DataType;
  typedef typename TensorType::DataPtrType DataPtrType;

  /// Constructor
  __host__ __device__ TensorSlice(TensorType& tensor)
      : tensor_(tensor), data_(tensor.data()) {}

  /// Constructor
  __host__ __device__ TensorSlice(TensorType& tensor, DataPtrType data)
      : tensor_(tensor), data_(data) {}

  /// operator=, set data value
  __host__ __device__ TensorSlice<TensorType, 0, PtrTraits> operator=(DataType val) {
    *data_ = val;
    return *this;
  }

  /// operator T&, return T& (T reference)
  __host__ __device__ operator DataType&() { return *data_; }

  /// operator const T&, return const T& (T const reference)
  __host__ __device__ operator const DataType&() const { return *data_; }

  /// operator&, return T*
  __host__ __device__ DataType* operator&() { return data_; }

  /// operator&, return const T*
  __host__ __device__ const DataType* operator&() const { return data_; }

  __host__ __device__ inline DataPtrType data() { return data_; }

  __host__ __device__ inline const DataPtrType data() const { return data_; }

  /// View data pointer as another type pointer
  template <typename U>
  __host__ __device__ inline typename PtrTraits<U>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<U>::PtrType>(data_);
  }

  /// View data pointer as another type pointer, const version
  template <typename U>
  __host__ __device__ inline typename PtrTraits<const U>::PtrType dataAs() const {
    return reinterpret_cast<typename PtrTraits<const U>::PtrType>(data_);
  }

  /// Use the texture cache for reads
  __device__ inline DataType ldg() const {
#if __CUDA_ARCH__ >= 350
    return __ldg(data_);
#else
    return *data_;
#endif
  }

  /// Use the texture cache for reads & cast to another type
  template <typename U>
  __device__ inline U ldgAs() const {
#if __CUDA_ARCH__ >= 350
    return __ldg(dataAs<U>());
#else
    return *dataAs<U>();
#endif
  }

 protected:
  TensorType& tensor_;
  DataPtrType data_;
};

}  // namespace curplsh
