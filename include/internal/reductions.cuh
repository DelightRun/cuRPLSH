#pragma once

#include "constants.h"
#include "kernel_utils.cuh"
#include "reduction_ops.h"

namespace curplsh {

template <typename T, template <typename U> class Op, int ReduceWidth = kWarpSize>
__device__ inline T warpReduce(T val, Op<T> op) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val = op(val, shfl_xor(val, mask));
  }

  return val;
}

template <typename T, template <typename U> class Op, bool KillWARDependency>
__device__ inline T blockReduce(T val, Op<T> op, T* smem) {
  int laneId = getLaneId();
  int warpId = threadIdx.x / kWarpSize;

  val = warpReduce<T, Op>(val, op);
  if (laneId == 0) {
    smem[warpId] = val;
  }

  __syncthreads();

  if (warpId == 0) {
    val = laneId < divUp(blockDim.x, warpSize) ? smem[laneId] : op.identity();
    val = warpReduce<T, Op>(val, op);

    // TODO if (BroadcastAll) {}
  }

  // TODO if (BroadcastAll) {}

  if (KillWARDependency) {
    __syncthreads();
  }

  return val;
}

template <typename T, int Num, template <typename U> class Op,
          bool KillWARDependency>
__device__ inline void blockReduce(T val[Num], Op<T> op, T* smem) {
  int laneId = getLaneId();
  int warpId = threadIdx.x / kWarpSize;

#pragma unroll
  for (int i = 0; i < Num; ++i) {
    warpReduce<T, Op>(val[i], op);
  }

  if (laneId == 0) {
#pragma unroll
    for (int i = 0; i < Num; ++i) {
      smem[warpId * Num + i] = val[i];
    }
  }

  __syncthreads();

  if (warpId == 0) {
#pragma unroll
    for (int i = 0; i < Num; ++i) {
      val[i] = laneId < divUp(blockDim.x, kWarpSize) ? smem[laneId * Num + i]
                                                     : op.identity();
      warpReduce<T, Op>(val[i], op);
    }

    // TODO if (BroadcastAll) {}
  }

  // TODO if (BroadcastAll) {}

  if (KillWARDependency) {
    __syncthreads();
  }
}

template <typename T, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceSum(T val) {
  return warpReduce<T, Sum, ReduceWidth>(val, Sum<T>());
}

template <typename T, bool KillWARDependency>
__device__ inline T blockReduceSum(T val, T* smem) {
  return blockReduce<T, Sum, KillWARDependency>(val, Sum<T>(), smem);
}

template <typename T, int Num, bool KillWARDependency>
__device__ inline void blockReduceSum(T val[Num], T* smem) {
  blockReduce<T, Num, Sum, KillWARDependency>(val, Sum<T>(), smem);
}
}  // namespace curplsh
