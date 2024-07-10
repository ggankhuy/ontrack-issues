template <int32_t kWorldSize>
__global__ void one_shot_all_reduce(
    int32_t rank,
    int32_t world_size,
    int32_t flag,
    std::array<int32_t*, 8> barriers,
    std::array<at::BFloat16*, 8> inputs,
    at::BFloat16* acc,
    at::BFloat16* output,
    int32_t N) {
  // Synchronize the ranks.
  volatile int32_t* barrier_d = barriers[rank];
  if (threadIdx.x < kWorldSize) {
    // The 1st block notifies the other ranks.
    if (blockIdx.x == 0) {
#if defined(USE_ROCM)
      __atomic_store_n(barriers[threadIdx.x] + rank, flag, __ATOMIC_RELEASE);
#else
      barriers[threadIdx.x][rank] = flag;
#endif
    }

    // Busy-wait until all ranks are ready.
#if defined(USE_ROCM)
    while (__atomic_load_n(barrier_d + threadIdx.x, __ATOMIC_ACQUIRE) != flag) {
    }
#else
    while (barrier_d[threadIdx.x] != flag) {
    }
#endif
  }

  // Make sure we can move on...
  __syncthreads();
  // The source pointers. Distributed round-robin for the different warps.
  const at::BFloat16* src_d[kWorldSize];
#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int src_rank = (rank + ii) % kWorldSize;
    src_d[ii] = inputs[src_rank];
  }

  // Each block accumulates the values from the different GPUs on the same
  // node.
  for (size_t i = blockDim.x * blockIdx.x * 8 + threadIdx.x * 8; i < N;
       i += blockDim.x * gridDim.x * 8) {
    // Iterate over the different ranks/devices on the node to load the
    // values.
    bf16x8 vals[kWorldSize];
#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      *reinterpret_cast<uint4*>(&vals[ii]) =
          reinterpret_cast<const uint4*>(&src_d[ii][i])[0];
    }

    // Sum the values from the different ranks.
    bf16x8 sums;
    if (acc) {
      *reinterpret_cast<uint4*>(&sums) =
          *reinterpret_cast<const uint4*>(&acc[i]);
    } else {
      memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));
    }

#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }

    // Store to the destination buffer.
    *reinterpret_cast<uint4*>(&output[i]) =
        *reinterpret_cast<const uint4*>(&sums);
  }

  // barrier to sync with all other ranks on the same blockIdx
  // this is needed to ensure this-rank won't override its inputs buffer
  // (as we always do memcpy from srcbuff to inputs buffer first)
  // while other ranks are still reading them.
  __syncthreads();

  if (threadIdx.x < kWorldSize) {
    // notify all other blocks this blockIdx is ready
    const int32_t flag_block_offset = kWorldSize + blockIdx.x * kWorldSize;

#if defined(USE_ROCM)
    __atomic_store_n(
        barriers[threadIdx.x] + flag_block_offset + rank,
        flag,
        __ATOMIC_RELEASE);
#else
    barriers[threadIdx.x][flag_block_offset + rank] = flag;
#endif

    // busy-wait until all ranks are ready
#if defined(USE_ROCM)
    while (__atomic_load_n(
               barrier_d + flag_block_offset + threadIdx.x, __ATOMIC_ACQUIRE) !=
           flag) {
    }
#else
    while (barrier_d[flag_block_offset + threadIdx.x] != flag) {
    }
#endif
  }
}
