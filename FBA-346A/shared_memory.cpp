#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"


#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

__global__ void bit_extract_kernel(int my_size, uint32_t* A) {
  const auto bt_start = blockIdx.x * blockDim.y + threadIdx.y;
  const auto stride = gridDim.x * blockDim.y;
  const int t_x = threadIdx.x;
  const int t_y = threadIdx.y;
  const int b_x = blockIdx.x;

  extern __shared__ uint64_t smem[];
  uint64_t* temp = &smem[my_size * threadIdx.y];

  __syncthreads();
  // __threadfence_block();
  for (auto i = threadIdx.x; i < my_size; i += blockDim.x) {
    temp[i] = 0;
  }
  /*
  if (threadIdx.x == 0) {
    temp[0] = 0;
  }
  */
  // __threadfence_block();
  __syncthreads();

    /*
  if (temp[0] != 0) {
    printf("ERROR: %lu, %d, %d, %d\n", temp[0], t_x, t_y, b_x);
  }
  */
  if (temp[0] != 0) {
    __assert_fail("none zero", __FILE__, b_x * 10000 + t_y * 100 + t_x, __func__);
  }
  // this is just to trigger seg fault
  A[temp[0]]++;
}


int main(int argc, char* argv[]) {
    uint32_t *A;
    CHECK(hipMalloc(&A, 10));

    printf("info: launch 'bit_extract_kernel' \n");
    int my_size = 1;
    // bit_extract_kernel<<<dim3(3328), dim3(64, 16), my_size * 16 * sizeof(uint64_t)>>>(my_size);
    bit_extract_kernel<<<dim3(3328), dim3(64, 16), my_size * 16 * sizeof(uint64_t)>>>(my_size, A);

    CHECK(hipDeviceSynchronize());
}
