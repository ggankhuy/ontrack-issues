#include <thrust/sort.h>
#include "hip/hip_runtime.h"

void func(float* values, int64_t* indices) {
  thrust::sort_by_key(
      thrust::device, values, values, indices, thrust::greater<float>());
}
