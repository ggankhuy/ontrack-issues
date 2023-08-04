#include <memory> include <type_traits> include <vector> include <iostream> include <limits> include <cmath> include 
#<algorithm> include <assert.h> include <stdlib.h> include <chrono> include <cstdio>
using namespace std;
#define HIP_LONG int32_t define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N) \
  HIP_LONG id = blockDim.x * blockIdx.x + threadIdx.x; \
  if (id >= N) \
    return; // The code below is based on section 4 Unsigned division of paper https://gmplib.org/~tege/divcnst-pldi94.pdf 
// In current ORT, fast_divmod is used for calculating the position of a element in tensor, // so unsigned integer division 
from the paper is good enough for ORT. The advantage is that div is very simple, // then GPU compiler can do loop unroll 
easilly when divmod is called in a loop. struct fast_divmod {
  fast_divmod(int d = 1) {
    d_ = d == 0 ? 1 : d;
    // ORT_ENFORCE(d_ >= 1 && d_ <= static_cast<uint32_t>(std::numeric_limits<int>::max()));
    for (l_ = 0; l_ < 32; l_++)
      if ((1U << l_) >= d_) break;
    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
    M_ = static_cast<uint32_t>(m);
    // according to paper, the value of m' should fit in a unsigned integer.
    // ORT_ENFORCE(M_ > 0 && M_ == m);
  }
  __host__ __device__ inline int div(int n) const {
#ifdef __CUDA_ARCH__
    uint32_t t = __umulhi(M_, n);
    return (t + n) >> l_;
#else
    // Using uint64_t for t, then t + n won't overflow.
    uint64_t t = ((uint64_t)M_ * n) >> 32;
    return static_cast<int>((t + n) >> l_);
#endif
  }
  __host__ __device__ inline int mod(int n) const {
    return n - div(n) * d_;
  }
  __host__ __device__ inline void divmod(int n, int& q, int& r) const {
    q = div(n);
    r = n - q * d_;
  }
  uint32_t d_; // divisor
  uint32_t M_; // m' in the paper.
  uint32_t l_; // l_ = ceil(log2(d_))
};
/*
  This is a utility wrapper for arbitrary type array
  Commonly used for passing small list of metadata during rocm kernel launch
  It's better to pass the array by value than having another cuMemcpy to pass the data to device. */ template <typename T, 
int32_t capacity = 8> struct TArray {
  TArray() : size_(0), data_() {
  }
  TArray(int32_t size) : size_(size), data_() {
    // ORT_ENFORCE(
   // 0 <= size && size <= capacity,
   // "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
  }
  TArray(const std::vector<T>& vec) : TArray(static_cast<int32_t>(vec.size())) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    memcpy(data_, vec.data(), vec.size() * sizeof(T));
  }
  void SetSize(int32_t size) {
    // ORT_ENFORCE(
    // 0 <= size && size <= capacity,
     // "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
    size_ = size;
  }
  __host__ __device__ int32_t Size() const {
    return size_;
  }
  __host__ __device__ T& operator[](int32_t index) {
    return data_[index];
  }
  __host__ __device__ __forceinline__ const T& operator[](int32_t index) const {
    return data_[index];
  }
  __host__ __device__ T* Data() {
    return data_;
  }
  __host__ __device__ const T* Data() const {
    return data_;
  }
  static constexpr int32_t Capacity() { return capacity; }; private:
  int32_t size_;
  T data_[capacity];
};
  
template <typename T> __global__ void TransposeKernel(int32_t shape_rank, const TArray<int64_t> input_strides,
                                const T* input_data, const TArray<fast_divmod> output_strides, T* output_data, HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  HIP_LONG input_index = 0;
  HIP_LONG output_index = id;
  #pragma unroll
  for (auto dim = 0; dim < input_strides.Capacity(); ++dim) {
    if (dim >= shape_rank) {
      break;
    }
    int out_coord, r;
    output_strides[dim].divmod(output_index, out_coord, r);
    output_index = r;
    input_index += input_strides[dim] * out_coord;
  }
  output_data[id] = input_data[input_index];
}
#define HIP_ASSERT(x) (assert((x)==cudaSuccess)) define NUM size_t(4)*1024*1024*1024
typedef std::chrono::high_resolution_clock Clock; decltype(auto) duration(std::chrono::time_point<Clock> start, 
std::chrono::time_point<Clock> end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
int main(int argc, char **argv) {
  int16_t* hostA;
  int16_t* hostB;
  int16_t* deviceA;
  int16_t* deviceB;
  size_t i;
  hostA = (int16_t*)malloc(NUM * sizeof(int16_t));
  hostB = (int16_t*)malloc(NUM * sizeof(int16_t));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (int16_t)i;
  }
  
  HIP_ASSERT(cudaMalloc((void**)&deviceA, NUM * sizeof(int16_t)));
  HIP_ASSERT(cudaMalloc((void**)&deviceB, NUM * sizeof(int16_t)));
  
  HIP_ASSERT(cudaMemcpy(deviceB, hostB, NUM*sizeof(int16_t), cudaMemcpyHostToDevice));
  // do transpose
  int N = 491520;
  std::vector<int> shape {15,512,64};
  int shape_rank = 3;
  int maxThreadsPerBlock = 256;
  int blocksPerGrid = (int)(std::ceil(static_cast<float>(N) / maxThreadsPerBlock));
  
  TArray<long> input_strides(3);
  input_strides[0] = 64;
  input_strides[1] = 32768;
  input_strides[2] = 1;
  TArray<fast_divmod> fdm_output_strides(3);
  fdm_output_strides[0] = 960;
  fdm_output_strides[1] = 64;
  fdm_output_strides[2] = 1;
  const int invocations = 100000;
  size_t ntile = std::floor(NUM/N) - 1;
  size_t step_b = 0;
  size_t step_a = 0;
  cudaStreamSynchronize(0);
  auto s1 = Clock::now();
  
  for (int i = 0; i < invocations; i++) {
    TransposeKernel<int16_t><<<dim3(blocksPerGrid), dim3(maxThreadsPerBlock)>>>(
          shape_rank, input_strides,
          deviceB + step_b,
          fdm_output_strides,
          deviceA + step_a,
          N);
    step_b = (rand() % ntile + 1)*size_t(N);
    step_a = (rand() % ntile + 1)*size_t(N);
  }
  auto m1 = Clock::now();
  cudaStreamSynchronize(0);
  auto e1 = Clock::now();
 
  std::printf("%.3f us per Transpose for %d calls (sync wait of %.3f ms)\n",
    double(duration(s1, e1))/1000.0/invocations, invocations, double(duration(m1, e1))/1000000.0);
  	
  HIP_ASSERT(cudaMemcpy(hostA, deviceA, NUM*sizeof(int16_t), cudaMemcpyDeviceToHost));
  HIP_ASSERT(cudaMemcpy(hostB, deviceB, NUM*sizeof(int16_t), cudaMemcpyDeviceToHost));
  HIP_ASSERT(cudaFree(deviceA));
  HIP_ASSERT(cudaFree(deviceB));
 
  bool bad = false;
  for (int i = 0; i < shape[0]; i++)
  for (int j = 0; j < shape[1]; j++)
  for (int k = 0; k < shape[2]; k++) {
          if ((hostB)[i*shape[1]*shape[2] + j*shape[2] + k] != (hostA)[ + j*shape[0]*shape[2] + i*shape[2] + k])
                  bad = true;
  
  }
  printf("success = %d\n", !bad);
  
  free(hostA);
  free(hostB);
}
