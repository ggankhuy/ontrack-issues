#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <chrono>
#include <unistd.h>
#define N 512 
#define NSTEP 1
#define NKERNEL 1000
#define CONSTANT 5.34

#define HIPCHECK(error)                                                                            \
    {                                                                                              \
        hipError_t localError = error;                                                             \
        if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {      \
            std::cout << "API returned error code." << std::endl;                                                    \
        }                                                                                          \
    }

__global__ void simpleKernel(float* out_d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out_d[idx] += 1;
}

bool hipTestWithGraph() {
  int deviceId;
  HIPCHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, deviceId));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  float *out_h;
  out_h = new float[N];
  float *out_d;
  HIPCHECK(hipMalloc(&out_d, N * sizeof(float)));

  auto start = std::chrono::high_resolution_clock::now();
  // start CPU wallclock timer
  bool graphCreated = false;
  hipGraph_t graph;
  hipGraphExec_t instance;

  hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
    simpleKernel<<<dim3(N / 512, 1, 1), dim3(N, 1, 1), 0, stream>>>(out_d);
  }
  hipStreamEndCapture(stream, &graph);
  hipGraphInstantiate(&instance, graph, NULL, NULL, 0);

  auto start1 = std::chrono::high_resolution_clock::now();
  for (int istep = 0; istep < NSTEP; istep++) {
    hipGraphLaunch(instance, stream);
    //hipStreamSynchronize(stream);
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto resultWithInit = std::chrono::duration<double, std::micro>(stop - start);
  auto resultWithoutInit = std::chrono::duration<double, std::micro>(stop - start1);
  std::cout << "Time taken for graph with Init: "
            << std::chrono::duration_cast<std::chrono::microseconds>(resultWithInit).count()
            << " microseconds without Init:"
            << std::chrono::duration_cast<std::chrono::microseconds>(resultWithoutInit).count()
            << " microseconds " << std::endl;

  HIPCHECK(hipMemcpy(out_h, out_d, N * sizeof(float), hipMemcpyDeviceToHost));
  delete[] out_h;
  HIPCHECK(hipFree(out_d));
  return true;
}

bool hipTestWithoutGraph() {
  int deviceId;
  HIPCHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, deviceId));
  printf("info: running on device #%d %s\n", deviceId, props.name);

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  float *out_h;
  out_h = new float[N];

  float *in_d, *out_d;
  HIPCHECK(hipMalloc(&out_d, N * sizeof(float)));

  // start CPU wallclock timer
  auto start = std::chrono::high_resolution_clock::now();
  for (int istep = 0; istep < NSTEP; istep++) {
    for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
      simpleKernel<<<dim3(N / 512, 1, 1), dim3(N, 1, 1), 0, stream>>>(out_d);
    }
    //HIPCHECK(hipStreamSynchronize(stream));
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto result = std::chrono::duration<double, std::micro>(stop - start);
  std::cout << "Time taken for test without graph: "
            << std::chrono::duration_cast<std::chrono::microseconds>(result).count()
            << " microsecs " << std::endl;
  HIPCHECK(hipMemcpy(out_h, out_d, N * sizeof(float), hipMemcpyDeviceToHost));
  delete[] out_h;
  HIPCHECK(hipFree(out_d));
  return true;
}

int main(int argc, char* argv[]) {
  bool status1, status2;
  //status1 = hipTestWithoutGraph()
  status2 = hipTestWithGraph();
}
