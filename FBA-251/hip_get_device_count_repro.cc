#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>

int main(int /*argc*/, char** /*argv*/) {
  hipError_t rv;
  int count;

  rv = hipGetDeviceCount(&count);
  if (rv != hipSuccess) {
    fprintf(stderr, "hipGetDeviceCount: %s\n", hipGetErrorString(rv));
    exit(1);
  } else {
    fprintf(stderr, "hipGetDeviceCount returned count=%d\n", count);
    fflush(stderr);
  }

  return 0;
}
