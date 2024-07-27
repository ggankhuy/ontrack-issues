#include <stdio.h>

#include <rccl/rccl.h>

#include <hip/hip_runtime_api.h>



int main() {

  int nDev = 1;

  int devs[] = {0};

  ncclComm_t comm;



  hipError_t err = hipSetDevice(0);



  ncclCommInitAll(&comm, 1, devs);

  ncclCommDestroy(comm);

}
