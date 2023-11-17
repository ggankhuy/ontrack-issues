#include <cstdio>
#include <string>
#include <iostream>

#include <mpi.h>
#include <vector>
#if defined (__NVCC__)
#include <cuda_runtime.h>
#define hipError_t cudaError_t
#define hipIpcGetMemHandle cudaIpcGetMemHandle
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipIpcMemHandle_t cudaIpcMemHandle_t
#define hipError_t cudaError_t
#define hipIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#define hipIpcOpenMemHandle cudaIpcOpenMemHandle
#define hipPointerAttribute_t cudaPointerAttributes
#define hipPointerGetAttributes cudaPointerGetAttributes
#define hipSetDevice cudaSetDevice
#define hipMalloc cudaMalloc
#else
#include <hip/hip_runtime.h>
#endif

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                                   \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cout << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  // Get rank information
  int rank;
  int numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  // Every rank allocates a memory buffer and opens an IPC handle to it
  std::vector<int*> buffers(numRanks);
  HIP_CALL(hipSetDevice(rank));
  size_t numBytes = (1 << 20);
  HIP_CALL(hipMalloc((void**)&buffers[rank], numBytes));
  for (int i = 0; i < numRanks; i++)
  {
    if (i == rank)
    {
      hipPointerAttribute_t attr;
      HIP_CALL(hipPointerGetAttributes(&attr, buffers[i]));
      printf("Rank %d: Allocated %p [On GPU %d]\n", i, buffers[i], attr.device);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Every rank gets an IPC handle for the memory
  std::vector<hipIpcMemHandle_t> handles(numRanks);
  HIP_CALL(hipIpcGetMemHandle(&handles[rank], buffers[rank]));

  for (int i = 0; i < numRanks; i++)
  {
    if (i == rank)
    {
      printf("Broadcasting from GPU %d\n", i);
      printf("=============================\n");
    }

    // Broadcast the handle to other ranks
    MPI_Bcast(&handles[i], sizeof(hipIpcMemHandle_t), MPI_BYTE, i, MPI_COMM_WORLD);

    for (int j = 0; j < numRanks; j++)
    {
      if (j == rank && j != i)
      {
        // Open IPC handle
        int* newPtr = buffers[i];
        if (i != rank)
          HIP_CALL(hipIpcOpenMemHandle((void**)&newPtr, handles[i], hipIpcMemLazyEnablePeerAccess));

        hipPointerAttribute_t attr;
        HIP_CALL(hipPointerGetAttributes(&attr, newPtr));
        printf("Rank %d: Recieved  %p [On GPU %d] %s\n", j, newPtr, attr.device, attr.device == j ? "PASS" : "FAIL");
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
  return 0;
}
