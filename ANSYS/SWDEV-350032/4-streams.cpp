// reproducer-2.cpp
// delta from original reproduce.cpp is attempt to ignore default stream 0.

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ void fast(int * d_a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
}

__global__ void fast2x(int * d_a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 200000000) {
        elapsed = clock64() - start;
    }
}

__global__ void slow(int * d_a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 1000000000) {
        elapsed = clock64() - start;
    }
}

__global__ void slow2x(int * d_a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 2000000000) {
        elapsed = clock64() - start;
    }
}

int main(int argc, char **argv) {
    const int num_iter = 2;
    const int num_streams = 4;
    int *h_a[num_streams];
    int *d_a[num_streams];
    size_t numbytes_a = 1000000;
    hipStream_t streams[num_streams];

    // Create two streams and allocate host and device memory for each.

    for (int i = 0; i < num_streams; i++) {
        //hipStreamCreate(&streams[i]);
        hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking);
        hipHostMalloc((void**)&h_a[i], numbytes_a);
        hipMalloc(&d_a[i], numbytes_a);
    }      
    // Expected behavior if fast_first = 1

    // iter from 0 to 4.

    bool fast_first = 1;
    for (int iter = 0; iter < num_streams * num_iter; ++iter) {

        // i = iterate through streams within each loop.

        int i = iter % num_streams;
    
        // make async copy for current stream.

        hipMemcpyAsync(d_a[i], h_a[i], numbytes_a, hipMemcpyHostToDevice, streams[i]);

        // i = 0, slow launches first or 3rd. i=1, fast launches second and 4th.

        printf("iter: %u.\n", iter);
        switch(i) {
            case 0:
                printf("Launching slow stream i: %u\n", i);
                slow<<<1, 256, 0, streams[i]>>>(d_a[i]);
                break;
            case 1:
                printf("Launching fast stream %u.\n", i);
                fast<<<1, 256, 0, streams[i]>>>(d_a[i]);
                break;
            case 2:
                printf("Launching slow2x stream %u.\n", i);
                slow2x<<<1, 256, 0, streams[i]>>>(d_a[i]);
                break;
            case 3:
                printf("Launching fast2x stream %u.\n", i);
                fast2x<<<1, 256, 0, streams[i]>>>(d_a[i]);
                break;
            default:
                printf("Bypassing stream %u.\n", i);
        }
        // copy back.
        
        hipMemcpyAsync(h_a[i], d_a[i], numbytes_a, hipMemcpyDeviceToHost, streams[i]);
    }

    hipDeviceSynchronize();

    for (int i = 0; i < num_streams; i++) {
        hipStreamDestroy(streams[i]);
        hipFree(d_a[i]);
    }
}
