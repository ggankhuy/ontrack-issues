/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 1024
#define ARRSIZE 3
#define LOOPSTRIDE 8
#define DEBUG 0

__global__ void slow(int *a, int*b, int *c) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 1000000000) {
        elapsed = clock64() - start;
    }

	int tid = hipBlockIdx_x * hipBlockIdx_x + hipThreadIdx_x;
	c[tid] = a[tid] + b[tid];
}

__global__ void slow2x(int *a, int*b, int *c) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 2000000000) {
        elapsed = clock64() - start;
    }

	int tid = hipBlockIdx_x * hipBlockIdx_x + hipThreadIdx_x;
	c[tid] = a[tid] + b[tid];
}

__global__ void fast(int *a, int*b, int *c) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
	int tid = hipBlockIdx_x * hipBlockIdx_x + hipThreadIdx_x;
	c[tid] = a[tid] + b[tid];
}

__global__ void fast2x(int *a, int*b, int *c) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 200000000) {
        elapsed = clock64() - start;
    }
	int tid = hipBlockIdx_x * hipBlockIdx_x + hipThreadIdx_x;
	c[tid] = a[tid] + b[tid];
}

int main (void) {
    int num_streams=4;
    int *a[num_streams], *b[num_streams], *c[num_streams];
    int *dev_a[num_streams], *dev_b[num_streams], *dev_c[num_streams];
    int i ;
    int printStride = 1;    

    printStride = N/8;
    for (int s = 0; s < num_streams; s++) {
        a[s] = (int*)malloc(N * sizeof(int));
        b[s] = (int*)malloc(N * sizeof(int));
        c[s] = (int*)malloc(N * sizeof(int));
 	    hipMalloc(&dev_a[s], N * sizeof(int) );
     	hipMalloc(&dev_b[s], N * sizeof(int) );
 	    hipMalloc(&dev_c[s], N * sizeof(int) );

    	for (int i = 0; i < N ; i ++ ) {
	    	a[s][i]  = i + num_streams * 1000 * s;
		    b[s][i] =  2 * i + num_streams * 2000 * s;
		    c[s][i] = 999;
	    }
    }

    for (int s = 0; s < num_streams; s++) {
        for (int j = 0; j < N; j+=printStride) {
            printf("stream %u, a[%u]: %u, b[%u]: %u.\n", s, j, a[s][j], j, b[s][j]);
        }
    }      

    hipStream_t streams[num_streams];

    for (int s = 0; s < num_streams; s++) {

        printf("==============================\n");
        //hipStream_t streams;
        //hipStreamCreateWithFlags(&streams, hipStreamNonBlocking);
        hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking);

        if (DEBUG == 1) {
            printf("h2d dev_a, a, src %x, dst: %x.\n", dev_a[s], a[s]);
            printf("h2d dev_b, b, src %x, dst: %x.\n", dev_b[s], b[s]);
            printf("h2d dev_c, c, src %x, dst: %x.\n", dev_c[s], c[s]);
        }

       	hipMemcpyAsync(dev_a[s], a[s], N * sizeof(int), hipMemcpyHostToDevice, streams[s]);
   	    hipMemcpyAsync(dev_b[s], b[s], N * sizeof(int), hipMemcpyHostToDevice, streams[s]);
       	hipMemcpyAsync(dev_c[s], c[s], N * sizeof(int), hipMemcpyHostToDevice, streams[s]);
    
        const unsigned threads = 1024;
        const unsigned blocks = N/threads + 1;
        if (printStride < 1)    
            printStride = 1;

        //hipLaunchKernelGGL(add, blocks, threads, 0, 0, dev_a, dev_b, dev_c);
        //add<<<blocks, threads, 0, streams[s]>>>(dev_a[s], dev_b[s], dev_c[s]);
        //add<<<blocks, threads, 0>>>(dev_a, dev_b, dev_c);

        switch(s) {
            case 0:
                printf("Launching slow stream: %u <<<%u, %u>>>\n", s, blocks, threads);
                slow<<<blocks, threads, 0, streams[s]>>>(dev_a[s], dev_b[s], dev_c[s]);
                break;
            case 1:
                printf("Launching slow2x stream %u <<<%u, %u>>>\n", s, blocks, threads);
                slow2x<<<blocks, threads, 0, streams[s]>>>(dev_a[s], dev_b[s], dev_c[s]);
                break;
            case 2:
                printf("Launching fast stream %u <<<%u, %u>>>\n", s, blocks, threads);
                fast<<<blocks, threads, 0, streams[s]>>>(dev_a[s], dev_b[s], dev_c[s]);
                break;
            case 3:
                printf("Launching fast2x stream %u <<<%u, %u>>>\n", s, blocks, threads);
                fast2x<<<blocks, threads, 0, streams[s]>>>(dev_a[s], dev_b[s], dev_c[s]);
                break;
        }
        if (DEBUG == 1) {
            printf("h2d a, dev_a, src %x, dst: %x.\n", a[s], dev_a[s]);
            printf("h2d b, dev_b, src %x, dst: %x.\n", b[s], dev_b[s]);
            printf("h2d c, dev_c, src %x, dst: %x.\n", c[s], dev_c[s]); 
        }

        hipMemcpyAsync(a[s], dev_a[s], N * sizeof(int), hipMemcpyDeviceToHost, streams[s]);
        hipMemcpyAsync(b[s], dev_b[s], N * sizeof(int), hipMemcpyDeviceToHost, streams[s]);
        hipMemcpyAsync(c[s], dev_c[s], N * sizeof(int), hipMemcpyDeviceToHost, streams[s]);

        hipDeviceSynchronize();

        //for (int s = 0; s < num_streams; s++) {
            for (int j = 0; j < N; j+=printStride) {
                printf("stream=%u. idx=%u: %u + %u = %u.\n", s, j, a[s][j], b[s][j], c[s][j]);
            }
        //}      
    }
        for (int s = 0; s < num_streams; s++) {
            hipStreamDestroy(streams[s]);
            hipFree(&dev_a[s]);
            hipFree(&dev_b[s]);
            hipFree(&dev_c[s]);
            
            free(a[s]);
            free(b[s]);
            free(c[s]);
        }
	return 0;
}
