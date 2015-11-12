#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
    #include "gpgpu.h"
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static lsampl_t *device_buffer;
static int n, blocks, threads_per_block;
static size_t buffer_size;

__global__ void mykernel(lsampl_t *arg, lsampl_t max, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;;
    if ( i < n )
        arg[i] = (max - arg[i]) % max;
}

void gpu_init(int n_samples) {
    n = n_samples;
    buffer_size = n * sizeof(lsampl_t);
    gpuErrchk( cudaMalloc(&device_buffer, buffer_size) );
    printf("Cuda: Allocated a buffer with %d samples (total size %u B)\n", n, buffer_size);
    threads_per_block = 256;
    blocks = (int)(n / threads_per_block) + (n % threads_per_block ? 1 : 0);
    printf("Cuda: Using %d blocks with %d threads each.\n", blocks, threads_per_block);
}

void gpu_exit() {
    gpuErrchk( cudaFree(device_buffer) );
}

void gpu_process(lsampl_t *in, lsampl_t *out, lsampl_t max) {
    gpuErrchk( cudaMemcpy(device_buffer, in, buffer_size, cudaMemcpyHostToDevice) );

    mykernel<<<blocks, threads_per_block>>>(device_buffer, max, n);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(out, device_buffer, buffer_size, cudaMemcpyDeviceToHost) );
}
