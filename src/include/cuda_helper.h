#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <iostream>

#ifndef CHECK_CUDA_ERRORS
#define CHECK_CUDA_ERRORS(call)                                 \
{                                                               \
    cudaError_t error = (call);                                 \
    if (error != cudaSuccess) {                                 \
        std::cerr << __FILE__ << ": " <<  __LINE__;             \
        std::cerr << ": cuda runtime error " << error << ": ";  \
        std::cerr << cudaGetErrorString(error) << std::endl;    \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}
#endif

#define PULL(namesp, hostvar) \
    CHECK_CUDA_ERRORS(cudaMemcpy(hostvar, \
                                 d_ ## hostvar, \
                                 sizeof(*hostvar) * namesp::NPOP, \
                                 cudaMemcpyDeviceToHost))

#define PUSH(namesp, hostvar) \
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ ## hostvar, \
                                 hostvar, \
                                 sizeof(*hostvar) * namesp::NPOP, \
                                 cudaMemcpyHostToDevice))

template <typename T>
void resizeArray(T *&arr, unsigned int &actualSize, unsigned int requestedSize)
{
    if ( actualSize < requestedSize ) {
        CHECK_CUDA_ERRORS(cudaFree(arr));
        CHECK_CUDA_ERRORS(cudaMalloc(&arr, requestedSize * sizeof(T)));
        actualSize = requestedSize;
    }
}

template <typename T>
void resizeHostArray(T *&arr, unsigned int &actualSize, unsigned int requestedSize)
{
    if ( actualSize < requestedSize ) {
        CHECK_CUDA_ERRORS(cudaFreeHost(arr));
        CHECK_CUDA_ERRORS(cudaHostAlloc(&arr, requestedSize * sizeof(T), cudaHostAllocPortable));
        actualSize = requestedSize;
    }
}

#endif // CUDA_HELPER_H
