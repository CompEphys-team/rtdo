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

#endif // CUDA_HELPER_H
