/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


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

#define CURAND_CALL(call)                                       \
{                                                               \
    curandStatus_t status = (call);                             \
    if (status != CURAND_STATUS_SUCCESS) {                      \
        std::cerr << __FILE__ << ": " <<  __LINE__;             \
        std::cerr << ": curand error " << status << std::endl;  \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

#define CUSOLVER_CALL(call)                                      \
{                                                                \
    cusolverStatus_t status = (call);                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                     \
        std::cerr << __FILE__ << ": " <<  __LINE__;              \
        std::cerr << ": cusolver error " << status << std::endl; \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
}

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

template <typename T>
void resizeArrayPair(T *&h_arr, T *&d_arr, unsigned int &actualSize, unsigned int requestedSize)
{
    if ( actualSize < requestedSize ) {
        CHECK_CUDA_ERRORS(cudaFree(d_arr));
        CHECK_CUDA_ERRORS(cudaFreeHost(h_arr));
        CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_arr, requestedSize * sizeof(T)));
        CHECK_CUDA_ERRORS(cudaHostAlloc((void **)&h_arr, requestedSize * sizeof(T), cudaHostAllocPortable));
        actualSize = requestedSize;
    }
}

#endif // CUDA_HELPER_H
