#ifndef EXPERIMENT_CU
#define EXPERIMENT_CU

#include "universallibrary.h"
#include "cuda_helper.h"

static scalar *target = nullptr, *d_target = nullptr;
static __device__ scalar *dd_target = nullptr;
static unsigned int target_size = 0, latest_target_size = 0;

static scalar *timeseries = nullptr, *d_timeseries = nullptr;
static __device__ scalar *dd_timeseries = nullptr;
static unsigned int timeseries_size = 0, latest_timeseries_size = 0;

void libInit(UniversalLibrary::Pointers &pointers, size_t numModels)
{
    pointers.pushV = [numModels](void *hostptr, void *devptr, size_t size){
        CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostptr, numModels * size, cudaMemcpyHostToDevice))
    };
    pointers.pullV = [numModels](void *hostptr, void *devptr, size_t size){
        CHECK_CUDA_ERRORS(cudaMemcpy(hostptr, devptr, numModels * size, cudaMemcpyDeviceToHost));
    };

    pointers.target =& target;
    pointers.output =& timeseries;

    allocateMem();
    initialize();
}

extern "C" void libExit(UniversalLibrary::Pointers &pointers)
{
    freeMem();
    pointers.pushV = pointers.pullV = nullptr;
}

extern "C" void resetDevice()
{
    cudaDeviceReset();
}

extern "C" void resizeTarget(size_t newSize)
{
    unsigned int tmp = target_size;
    resizeHostArray(target, tmp, newSize);
    resizeArray(d_target, target_size, newSize);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_target, &d_target, sizeof(scalar*)));
    latest_target_size = newSize;
}

extern "C" void pushTarget()
{
    CHECK_CUDA_ERRORS(cudaMemcpy(d_target, target, latest_target_size * sizeof(scalar), cudaMemcpyHostToDevice))
}

extern "C" void resizeOutput(size_t newSize)
{
    unsigned int tmp = target_size;
    resizeHostArray(timeseries, tmp, newSize);
    resizeArray(d_timeseries, timeseries_size, newSize);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_timeseries, &d_timeseries, sizeof(scalar*)));
    latest_timeseries_size = newSize;
}

extern "C" void pullOutput()
{
    CHECK_CUDA_ERRORS(cudaMemcpy(timeseries, d_timeseries, latest_timeseries_size * sizeof(scalar), cudaMemcpyDeviceToHost))
}

#endif
