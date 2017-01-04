#ifndef EXPERIMENT_CU
#define EXPERIMENT_CU

#include "experimentlibrary.h"
#include "cuda_helper.h" // For syntax highlighting only

void libInit(ExperimentLibrary::Pointers &pointers, size_t numModels)
{
    pointers.pushErr = [&pointers, numModels](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_err, pointers.err, numModels * sizeof(scalar), cudaMemcpyHostToDevice))
    };
    pointers.pullErr = [&pointers, numModels](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.err, pointers.d_err, numModels * sizeof(scalar), cudaMemcpyDeviceToHost))
    };

    allocateMem();
    initialize();
}

extern "C" void libExit(ExperimentLibrary::Pointers &pointers)
{
    freeMem();
    pointers.pushErr = pointers.pullErr = nullptr;
}

extern "C" void resetDevice()
{
    cudaDeviceReset();
}

#endif
