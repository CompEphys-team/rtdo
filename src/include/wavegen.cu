#ifndef WAVEGEN_CU
#define WAVEGEN_CU

#include "wavegenconstructor.h"
#include "cuda_helper.h" // For syntax highlighting only

static __device__ WaveStats *dd_wavestats;
static __device__ Stimulation *dd_waveforms;

void allocateGroupMem(WavegenConstructor::Pointers &pointers)
{
    cudaHostAlloc(&pointers.wavestats, MM_NumGroups * sizeof(WaveStats), cudaHostAllocPortable);
        deviceMemAllocate(&pointers.d_wavestats, dd_wavestats, MM_NumGroups * sizeof(WaveStats));
    cudaHostAlloc(&pointers.waveforms, MM_NumGroups * sizeof(Stimulation), cudaHostAllocPortable);
        deviceMemAllocate(&pointers.d_waveforms, dd_waveforms, MM_NumGroups * sizeof(Stimulation));

    cudaHostAlloc(&pointers.clear_wavestats, MM_NumGroups * sizeof(WaveStats), cudaHostAllocPortable);
    for ( unsigned i = 0; i < MM_NumGroups; i++ )
        pointers.clear_wavestats[i] = {};
}

void freeGroupMem(WavegenConstructor::Pointers &pointers)
{
    cudaFreeHost(pointers.wavestats);
    cudaFreeHost(pointers.clear_wavestats);
    CHECK_CUDA_ERRORS(cudaFree(pointers.d_wavestats));
    cudaFreeHost(pointers.waveforms);
    CHECK_CUDA_ERRORS(cudaFree(pointers.d_waveforms));
}

void libInit(WavegenConstructor::Pointers &pointers, size_t numGroups, size_t numModels)
{
    pointers.clearStats = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_wavestats, pointers.clear_wavestats, numGroups * sizeof(WaveStats), cudaMemcpyHostToDevice))
    };
    pointers.pullStats = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.wavestats, pointers.d_wavestats, numGroups * sizeof(WaveStats), cudaMemcpyDeviceToHost))
    };
    pointers.pushWaveforms = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_waveforms, pointers.waveforms, numGroups * sizeof(Stimulation), cudaMemcpyHostToDevice))
    };
    pointers.pullWaveforms = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.waveforms, pointers.d_waveforms, numGroups * sizeof(Stimulation), cudaMemcpyDeviceToHost))
    };
    pointers.pushErr = [&pointers, numModels](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_err, pointers.err, numModels * sizeof(scalar), cudaMemcpyHostToDevice))
    };
    pointers.pullErr = [&pointers, numModels](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.err, pointers.d_err, numModels * sizeof(scalar), cudaMemcpyDeviceToHost))
    };

    allocateMem();
    allocateGroupMem(pointers);
    initialize();
    pointers.clearStats();
}

extern "C" void libExit(WavegenConstructor::Pointers &pointers)
{
    freeMem();
    freeGroupMem(pointers);
    pointers.clearStats = pointers.pullStats = pointers.pushWaveforms = pointers.pullWaveforms = pointers.pushErr = pointers.pullErr = nullptr;
}

extern "C" void resetDevice()
{
    cudaDeviceReset();
}

#endif
