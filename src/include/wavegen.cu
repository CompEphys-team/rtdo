#ifndef WAVEGEN_CU
#define WAVEGEN_CU

#include "wavegen_globals.h"
#include "cuda_helper.h" // For syntax highlighting only

__device__ WaveStats *dd_wavestats;
__device__ Stimulation *dd_waveforms;

void allocateGroupMem()
{
    using namespace Wavegen_Global;
    cudaHostAlloc(&wavestats, MM_NumGroups * sizeof(WaveStats), cudaHostAllocPortable);
        deviceMemAllocate(&d_wavestats, dd_wavestats, MM_NumGroups * sizeof(WaveStats));
    cudaHostAlloc(&waveforms, MM_NumGroups * sizeof(Stimulation), cudaHostAllocPortable);
        deviceMemAllocate(&d_waveforms, dd_waveforms, MM_NumGroups * sizeof(Stimulation));

    cudaHostAlloc(&clear_wavestats, MM_NumGroups * sizeof(WaveStats), cudaHostAllocPortable);
    for ( unsigned i = 0; i < MM_NumGroups; i++ )
        clear_wavestats[i] = {};
}
void clearStats()
{
    using namespace Wavegen_Global;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_wavestats, clear_wavestats, MM_NumGroups * sizeof(WaveStats), cudaMemcpyHostToDevice))
}
void pullStats()
{
    using namespace Wavegen_Global;
    CHECK_CUDA_ERRORS(cudaMemcpy(wavestats, d_wavestats, MM_NumGroups * sizeof(WaveStats), cudaMemcpyDeviceToHost))
}
void pushWaveforms()
{
    using namespace Wavegen_Global;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_waveforms, waveforms, MM_NumGroups * sizeof(Stimulation), cudaMemcpyHostToDevice))
}
void pullWaveforms()
{
    using namespace Wavegen_Global;
    CHECK_CUDA_ERRORS(cudaMemcpy(waveforms, d_waveforms, MM_NumGroups * sizeof(Stimulation), cudaMemcpyDeviceToHost))
}
void freeGroupMem()
{
    using namespace Wavegen_Global;
    cudaFreeHost(wavestats);
    cudaFreeHost(clear_wavestats);
    CHECK_CUDA_ERRORS(cudaFree(d_wavestats));
    cudaFreeHost(waveforms);
    CHECK_CUDA_ERRORS(cudaFree(d_waveforms));
}

void libManualInit(MetaModel &m) // Must be called separately (through Wavegen_Global::init())
{
    allocateMem();
    allocateGroupMem();
    initialize();
    clearStats();
    Wavegen_Global::populate(m);
}

void __attribute__ ((constructor)) libInit()
{
    Wavegen_Global::init =& libManualInit;
    Wavegen_Global::push =& pushHHStateToDevice;
    Wavegen_Global::pull =& pullHHStateFromDevice;
    Wavegen_Global::step =& stepTimeGPU;
    Wavegen_Global::reset =& initialize;
    Wavegen_Global::pullStats =& pullStats;
    Wavegen_Global::clearStats =& clearStats;
    Wavegen_Global::pushWaveforms =& pushWaveforms;
    Wavegen_Global::pullWaveforms =& pullWaveforms;

    Wavegen_Global::t =& t;
    Wavegen_Global::iT =& iT;
}

void libExit()
{
    freeMem();
    freeGroupMem();
    cudaDeviceReset();
    Wavegen_Global::init = 0;
    Wavegen_Global::push = 0;
    Wavegen_Global::pull = 0;
    Wavegen_Global::step = 0;
    Wavegen_Global::reset = 0;
    Wavegen_Global::pullStats = 0;
    Wavegen_Global::clearStats = 0;
    Wavegen_Global::pushWaveforms = 0;
    Wavegen_Global::pullWaveforms = 0;

    Wavegen_Global::t = 0;
    Wavegen_Global::iT = 0;
}

#endif
