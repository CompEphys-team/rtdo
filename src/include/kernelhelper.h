#ifndef KERNELHELPER_H
#define KERNELHELPER_H

#include "types.h"
#include "metamodel.h"

namespace GeNN_Bridge {

extern WaveStats *wavestats;
extern WaveStats *clear_wavestats;
extern WaveStats *d_wavestats;

extern Stimulation *waveforms;
extern Stimulation *d_waveforms;

extern void (*push)(void);
extern void (*pull)(void);
extern void (*step)(void);
extern void (*init)(MetaModel&);
extern void (*reset)(void);
extern void (*pullStats)(void);
extern void (*clearStats)(void);
extern void (*pushWaveforms)(void);
extern void (*pullWaveforms)(void);

extern size_t NPOP;
extern scalar *t;
extern unsigned long long *iT;

// Always present : model globals
extern int *simCycles;
extern scalar *clampGain;
extern scalar *accessResistance;

// Wavegen model globals
extern int *targetParam;
extern bool *final;
extern bool *getErr;

// Always present: model vars
extern scalar * err;
extern scalar * d_err;
}

#ifdef RUNNER_CC_COMPILE
// Model-independent, but GeNN-presence-dependent bridge code

__device__ WaveStats *dd_wavestats;
__device__ Stimulation *dd_waveforms;

void allocateGroupMem()
{
    using namespace GeNN_Bridge;
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
    using namespace GeNN_Bridge;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_wavestats, clear_wavestats, MM_NumGroups * sizeof(WaveStats), cudaMemcpyHostToDevice))
}
void pullStats()
{
    using namespace GeNN_Bridge;
    CHECK_CUDA_ERRORS(cudaMemcpy(wavestats, d_wavestats, MM_NumGroups * sizeof(WaveStats), cudaMemcpyDeviceToHost))
}
void pushWaveforms()
{
    using namespace GeNN_Bridge;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_waveforms, waveforms, MM_NumGroups * sizeof(Stimulation), cudaMemcpyHostToDevice))
}
void pullWaveforms()
{
    using namespace GeNN_Bridge;
    CHECK_CUDA_ERRORS(cudaMemcpy(waveforms, d_waveforms, MM_NumGroups * sizeof(Stimulation), cudaMemcpyDeviceToHost))
}
void freeGroupMem()
{
    using namespace GeNN_Bridge;
    cudaFreeHost(wavestats);
    cudaFreeHost(clear_wavestats);
    CHECK_CUDA_ERRORS(cudaFree(d_wavestats));
    cudaFreeHost(waveforms);
    CHECK_CUDA_ERRORS(cudaFree(d_waveforms));
}

void populate(MetaModel &m); // Defined in GeNN-produced support_code.h

void libManualInit(MetaModel &m) // Must be called separately (through GeNN_Bridge::init())
{
    allocateMem();
    allocateGroupMem();
    initialize();
    clearStats();
    populate(m);
}

void __attribute__ ((constructor)) libInit()
{
    GeNN_Bridge::init =& libManualInit;
    GeNN_Bridge::push =& pushHHStateToDevice;
    GeNN_Bridge::pull =& pullHHStateFromDevice;
    GeNN_Bridge::step =& stepTimeGPU;
    GeNN_Bridge::reset =& initialize;
    GeNN_Bridge::pullStats =& pullStats;
    GeNN_Bridge::clearStats =& clearStats;
    GeNN_Bridge::pushWaveforms =& pushWaveforms;
    GeNN_Bridge::pullWaveforms =& pullWaveforms;

    GeNN_Bridge::t =& t;
    GeNN_Bridge::iT =& iT;
}

void libExit()
{
    freeMem();
    freeGroupMem();
    cudaDeviceReset();
    GeNN_Bridge::init = 0;
    GeNN_Bridge::push = 0;
    GeNN_Bridge::pull = 0;
    GeNN_Bridge::step = 0;
    GeNN_Bridge::reset = 0;
    GeNN_Bridge::pullStats = 0;
    GeNN_Bridge::clearStats = 0;
    GeNN_Bridge::pushWaveforms = 0;
    GeNN_Bridge::pullWaveforms = 0;

    GeNN_Bridge::t = 0;
    GeNN_Bridge::iT = 0;
}
#endif

#endif // KERNELHELPER_H
