#ifndef KERNELHELPER_H
#define KERNELHELPER_H

#include "types.h"
#include "metamodel.h"

namespace GeNN_Bridge {

extern WaveStats *wavestats;
extern WaveStats *clear_wavestats;
extern WaveStats *d_wavestats;

extern void (*push)(void);
extern void (*pull)(void);
extern void (*step)(void);
extern void (*init)(MetaModel&);
extern void (*reset)(void);
extern void (*pullStats)(void);
extern void (*clearStats)(void);

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

// Always present: model vars
extern scalar * err;
extern scalar * d_err;

// Wavegen model vars
extern scalar * Vmem;
extern scalar * d_Vmem;
extern scalar * Vramp;
extern scalar * d_Vramp;
extern bool * getErr;
extern bool * d_getErr;
}

#ifdef RUNNER_CC_COMPILE
// Model-independent, but GeNN-presence-dependent bridge code

__device__ WaveStats *dd_wavestats;

void allocateStats()
{
    using namespace GeNN_Bridge;
    cudaHostAlloc(&wavestats, MM_NumGroups * sizeof(WaveStats), cudaHostAllocPortable);
        deviceMemAllocate(&d_wavestats, dd_wavestats, MM_NumGroups * sizeof(WaveStats));

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
void freeStats()
{
    cudaFreeHost(GeNN_Bridge::wavestats);
    cudaFreeHost(GeNN_Bridge::clear_wavestats);
    CHECK_CUDA_ERRORS(cudaFree(GeNN_Bridge::d_wavestats));
}

void populate(MetaModel &m); // Defined in GeNN-produced support_code.h

void libManualInit(MetaModel &m) // Must be called separately (through GeNN_Bridge::init())
{
    allocateMem();
    allocateStats();
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

    GeNN_Bridge::t =& t;
    GeNN_Bridge::iT =& iT;
}

void libExit()
{
    freeMem();
    freeStats();
    cudaDeviceReset();
    GeNN_Bridge::init = 0;
    GeNN_Bridge::push = 0;
    GeNN_Bridge::pull = 0;
    GeNN_Bridge::step = 0;
    GeNN_Bridge::reset = 0;
    GeNN_Bridge::pullStats = 0;
    GeNN_Bridge::clearStats = 0;

    GeNN_Bridge::t = 0;
    GeNN_Bridge::iT = 0;
}
#endif

#endif // KERNELHELPER_H
