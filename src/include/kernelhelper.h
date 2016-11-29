#ifndef KERNELHELPER_H
#define KERNELHELPER_H

#include "types.h"
#include "metamodel.h"

namespace GeNN_Bridge {

struct WaveStats
{
    struct Bubble //!< Streak of winning over all other deviations.
    {
        int cycles; //!< Number of cycles won (= target param err > other param errs)
        scalar tEnd; //!< Time at end
        scalar abs; //!< Absolute distance to the next param err down (= target err - next err)
        scalar rel; //!< Relative distance to the next param err down (= (target err - next err)/next err)
        scalar meanAbs; //!< Absolute distance to mean param err
        scalar meanRel; //!< Relative distance to mean param err
    };

    int bubbles; // Number of bubbles
    Bubble totalBubble;
    Bubble currentBubble;
    Bubble longestBubble;
    Bubble bestAbsBubble;
    Bubble bestRelBubble;
    Bubble bestMeanAbsBubble;
    Bubble bestMeanRelBubble;

    struct Bud //!< Streak of winning over the mean deviation, rather than over all other deviations. May include bubbles.
    {
        int cycles; //!< Number of cycles over mean
        scalar tEnd; //!< Time at end
        scalar meanAbs; //!< Total absolute distance to mean
        scalar meanRel; //!< Total relative distance to mean (= (target err - mean err) / mean err)
        scalar shortfallAbs; //!< Total distance from winner (= target err - top err), note: piecewise negative where a bud is not also a bubble
        scalar shortfallRel; //!< Total relative distance from winner (= (target err - top err) / top err), note: may cross zero
    };

    int buds; // Number of buds
    Bud totalBud;
    Bud currentBud;
    Bud longestBud;
    Bud bestMeanAbsBud;
    Bud bestMeanRelBud;
    Bud bestShortfallAbsBud;
    Bud bestShortfallRelBud;
};

extern WaveStats *wavestats;
extern WaveStats *d_wavestats;

extern void (*push)(void);
extern void (*pull)(void);
extern void (*step)(void);
extern void (*init)(MetaModel&);
extern void (*reset)(void);
extern void (*pullStats)(void);

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
extern bool * getErr;
extern bool * d_getErr;
}

#ifdef RUNNER_CC_COMPILE
// Model-independent, but GeNN-presence-dependent bridge code

__device__ GeNN_Bridge::WaveStats *dd_wavestats;

void allocateStats()
{
    using namespace GeNN_Bridge;
    cudaHostAlloc(&wavestats, NGROUPS * sizeof(WaveStats), cudaHostAllocPortable);
        deviceMemAllocate(&d_wavestats, dd_wavestats, NGROUPS * sizeof(WaveStats));
}
void clearStats()
{
    using namespace GeNN_Bridge;
    for ( unsigned i = 0; i < NGROUPS; i++ )
        wavestats[i] = {0};
    CHECK_CUDA_ERRORS(cudaMemcpy(d_wavestats, wavestats, NGROUPS * sizeof(WaveStats), cudaMemcpyHostToDevice))
}
void pullStats()
{
    using namespace GeNN_Bridge;
    CHECK_CUDA_ERRORS(cudaMemcpy(wavestats, d_wavestats, NGROUPS * sizeof(WaveStats), cudaMemcpyDeviceToHost))
}
void freeStats()
{
    cudaFreeHost(GeNN_Bridge::wavestats);
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

    GeNN_Bridge::t = 0;
    GeNN_Bridge::iT = 0;
}
#endif

#endif // KERNELHELPER_H
