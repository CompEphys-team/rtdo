#ifndef KERNELHELPER_H
#define KERNELHELPER_H

#include "types.h"
#include "metamodel.h"

namespace GeNN_Bridge {

extern void (*push)(void);
extern void (*pull)(void);
extern void (*step)(void);
extern void (*init)(MetaModel&);
extern void (*reset)(void);

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
extern int * bubbles;
extern int * d_bubbles;
extern int * cyclesWon;
extern int * d_cyclesWon;
extern scalar * wonByAbs;
extern scalar * d_wonByAbs;
extern scalar * wonByRel;
extern scalar * d_wonByRel;
extern scalar * wonOverMean;
extern scalar * d_wonOverMean;
extern int * cCyclesWon;
extern int * d_cCyclesWon;
extern scalar * cWonByAbs;
extern scalar * d_cWonByAbs;
extern scalar * cWonByRel;
extern scalar * d_cWonByRel;
extern scalar * cWonOverMean;
extern scalar * d_cWonOverMean;
extern int * bCyclesWon;
extern int * d_bCyclesWon;
extern scalar * bCyclesWonT;
extern scalar * d_bCyclesWonT;
extern scalar * bCyclesWonA;
extern scalar * d_bCyclesWonA;
extern scalar * bCyclesWonR;
extern scalar * d_bCyclesWonR;
extern scalar * bCyclesWonM;
extern scalar * d_bCyclesWonM;
extern scalar * bWonByAbs;
extern scalar * d_bWonByAbs;
extern scalar * bWonByAbsT;
extern scalar * d_bWonByAbsT;
extern int * bWonByAbsC;
extern int * d_bWonByAbsC;
extern scalar * bWonByAbsR;
extern scalar * d_bWonByAbsR;
extern scalar * bWonByAbsM;
extern scalar * d_bWonByAbsM;
extern scalar * bWonByRel;
extern scalar * d_bWonByRel;
extern scalar * bWonByRelT;
extern scalar * d_bWonByRelT;
extern int * bWonByRelC;
extern int * d_bWonByRelC;
extern scalar * bWonByRelA;
extern scalar * d_bWonByRelA;
extern scalar * bWonByRelM;
extern scalar * d_bWonByRelM;
extern scalar * bWonOverMean;
extern scalar * d_bWonOverMean;
extern scalar * bWonOverMeanT;
extern scalar * d_bWonOverMeanT;
extern int * bWonOverMeanC;
extern int * d_bWonOverMeanC;
extern scalar * bWonOverMeanA;
extern scalar * d_bWonOverMeanA;
extern scalar * bWonOverMeanR;
extern scalar * d_bWonOverMeanR;
}

#ifdef RUNNER_CC_COMPILE
// Model-independent, but GeNN-presence-dependent bridge code

void populate(MetaModel &m); // Defined in GeNN-produced support_code.h

void libManualInit(MetaModel &m) // Must be called separately (through GeNN_Bridge::init())
{
    allocateMem();
    initialize();
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
    GeNN_Bridge::init = 0;
    GeNN_Bridge::push = 0;
    GeNN_Bridge::pull = 0;
    GeNN_Bridge::step = 0;
    GeNN_Bridge::reset = 0;

    GeNN_Bridge::t = 0;
    GeNN_Bridge::iT = 0;
}
#endif

#endif // KERNELHELPER_H
