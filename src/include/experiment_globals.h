#ifndef EXPERIMENT_GLOBALS_H
#define EXPERIMENT_GLOBALS_H

#include "types.h"
#include "metamodel.h"
#include "daq.h"

namespace Experiment_Global {

extern void (*push)(void);
extern void (*pull)(void);
extern void (*step)(void);
extern void (*init)(MetaModel&);
extern void (*reset)(void);
extern DAQ* (*createSimulator)(DAQData*, RunData*);
extern void (*destroySimulator)(DAQ*);

void populate(MetaModel &);
DAQ *createSim(DAQData*, RunData*);
void destroySim(DAQ*);

extern size_t NPOP;
extern scalar *t;
extern unsigned long long *iT;

// Model globals
extern int *simCycles;
extern bool *getErrG;
extern scalar *clampGain;
extern scalar *accessResistance;
extern scalar *VmemG;
extern scalar *ImemG;
extern bool *VC;

// Model vars
extern scalar * err;
extern scalar * d_err;

}

#endif // EXPERIMENT_GLOBALS_H
