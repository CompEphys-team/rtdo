#include "experiment_globals.h"

namespace Experiment_Global {

void (*push)(void);
void (*pull)(void);
void (*step)(void);
void (*init)(MetaModel&);
void (*reset)(void);
DAQ* (*createSimulator)(DAQData*, RunData*);
void (*destroySimulator)(DAQ*);

size_t NPOP;
scalar *t;
unsigned long long *iT;

// Model globals
int *simCycles;
bool *getErrG;
scalar *clampGain;
scalar *accessResistance;
scalar *VmemG;
scalar *ImemG;
bool *VC;

// Model vars
scalar * err;
scalar * d_err;

}
