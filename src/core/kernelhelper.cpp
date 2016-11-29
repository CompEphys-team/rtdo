#include "kernelhelper.h"

namespace GeNN_Bridge {

WaveStats *wavestats;
WaveStats *d_wavestats;

void (*push)(void);
void (*pull)(void);
void (*step)(void);
void (*init)(MetaModel&);
void (*reset)(void);
void (*pullStats)(void);

// Global
size_t NPOP;
scalar *t;
unsigned long long *iT;

// Always present : model globals
int *simCycles;
scalar *clampGain;
scalar *accessResistance;

// Experiment model globals
scalar *VmemG;

// Wavegen model globals
int *targetParam;
bool *final;

// Always present: model vars
scalar * err;
scalar * d_err;

// Wavegen model vars
scalar * Vmem;
scalar * d_Vmem;
bool * getErr;
bool * d_getErr;

}
