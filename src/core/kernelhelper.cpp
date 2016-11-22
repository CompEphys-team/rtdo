#include "kernelhelper.h"

namespace GeNN_Bridge {

void (*push)(void);
void (*pull)(void);
void (*step)(void);
void (*init)(MetaModel&);
void (*reset)(void);

// Global
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
int * bubbles;
int * d_bubbles;
int * cyclesWon;
int * d_cyclesWon;
scalar * wonByAbs;
scalar * d_wonByAbs;
scalar * wonByRel;
scalar * d_wonByRel;
scalar * wonOverMean;
scalar * d_wonOverMean;
int * cCyclesWon;
int * d_cCyclesWon;
scalar * cWonByAbs;
scalar * d_cWonByAbs;
scalar * cWonByRel;
scalar * d_cWonByRel;
scalar * cWonOverMean;
scalar * d_cWonOverMean;
int * bCyclesWon;
int * d_bCyclesWon;
scalar * bCyclesWonT;
scalar * d_bCyclesWonT;
scalar * bCyclesWonA;
scalar * d_bCyclesWonA;
scalar * bCyclesWonR;
scalar * d_bCyclesWonR;
scalar * bCyclesWonM;
scalar * d_bCyclesWonM;
scalar * bWonByAbs;
scalar * d_bWonByAbs;
scalar * bWonByAbsT;
scalar * d_bWonByAbsT;
int * bWonByAbsC;
int * d_bWonByAbsC;
scalar * bWonByAbsR;
scalar * d_bWonByAbsR;
scalar * bWonByAbsM;
scalar * d_bWonByAbsM;
scalar * bWonByRel;
scalar * d_bWonByRel;
scalar * bWonByRelT;
scalar * d_bWonByRelT;
int * bWonByRelC;
int * d_bWonByRelC;
scalar * bWonByRelA;
scalar * d_bWonByRelA;
scalar * bWonByRelM;
scalar * d_bWonByRelM;
scalar * bWonOverMean;
scalar * d_bWonOverMean;
scalar * bWonOverMeanT;
scalar * d_bWonOverMeanT;
int * bWonOverMeanC;
int * d_bWonOverMeanC;
scalar * bWonOverMeanA;
scalar * d_bWonOverMeanA;
scalar * bWonOverMeanR;
scalar * d_bWonOverMeanR;

}
