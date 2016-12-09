#include "kernelhelper.h"

namespace GeNN_Bridge {

WaveStats *wavestats;
WaveStats *clear_wavestats;
WaveStats *d_wavestats;

Stimulation *waveforms;
Stimulation *d_waveforms;

void (*push)(void);
void (*pull)(void);
void (*step)(void);
void (*init)(MetaModel&);
void (*reset)(void);
void (*pullStats)(void);
void (*clearStats)(void);
void (*pushWaveforms)(void);
void (*pullWaveforms)(void);

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
bool *getErr;

// Always present: model vars
scalar * err;
scalar * d_err;

}
