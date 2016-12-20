#include "wavegen_globals.h"

namespace Wavegen_Global {

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

// Model globals
int *simCycles;
scalar *clampGain;
scalar *accessResistance;
int *targetParam;
bool *final;
bool *getErr;

// Model vars
scalar * err;
scalar * d_err;

}
