#ifndef WAVEGEN_GLOBALS_H
#define WAVEGEN_GLOBALS_H

#include "types.h"
#include "metamodel.h"

namespace Wavegen_Global {

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

void populate(MetaModel &);

extern size_t NPOP;
extern scalar *t;
extern unsigned long long *iT;

// Model globals
extern int *simCycles;
extern scalar *clampGain;
extern scalar *accessResistance;
extern int *targetParam;
extern bool *final;
extern bool *getErr;

// Model vars
extern scalar * err;
extern scalar * d_err;
}

#endif // WAVEGEN_GLOBALS_H
