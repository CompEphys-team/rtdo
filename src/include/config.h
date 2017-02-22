#ifndef CONFIG_H
#define CONFIG_H

#include "types.h"

namespace Config
{
void init();

extern ModelData Model;
extern StimulationData Stimulation;
extern WavegenLibraryData WavegenLibrary;
extern WavegenData Wavegen;
extern RunData Run;
extern ExperimentData Experiment;

}

#endif // CONFIG_H
