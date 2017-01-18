#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "experimentlibrary.h"

class Experiment
{
public:
    Experiment(ExperimentLibrary &lib, DAQ *daq = nullptr);
    ~Experiment();

    ExperimentLibrary &lib;

protected:
    DAQ *simulator;
    DAQ *daq;

    void stimulate(const Stimulation &I);
};

#endif // EXPERIMENT_H
