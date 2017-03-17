#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "experimentlibrary.h"

class Experiment
{
public:
    Experiment(ExperimentLibrary &lib, const ExperimentData &expd, DAQ *daq = nullptr);
    ~Experiment();

    const ExperimentData &expd;
    ExperimentLibrary &lib;

protected:
    DAQ *simulator;
    DAQ *daq;

    void stimulate(const Stimulation &I);
};

#endif // EXPERIMENT_H
