#include "experiment.h"
#include "supportcode.h"
#include <cassert>
#include <cmath>

Experiment::Experiment(ExperimentLibrary &lib, const ExperimentData &expd, DAQ *daq) :
    expd(expd),
    lib(lib),
    simulator(lib.createSimulator()),
    daq(daq ? daq : simulator)
{

}

Experiment::~Experiment()
{
    lib.destroySimulator(simulator);
}

void Experiment::stimulate(const Stimulation &I)
{
    // Set up library
    lib.t = 0.;
    lib.iT = 0;
    lib.VC = true;

    // Set up DAQ
    daq->reset();
    daq->run(I);

    // Stimulate both
    while ( lib.t < I.duration ) {
        daq->next();
        lib.Imem = daq->current;
        lib.Vmem = getCommandVoltage(I, lib.t);
        lib.getErr = (lib.t > I.tObsBegin && lib.t < I.tObsEnd);
        lib.step();
    }

    daq->reset();
}
