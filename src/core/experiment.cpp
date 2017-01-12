#include "experiment.h"
#include "supportcode.cu"
#include <cassert>
#include <functional>

Experiment::Experiment(MetaModel &model, const std::string &dir, const ExperimentData &expd, const RunData &rund, DAQ *daq) :
    expd(expd),
    lib(model, dir, expd),
    simulator(lib.createSimulator()),
    daq(daq ? daq : simulator)
{
    setRunData(rund);
}

Experiment::~Experiment()
{
    lib.destroySimulator(simulator);
}

void Experiment::setRunData(const RunData &r)
{
    rund = r;
    lib.simCycles = r.simCycles;
    lib.clampGain = r.clampGain;
    lib.accessResistance = r.accessResistance;
}

std::vector<scalar> Experiment::errProfile(const Stimulation &I, std::vector<scalar> model, size_t targetParam)
{
    assert(model.size() == lib.adjustableParams.size());
    assert(targetParam < lib.adjustableParams.size());
    std::vector<scalar> ret(expd.numCandidates);

    // Set parameter values
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        if ( i == targetParam ) {
            for ( size_t j = 0; j < expd.numCandidates; j++ )
                lib.adjustableParams[i][j] = errProfile_value(i, j);
        } else {
            for ( size_t j = 0; j < expd.numCandidates; j++ )
                lib.adjustableParams[i][j] = model[i];
        }
    }

    // Reset err
    for ( size_t i = 0; i < expd.numCandidates; i++ )
        lib.err[i] = 0.0;

    // Stimulate
    lib.push();
    stimulate(I);

    // Return err
    lib.pullErr();
    for ( size_t i = 0; i < expd.numCandidates; i++ )
        ret[i] = lib.err[i];
    return ret;
}

scalar Experiment::errProfile_value(size_t targetParam, size_t idx)
{
    const AdjustableParam &p = lib.adjustableParams[targetParam];
    return p.min + (p.max-p.min) * idx / (expd.numCandidates-1);
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
