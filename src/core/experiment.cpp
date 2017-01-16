#include "experiment.h"
#include "supportcode.cu"
#include <cassert>
#include <cmath>

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

    lib.push();

    // Settle
    Stimulation settle {};
    settle.duration = expd.settleDuration;
    settle.baseV = I.baseV;
    daq->run(settle);
    lib.t = 0.;
    lib.iT = 0;
    lib.getErr = false;
    lib.VC = true;
    lib.Vmem = settle.baseV;
    while ( lib.t < settle.duration ) {
        lib.step();
        daq->next();
    }

    // Stimulate
    stimulate(I);

    // Get err
    lib.pullErr();
    for ( size_t i = 0; i < expd.numCandidates; i++ )
        ret[i] = lib.err[i];

    // Correct retained & reset values
    auto iter = errProfile_retained.begin();
    for ( size_t idx : errProfile_retainedIdx ) {
        scalar sumErr = 0;
        for ( const scalar &e : *iter )
            sumErr += e;
        ret[idx] = sumErr;
        ++iter;
    }

    return ret;
}

scalar Experiment::errProfile_value(size_t targetParam, size_t idx)
{
    const AdjustableParam &p = lib.adjustableParams[targetParam];
    return p.min + (p.max-p.min) * idx / (expd.numCandidates-1);
}

size_t Experiment::errProfile_idx(size_t targetParam, scalar value)
{
    const AdjustableParam &p = lib.adjustableParams[targetParam];
    double precise = (value - p.min) / (p.max-p.min) * (expd.numCandidates-1);
    return std::round(precise);
}

std::vector<scalar> Experiment::errProfile_getRetained(size_t idx)
{
    size_t i = 0;
    for ( ; i < errProfile_retainedIdx.size(); i++ )
        if ( errProfile_retainedIdx[i] == idx )
            break;
    if ( i == errProfile_retainedIdx.size() )
        return std::vector<scalar>();
    return *std::next(errProfile_retained.begin(), i);
}

void Experiment::stimulate(const Stimulation &I)
{
    bool retain = errProfile_retainedIdx.size();
    if ( retain ) {
        size_t numSamples = I.duration / lib.model.cfg.dt;
        errProfile_retained = std::list<std::vector<scalar>>(errProfile_retainedIdx.size(), std::vector<scalar>(numSamples));
    }

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

        // Retain & reset requested values to build a diachronic error profile
        if ( retain ) {
            lib.pullErr();
            auto iter = errProfile_retained.begin();
            for ( size_t idx : errProfile_retainedIdx ) {
                (*iter)[lib.iT] = lib.err[idx];
                lib.err[idx] = 0;
                ++iter;
            }
            lib.pushErr();
        }
    }

    daq->reset();
}
