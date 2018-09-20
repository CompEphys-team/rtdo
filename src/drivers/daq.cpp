#include "daq.h"
#include "session.h"

DAQ::DAQ(Session &session, const Settings &settings) :
    current(0.0),
    voltage(0.0),
    voltage_2(0.0),
    project(session.project),
    p(settings.daqd),
    rund(settings.rund),
    RNG(session.RNG),
    running(false)
{

}

DAQ::~DAQ()
{
}

double DAQ::samplingDt() const
{
    return p.filter.active
            ? rund.dt / p.filter.samplesPerDt
            : rund.dt;
}

void DAQ::extendStimulation(Stimulation &stim, scalar settleDuration)
{
    stim.duration += settleDuration;
    for ( Stimulation::Step &step : stim ) {
        step.t += settleDuration;
    }
    if ( stim.begin()->ramp )
        stim.insert(stim.begin(), Stimulation::Step {settleDuration, stim.baseV, false});
}
