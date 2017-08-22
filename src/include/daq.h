#ifndef DAQ_H
#define DAQ_H

#include "types.h"
#include "randutils.hpp"

class DAQ
{
public:
    DAQ(Session &session);
    virtual ~DAQ();

    /// For simulator instances only: Change the model parameters. For analog DAQs, this should be a no-op.
    virtual void setAdjustableParam(size_t idx, double value) = 0;

    /// For throttled instances: Returns the number of ms remaining until responses to @s can start to be provided
    virtual int throttledFor(const Stimulation &s) = 0;

    /// Run a stimulation/acquisition epoch, starting immediately
    virtual void run(Stimulation s) = 0;

    /// Advance to next set of inputs, populating DAQ::current and DAQ::voltage
    virtual void next() = 0;

    double current;
    double voltage;
    int samplesRemaining;

    /// Stop stimulation/acquisition, discarding any acquired inputs
    virtual void reset() = 0;

    Session &session;
    const DAQData &p;

    /// Voltage clamp flag. Affects the choice of output channel (VC command or current output)
    bool VC;

    /// For Simulator use:
    const RunData &rund;
    randutils::mt19937_rng &RNG;

protected:
    bool running;
    Stimulation currentStim;

    double samplingDt() const;
};

#endif // DAQ_H
