#ifndef DAQ_H
#define DAQ_H

#include "types.h"
#include "randutils.hpp"

class DAQ
{
public:
    DAQ(Session &session, const Settings &settings);
    virtual ~DAQ();

    /// For simulator instances only: Get/set the model parameters. For analog DAQs, this should be a return 0 / no-op.
    virtual double getAdjustableParam(size_t idx) = 0;
    virtual void setAdjustableParam(size_t idx, double value) = 0;

    /// For throttled instances: Returns the number of ms remaining until responses to @s can start to be provided
    virtual int throttledFor(const Stimulation &s) = 0;

    /// Run a stimulation/acquisition epoch, starting immediately
    virtual void run(Stimulation s, double settleDuration = 0) = 0;

    /// Advance to next set of inputs, populating DAQ::current and DAQ::voltage
    virtual void next() = 0;

    double current;
    double voltage;
    double voltage_2;
    int samplesRemaining;

    double outputResolution; //!< Time in ms between command output updates. Populated upon calling @fn run().

    /// Stop stimulation/acquisition, discarding any acquired inputs
    virtual void reset() = 0;

    Project &project;
    const DAQData &p;

    /// Voltage clamp flag. Affects the choice of output channel (VC command or current output)
    bool VC;

    const RunData &rund;
    randutils::mt19937_rng &RNG;

protected:
    bool running;
    Stimulation currentStim;

    double samplingDt() const;
    void extendStimulation(Stimulation &stim, scalar settleDuration);
};

#endif // DAQ_H
