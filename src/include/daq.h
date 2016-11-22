#ifndef DAQ_H
#define DAQ_H

#include "types.h"

class DAQ
{
public:
    DAQ(DAQData *p);
    virtual ~DAQ();

    /// Run a stimulation/acquisition epoch, starting immediately
    virtual void run(Stimulation s) = 0;

    /// Advance to next set of inputs, populating DAQ::current and DAQ::voltage
    virtual void next() = 0;

    double current;
    double voltage;

    /// Stop stimulation/acquisition, discarding any acquired inputs
    virtual void reset() = 0;

    DAQData *p;

protected:
    bool running;
    Stimulation currentStim;
};

#endif // DAQ_H
