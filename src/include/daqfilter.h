#ifndef DAQFILTER_H
#define DAQFILTER_H

#include "daq.h"

class DAQFilter : public DAQ
{
public:
    DAQFilter(Session &s);
    ~DAQFilter();

    void run(Stimulation s);
    void next();
    void reset();

protected:
    DAQ *daq;
    bool initial;
    std::vector<double> kernel, currentBuffer, voltageBuffer;
    int bufferIndex;
};

#endif // DAQFILTER_H
