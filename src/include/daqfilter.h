#ifndef DAQFILTER_H
#define DAQFILTER_H

#include "daq.h"
#include "filter.h"

class DAQFilter : public DAQ
{
public:
    DAQFilter(Session &s);
    ~DAQFilter();

    inline double getAdjustableParam(size_t idx) { return daq->getAdjustableParam(idx); }
    inline void setAdjustableParam(size_t idx, double value) { daq->setAdjustableParam(idx, value); }
    inline int throttledFor(const Stimulation &s) { return daq->throttledFor(s); }
    void run(Stimulation s);
    void next();
    void reset();

protected:
    DAQ *daq;
    bool initial;
    std::vector<double> currentBuffer, voltageBuffer, V2Buffer;
    int bufferIndex;
    Filter filter;
};

#endif // DAQFILTER_H
