#ifndef COMEDIDAQ_H
#define COMEDIDAQ_H

#include "daq.h"
#include "thread.h"
#include "conditionvariable.h"
#include "queue.h"
#include "comediconverter.h"
#include <QTime>

namespace RTMaybe {
    class ComediDAQ;
}

class RTMaybe::ComediDAQ : public DAQ
{
public:
    ComediDAQ(Session &session);
    ~ComediDAQ();

    inline void setAdjustableParam(size_t, double) {}
    int throttledFor(const Stimulation &s);
    void run(Stimulation s);
    void next();
    void reset();

    inline int nSamples() const { return currentStim.duration / samplingDt() + (p.filter.active ? p.filter.width : 0); }

protected:
    bool live;
    RTMaybe::ConditionVariable ready, set, go, finish;
    RTMaybe::Queue<lsampl_t> qI, qV;
    RTMaybe::Thread t;

    ComediConverter conI, conV, conVC, conCC;

    QTime wallclock;

    static void *launchStatic(void *);
    void *launch();

    template <typename aisampl_t, typename aosampl_t>
    void acquisitionLoop(void *dev, int aidev, int aodev);
};

#endif // COMEDIDAQ_H
