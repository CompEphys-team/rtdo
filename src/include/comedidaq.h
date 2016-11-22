#ifndef COMEDIDAQ_H
#define COMEDIDAQ_H

#include "daq.h"
#include "thread.h"
#include "conditionvariable.h"
#include "queue.h"
#include "comediconverter.h"

namespace RTMaybe {
    class ComediDAQ;
}

class RTMaybe::ComediDAQ : public DAQ
{
public:
    ComediDAQ(ComediData *p);
    ~ComediDAQ();

    void run(Stimulation s);
    void next();
    void reset();

protected:
    bool live;
    RTMaybe::ConditionVariable ready, set, go, finish;
    RTMaybe::Queue<lsampl_t> qI, qV;
    RTMaybe::Thread t;

    ComediConverter conI, conV, conO;

    static void *launchStatic(void *);
    void *launch();
};

#endif // COMEDIDAQ_H
