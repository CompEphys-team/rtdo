/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-25

--------------------------------------------------------------------------*/
#ifndef ANALOGTHREAD_H
#define ANALOGTHREAD_H

#include <vector>
#include "channel.h"
#include "realtimethread.h"
#include "realtimeconditionvariable.h"
#include "realtimequeue.h"

class AnalogThread
{
public:
    AnalogThread(bool in, std::shared_ptr<RealtimeConditionVariable> sync);
    ~AnalogThread(); //!< Joins the thread. Warning: This destructor calls sync.broadcast()!

    inline void pause() { running = false; } //!< Pause thread execution. Returns immediately, thread will wait on load.
    inline void prime() { running = true; }

    inline void quit() { exit = true; } //!< End thread execution. Returns immediately, thread may be left running

    inline void setSupersamplingRate(int acquisitionsPerSample) { supersampling = acquisitionsPerSample; }

    void *inputFn();
    void *outputFn();

    //!< Copies of channels added through RealtimeEnvironment::addChannel
    std::vector<Channel> channels;

    RealtimeConditionVariable preload;
    RealtimeConditionVariable load;
    RealtimeConditionVariable presync;
    std::shared_ptr<RealtimeConditionVariable> sync;

    bool reloadChannels;

    bool reporting;
    RealtimeQueue<RTIME> reportQ;

private:
    bool running;
    bool exit;

    int supersampling;

    std::unique_ptr<RealtimeThread> t;
};

#endif // ANALOGTHREAD_H
