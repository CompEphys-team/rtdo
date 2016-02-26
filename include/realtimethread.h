/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-22

--------------------------------------------------------------------------*/
#ifndef REALTIMETHREAD_H
#define REALTIMETHREAD_H

#include <memory>

class RealtimeThread
{
public:
    /*! Note: Use of RealtimeThread without CONFIG_RT is supported, but it uses std::thread and ignores all arguments
     *  except for @a fn and @a arg.
     * @param priority goes from 0 (high) to 99 (low) in RTAI.
     */
    RealtimeThread(void *(*fn)(void *), void *arg, int priority = 50, int stackSize = 0, int cpusAllowed = 0xFF,
                   int policy = SCHED_FIFO, int maxMessageSize = 0, unsigned long name = 0);
    ~RealtimeThread();

    RealtimeThread(const RealtimeThread&) = delete;
    RealtimeThread &operator=(const RealtimeThread&) = delete;

    void *join();
    bool joinable() const;

    class Impl;
private:
    std::unique_ptr<Impl> pImpl;
};

#endif // REALTIMETHREAD_H
