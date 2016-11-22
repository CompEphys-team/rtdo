#ifndef THREAD_H
#define THREAD_H

#include <sched.h>
#include "types.h"
#include "conditionvariable.h"

namespace RTMaybe {
    class Thread;
}

class RTMaybe::Thread
{
public:
    /*! Note: Use of Thread in non-RT builds is supported, but the ThreadData argument is ignored.
     * @param priority goes from 0 (high) to 99 (low) in RTAI.
     */
    Thread(void *(*fn)(void *), void *arg, ThreadData p = ThreadData());
    ~Thread();

    Thread(const Thread&) = delete;
    Thread &operator=(const Thread&) = delete;

    void *join();
    bool joinable() const;

    static void initEnv(ThreadData p); /// non-RT: noop. RT: Sets up the calling thread as an RT-capable task.

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // THREAD_H
