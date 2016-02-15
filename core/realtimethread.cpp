/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-22

--------------------------------------------------------------------------*/
#include "realtimethread.h"
#include "realtimeenvironment.h"

#ifdef CONFIG_RT
// ----------------------------- Realtime implementation -----------------------------------
extern "C" {
#include <rtai_lxrt.h>
}
#include "realtimeconditionvariable.h"

class RealtimeThread::Impl
{
public:
    Impl(void *(*fn)(void *), void *arg, int priority, int stackSize, int policy,
         int maxMessageSize, int cpusAllowed, unsigned long name) :
        joinable(false),
        fn(fn),
        arg(arg),
        policy(policy),
        priority(priority),
        maxMessageSize(maxMessageSize),
        cpusAllowed(cpusAllowed),
        name(name),
        cv(new RealtimeConditionVariable)
    {
        id = rt_thread_create((void *)&Impl::launchStatic, this, stackSize);
        if ( !id )
            throw RealtimeException(RealtimeException::Setup, "rt_thread_create");

        cv->wait();
        if ( !joinable ) // Launch failed
            throw RealtimeException(RealtimeException::Setup, "rt_thread_init");

        delete cv;
        cv = 0;
    }

    long id;
    bool joinable;

private:
    void *(*fn)(void *);
    void *arg;

    int policy;
    int priority;

    int maxMessageSize;
    int cpusAllowed;
    unsigned long name;

    RealtimeConditionVariable *cv;

    static void *launchStatic(RealtimeThread::Impl *_this)
    {
        return _this->launch();
    }

    void *launch()
    {
        RT_TASK *task;
        task = rt_thread_init(name, priority, maxMessageSize, policy, cpusAllowed);
        if ( !task ) {
            cv->signal();
            // Defer throwing to the constructor
            return (void *)EXIT_FAILURE;
        }

        joinable = true;
        rt_make_hard_real_time();
        cv->signal();

        void *ret = fn(arg);

        rt_thread_delete(task);
        return ret;
    }
};

RealtimeThread::RealtimeThread(void *(*fn)(void *), void *arg, int priority, int stackSize, int policy,
                               int maxMessageSize, int cpusAllowed, unsigned long name) :
    pImpl(new Impl(fn, arg, priority, stackSize, policy, maxMessageSize, cpusAllowed, name))
{}

RealtimeThread::~RealtimeThread()
{
    if ( joinable() )
        join();
}

void *RealtimeThread::join()
{
    if ( !pImpl->joinable )
        return (void *)ESRCH;
    // Since rt_thread_join is just a wrapper that throws away the return value of pthread_join:
    void *tret;
    int jret = pthread_join(pImpl->id, &tret);
    pImpl->joinable = false;
    if ( jret )
        throw RealtimeException(RealtimeException::RuntimeFunc, "pthread_join", jret);
    return tret;
}

bool RealtimeThread::joinable() const
{
    return pImpl->joinable;
}

#else
// -------------------------------------- Non-realtime implementation --------------------
#include <thread>

class RealtimeThread::Impl
{
public:
    Impl(void *(*fn)(void *), void *arg) :
        t(std::thread(fn, arg))
    {}

    std::thread t;
};

RealtimeThread::RealtimeThread(void *(*fn)(void *), void *arg, int, int, int, int, int, unsigned long) :
    pImpl(new Impl(fn, arg))
{}

RealtimeThread::~RealtimeThread()
{
    if ( joinable() )
        join();
}

void *RealtimeThread::join()
{
    pImpl->t.join();
    return 0;
}

bool RealtimeThread::joinable() const
{
    return pImpl->t.joinable();
}

#endif
