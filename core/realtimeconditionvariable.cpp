/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-25

--------------------------------------------------------------------------*/
#include "realtimeconditionvariable.h"
#include "realtimeenvironment.h"

#ifdef CONFIG_RT
// ------------------------------ Realtime implementation ---------------------
extern "C" {
#include <rtai_sem.h>
}

class RealtimeConditionVariable::Impl
{
public:
    Impl() :
        sem(rt_typed_sem_init(0, 0, BIN_SEM))
    {
        if ( !sem ) {
            throw RealtimeException(RealtimeException::Setup, "rt_typed_sem_init");
        }
        rt_sem_signal(sem);
        if ( rt_sem_wait(sem) != 1 ) {
            throw RealtimeException(RealtimeException::SemaphoreTest);
        }
    }

    ~Impl()
    {
        rt_sem_delete(sem);
    }

    SEM *sem;
};

RealtimeConditionVariable::RealtimeConditionVariable() :
    pImpl(new Impl)
{}

RealtimeConditionVariable::~RealtimeConditionVariable() {}

void RealtimeConditionVariable::wait()
{
    int ret = rt_sem_wait(pImpl->sem);
    if ( ret > 1 ) {
        throw RealtimeException(RealtimeException::RuntimeFunc, "rt_sem_wait", ret);
    }
}

bool RealtimeConditionVariable::wait_if()
{
    int ret = rt_sem_wait_if(pImpl->sem);
    if ( ret > 1 ) {
        throw RealtimeException(RealtimeException::RuntimeFunc, "rt_sem_wait_if", ret);
    }
    return (bool)ret;
}

void RealtimeConditionVariable::signal()
{
    int ret = rt_sem_signal(pImpl->sem);
    if ( ret > 1 ) {
        throw RealtimeException(RealtimeException::RuntimeFunc, "rt_sem_signal", ret);
    }
}

void RealtimeConditionVariable::broadcast()
{
    int ret = rt_sem_broadcast(pImpl->sem);
    if ( ret > 1 ) {
        throw RealtimeException(RealtimeException::RuntimeFunc, "rt_sem_broadcast", ret);
    }
}

#else
// ------------------------------ Non-realtime implementation -----------------
#include <condition_variable>

class RealtimeConditionVariable::Impl
{
public:
    Impl() :
        count(0),
        release(0)
    {}

    std::condition_variable v;
    std::mutex m;
    int count;
    int release;
};

RealtimeConditionVariable::RealtimeConditionVariable() :
    pImpl(new Impl)
{}

RealtimeConditionVariable::~RealtimeConditionVariable()
{
    bool retry;
    do {
        broadcast();
        pImpl->m.lock();
        retry = ( pImpl->count < 0 );
        pImpl->m.unlock();
    } while ( retry );
}

void RealtimeConditionVariable::wait()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    --pImpl->count;
    while ( !pImpl->release ) pImpl->v.wait(lock);
    --pImpl->release;
    ++pImpl->count;
}

bool RealtimeConditionVariable::wait_if()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    if ( pImpl->count > 0 ) {
        --pImpl->count;
        return true;
    } else {
        return false;
    }
}

void RealtimeConditionVariable::signal()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    pImpl->release = 1;
    pImpl->v.notify_all();
}

void RealtimeConditionVariable::broadcast()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    pImpl->release = -1 * pImpl->count;
    pImpl->v.notify_all();
}

#endif
