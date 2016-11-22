#include "conditionvariable.h"
#include <rtai_sem.h>
#include <stdexcept>
#include <QString>

namespace RTMaybe {

class ConditionVariable::Impl
{
public:
    Impl() :
        sem(rt_typed_sem_init(0, 0, BIN_SEM))
    {
        if ( !sem ) {
            throw std::runtime_error("RTAI semaphore setup failed. Is the rtai_sem kernel module active?");
        }
        rt_sem_signal(sem);
        if ( rt_sem_wait(sem) != 1 ) {
            rt_sem_delete(sem);
            throw std::runtime_error("RTAI semaphore function test failed. Is the rtai_sem kernel module active?");
        }
    }

    ~Impl()
    {
        rt_sem_delete(sem);
    }

    SEM *sem;
};

ConditionVariable::ConditionVariable() :
    pImpl(new Impl)
{}

ConditionVariable::~ConditionVariable() {}

void ConditionVariable::wait()
{
    int ret;
    do {
        ret = rt_sem_wait(pImpl->sem);
    } while ( ret == RTE_UNBLKD );
    if ( ret > RTE_BASE ) {
        throw std::runtime_error(QString("RTAI semaphore wait failed with error code %1").arg(ret-RTE_BASE).toStdString());
    }
}

bool ConditionVariable::wait_if()
{
    int ret;
    do {
        ret = rt_sem_wait_if(pImpl->sem);
    } while ( ret == RTE_UNBLKD );
    if ( ret > RTE_BASE ) {
        throw std::runtime_error(QString("RTAI semaphore wait_if failed with error code %1").arg(ret-RTE_BASE).toStdString());
    }
    return ret > 0;
}

bool ConditionVariable::wait_timed(long nanos)
{
    int ret;
    do {
        ret = rt_sem_wait_timed(pImpl->sem, nano2count(nanos));
    } while ( ret == RTE_UNBLKD );
    if ( ret > RTE_BASE ) {
        throw std::runtime_error(QString("RTAI semaphore wait_if failed with error code %1").arg(ret-RTE_BASE).toStdString());
    }
    return ret > 0;
}

void ConditionVariable::signal()
{
    int ret = rt_sem_signal(pImpl->sem);
    if ( ret > RTE_BASE ) {
        throw std::runtime_error(QString("RTAI semaphore signal failed with error code %1").arg(ret-RTE_BASE).toStdString());
    }
}

void ConditionVariable::broadcast()
{
    int ret = rt_sem_broadcast(pImpl->sem);
    if ( ret > RTE_BASE ) {
        throw std::runtime_error(QString("RTAI semaphore broadcast failed with error code %1").arg(ret-RTE_BASE).toStdString());
    }
}

}
