#include "conditionvariable.h"
#include <condition_variable>

namespace RTMaybe {

class ConditionVariable::Impl
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

ConditionVariable::ConditionVariable() :
    pImpl(new Impl)
{}

ConditionVariable::~ConditionVariable()
{
    bool retry;
    do {
        broadcast();
        pImpl->m.lock();
        retry = ( pImpl->count < 0 );
        pImpl->m.unlock();
    } while ( retry );
}

void ConditionVariable::wait()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    --pImpl->count;
    while ( !pImpl->release ) pImpl->v.wait(lock);
    --pImpl->release;
    ++pImpl->count;
}

bool ConditionVariable::wait_if()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    if ( pImpl->count > 0 ) {
        --pImpl->count;
        return true;
    } else {
        return false;
    }
}

void ConditionVariable::signal()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    pImpl->release = 1;
    pImpl->v.notify_all();
}

void ConditionVariable::broadcast()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    pImpl->release = -1 * pImpl->count;
    pImpl->v.notify_all();
}

}
