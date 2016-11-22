#include "thread.h"
#include <thread>

namespace RTMaybe
{

class Thread::Impl
{
public:
    Impl(void *(*fn)(void *), void *arg) :
        t(std::thread(fn, arg))
    {}

    std::thread t;
};

Thread::Thread(void *(*fn)(void *), void *arg, ThreadData) :
    pImpl(new Impl(fn, arg))
{}

Thread::~Thread()
{
    if ( joinable() )
        join();
}

void *Thread::join()
{
    pImpl->t.join();
    return 0;
}

bool Thread::joinable() const
{
    return pImpl->t.joinable();
}

void Thread::initEnv(ThreadData) {}

}
