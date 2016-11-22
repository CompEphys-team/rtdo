#ifndef CONDITIONVARIABLE_H
#define CONDITIONVARIABLE_H

#include <memory>

namespace RTMaybe {
    class ConditionVariable;
}

class RTMaybe::ConditionVariable
{
public:
    ConditionVariable();
    ~ConditionVariable();

    ConditionVariable(const ConditionVariable&) = delete;
    ConditionVariable &operator=(const ConditionVariable&) = delete;

    void wait();
    bool wait_if(); /// @return true if the semaphore was taken successfully; false if it is occupied
    bool wait_timed(long nanos); /// Same here. Not implemented in non-RT build.
    void signal();
    void broadcast();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // CONDITIONVARIABLE_H
