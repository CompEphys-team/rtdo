/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-25

--------------------------------------------------------------------------*/
#ifndef REALTIMECONDITIONVARIABLE_H
#define REALTIMECONDITIONVARIABLE_H

#include <memory>

class RealtimeConditionVariable
{
public:
    RealtimeConditionVariable();
    ~RealtimeConditionVariable();

    RealtimeConditionVariable(const RealtimeConditionVariable&) = delete;
    RealtimeConditionVariable &operator=(const RealtimeConditionVariable&) = delete;

    void wait();
    bool wait_if();
    void signal();
    void broadcast();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // REALTIMECONDITIONVARIABLE_H
