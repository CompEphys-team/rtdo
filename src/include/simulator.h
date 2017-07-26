#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "daq.h"

class Simulator : public DAQ
{
public:
    Simulator(Session &session) : DAQ(session) {}
    ~Simulator() {}

    virtual void setAdjustableParam(size_t idx, double value) = 0;
};

#endif // SIMULATOR_H
