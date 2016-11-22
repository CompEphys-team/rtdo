#include "daq.h"

DAQ::DAQ(DAQData *p) :
    current(0.0),
    voltage(0.0),
    p(p),
    running(false)
{

}

DAQ::~DAQ()
{
}
