#include "daq.h"
#include "session.h"

DAQ::DAQ(Session &session) :
    current(0.0),
    voltage(0.0),
    session(session),
    p(session.daqData()),
    running(false)
{

}

DAQ::~DAQ()
{
}
