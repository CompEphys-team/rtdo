/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-25

--------------------------------------------------------------------------*/
#include "realtimeenvironment.h"
#include "config.h"

#ifdef CONFIG_RT
// ------------------------------ Realtime implementation ---------------------
#include "impl/realtimeenvironment_impl.h"

RealtimeEnvironment::RealtimeEnvironment()
{
    rt_allow_nonroot_hrt();
    pImpl.reset(new Impl);
}

RealtimeEnvironment::~RealtimeEnvironment() {}

void RealtimeEnvironment::sync()
{
    rt_make_hard_real_time();

    // Stop threads
    pImpl->in->pause();
    pImpl->out->pause();

    // Load output assignment
    pImpl->out->preload.wait();
    pImpl->out->load.signal();

    // Flush the message buffer & load input assignment
    pImpl->in->preload.wait();
    for ( Channel& c : pImpl->in->channels ) {
        c.flush();
    }
    pImpl->in->load.signal();

    // Wait for load
    pImpl->in->prime();
    pImpl->out->prime();
    pImpl->in->presync.wait();
    pImpl->out->presync.wait();

    // Release the hounds!
    pImpl->sync->broadcast();

    rt_make_soft_real_time();
}

void RealtimeEnvironment::pause()
{
    pImpl->in->pause();
    pImpl->out->pause();
}

void RealtimeEnvironment::setSupersamplingRate(int r)
{
    pImpl->in->setSupersamplingRate(r);
}

void RealtimeEnvironment::addChannel(Channel &c)
{
    if ( c.type() == Channel::AnalogOut ) {
        pImpl->out->channels[0] = c;
        pImpl->out->reloadChannels = true;
    } else if ( c.type() == Channel::AnalogIn ) {
        pImpl->in->channels.push_back(c);
        pImpl->in->reloadChannels = true;
    }
}

void RealtimeEnvironment::clearChannels()
{
    pImpl->out->channels.clear();
    pImpl->out->reloadChannels = true;
    pImpl->in->channels.clear();
    pImpl->in->reloadChannels = true;
}

struct comedi_t_struct *RealtimeEnvironment::getDevice(int n, bool RT)
{
    return RT ? pImpl->getDeviceRC(n) : pImpl->getDeviceSC(n);
}

std::string RealtimeEnvironment::getDeviceName(int deviceno)
{
    return pImpl->getDeviceName(deviceno);
}

unsigned int RealtimeEnvironment::getSubdevice(int deviceno, Channel::Type type)
{
    return pImpl->getSubdevice(deviceno, type);
}

#else
// ------------------------------ Non-realtime implementation --------------------------

class RealtimeEnvironment::Impl {
    // NYI
};
RealtimeEnvironment::RealtimeEnvironment() {}
RealtimeEnvironment::~RealtimeEnvironment() {}

void RealtimeEnvironment::sync()
{
    // NYI
}

void RealtimeEnvironment::addChannel(Channel &c)
{
    // NYI
}

void RealtimeEnvironment::clearChannels()
{
    // NYI
}

void RealtimeEnvironment::pause() {}
void RealtimeEnvironment::setSupersamplingRate(int) {}
struct comedi_t_struct *RealtimeEnvironment::getDevice(int, bool) { return 0; }
std::string RealtimeEnvironment::getDeviceName(int deviceno) { return std::string("Simulator"); }
unsigned int RealtimeEnvironment::getSubdevice(int, Channel::Type) { return 0; }

#endif


// ------------------------------------- Exception ---------------------------------------
RealtimeException::RealtimeException(type t, std::string funcname, int errval) :
    errtype(t),
    funcname(funcname),
    errval(errval)
{}

const char *RealtimeException::what() const noexcept
{
    std::string str;
    switch ( errtype ) {
    case Setup:
        str = string("RTAI setup failed in function ") + funcname + ". Is the rtai_sched kernel module active?";
        return str.c_str();
    case MailboxTest:
        return "RTAI mailbox function test failed. Is the rtai_mbx kernel module active?";
    case SemaphoreTest:
        return "RTAI semaphore function test failed. Is the rtai_sem kernel module active?";
    case RuntimeFunc:
        str = funcname + " returned the unexpected error value " + std::to_string(errval);
        return str.c_str();
    case RuntimeMsg:
        str = string("A runtime error occurred: ") + funcname;
        return str.c_str();
    case Timeout:
        str = string("An operation timed out while ") + funcname;
        return str.c_str();
    default:
        return "Unidentified exception";
    }
}
