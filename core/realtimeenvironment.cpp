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

class RealtimeEnvironment::Simulator
{
public:
    inputSpec waveform;
    bool useFloat;
    double (*simDouble)(double*, double*, double);
    float (*simFloat)(float *, float *, float);
    double *simVarsDouble, *simParamsDouble;
    float *simVarsFloat, *simParamsFloat;
    double DT;

    void sync()
    {
        lt = 0.0;
        stepV = waveform.baseV;
        sn = 0;
    }

    double nextSample()
    {
        double sample = useFloat
                ? simFloat(simVarsFloat, simParamsFloat, stepV)
                : simDouble(simVarsDouble, simParamsDouble, stepV);
        lt += DT;
        if ((sn < waveform.N) && (((lt - DT < waveform.st[sn]) && (lt >= waveform.st[sn])) || (waveform.st[sn] == 0))) {
            stepV = waveform.V[sn];
            sn++;
        }
        return sample;
    }

private:
    double lt;
    double stepV;
    int sn;
};


void RealtimeEnvironment::setDT(double dt)
{
    sImpl->DT = dt;
}

void RealtimeEnvironment::setSimulator(double (*fn)(double *, double *, double))
{
    sImpl->simDouble = fn;
    sImpl->useFloat = false;
}

void RealtimeEnvironment::setSimulator(float (*fn)(float *, float *, float))
{
    sImpl->simFloat = fn;
    sImpl->useFloat = true;
}

void RealtimeEnvironment::setSimulatorVariables(double *vars)
{
    sImpl->simVarsDouble = vars;
}

void RealtimeEnvironment::setSimulatorVariables(float *vars)
{
    sImpl->simVarsFloat = vars;
}

void RealtimeEnvironment::setSimulatorParameters(double *params)
{
    sImpl->simParamsDouble = params;
}

void RealtimeEnvironment::setSimulatorParameters(float *params)
{
    sImpl->simParamsFloat = params;
}

double RealtimeEnvironment::getClampGain()
{
    return config->vc.gain / config->vc.resistance;
    // Units are not an oversight: with gain in V/V, resistance in MOhm, current in nA, and voltage in mV,
    // units do cancel out nicely.
}

#ifdef CONFIG_RT
// ------------------------------ Realtime implementation ---------------------
#include "realtime/realtimeenvironment_impl.h"

RealtimeEnvironment::RealtimeEnvironment() :
    sImpl(new Simulator)
{
    rt_allow_nonroot_hrt();
    pImpl.reset(new Impl);
}

RealtimeEnvironment::~RealtimeEnvironment() {}

void RealtimeEnvironment::sync()
{
    if ( sImpl->useFloat )
        *((float *)clampGainParam) = (float) getClampGain();
    else
        *((double *)clampGainParam) = getClampGain();

    if ( _useSim ) {
        sImpl->sync();
    } else {
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
}

void RealtimeEnvironment::pause()
{
    if ( !_useSim ) {
        pImpl->in->pause();
        pImpl->out->pause();
    }
}

void RealtimeEnvironment::setSupersamplingRate(int r)
{
    pImpl->in->setSupersamplingRate(r);
}

void RealtimeEnvironment::setWaveform(const inputSpec &i)
{
    pImpl->out->channels.at(0).setWaveform(i);
    sImpl->waveform = i;
}

double RealtimeEnvironment::nextSample(int channelIndex)
{
    if ( _useSim )
        return sImpl->nextSample();
    else
        return pImpl->in->channels.at(channelIndex).nextSample();
}

bool RealtimeEnvironment::useSimulator(bool set)
{
    _useSim = set;
    return true;
}

void RealtimeEnvironment::addChannel(Channel &c)
{
    if ( c.direction() == Channel::AnalogOut ) {
        pImpl->out->channels.clear();
        pImpl->out->channels.push_back(c);
        pImpl->out->reloadChannels = true;
    } else if ( c.direction() == Channel::AnalogIn ) {
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

unsigned int RealtimeEnvironment::getSubdevice(int deviceno, Channel::Direction type)
{
    return pImpl->getSubdevice(deviceno, type);
}

#else
// ------------------------------ Non-realtime implementation --------------------------

class RealtimeEnvironment::Impl {};
RealtimeEnvironment::RealtimeEnvironment() :
    sImpl(new Simulator)
{}

RealtimeEnvironment::~RealtimeEnvironment() {}

void RealtimeEnvironment::sync()
{
    if ( sImpl->useFloat )
        *((float *)clampGainParam) = (float) getClampGain();
    else
        *((double *)clampGainParam) = getClampGain();
    sImpl->sync();
}

void RealtimeEnvironment::setWaveform(const inputSpec &i)
{
    sImpl->waveform = i;
}

double RealtimeEnvironment::nextSample(int)
{
    return sImpl->nextSample();
}

bool RealtimeEnvironment::useSimulator(bool)
{
    return false;
}

void RealtimeEnvironment::addChannel(Channel &) {}
void RealtimeEnvironment::clearChannels() {}
void RealtimeEnvironment::pause() {}
void RealtimeEnvironment::setSupersamplingRate(int) {}
struct comedi_t_struct *RealtimeEnvironment::getDevice(int, bool) { return 0; }
std::string RealtimeEnvironment::getDeviceName(int deviceno) { return std::string("Simulator"); }
unsigned int RealtimeEnvironment::getSubdevice(int, Channel::Direction) { return 0; }

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
