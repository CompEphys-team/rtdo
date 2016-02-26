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

RealtimeEnvironment *RealtimeEnvironment::_instance = nullptr;

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

#ifdef CONFIG_RT
// ------------------------------ Realtime implementation ---------------------
#include "realtime/realtimeenvironment_impl.h"
#include <iomanip>

RealtimeEnvironment::RealtimeEnvironment() :
    sImpl(new Simulator)
{
    rt_allow_nonroot_hrt();
    pImpl.reset(new Impl);
}

RealtimeEnvironment::~RealtimeEnvironment() {}

void RealtimeEnvironment::sync()
{
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

        // Flush the read queue
        pImpl->in->preload.wait();
        for ( Channel& c : pImpl->in->channels ) {
            c.flush();
        }

        // Reporting
        if ( pImpl->in->reporting ) {
            RTIME loop;
            // Ignore first iteration after reset
            if ( pImpl->nSyncs++ && pImpl->in->reportQ.pop(loop, false) ) {
                RTIME best = 0, worst = loop, sum = 0, buffer;
                int n, overruns = 0;
                for ( n = 0; pImpl->in->reportQ.pop(buffer, false); n++ ) {
                    sum += buffer;
                    if ( buffer < worst )
                        worst = buffer;
                    if ( buffer > best )
                        best = buffer;
                    if ( buffer < 0 )
                        overruns++;
                }
                RTIME average = count2nano(sum/n);
                double saturation = (1.0 - average * 1.0/count2nano(loop));
                int recommendation = config->io.ai_supersampling * 1.0 / saturation;
                pImpl->supersamplingRecommend += recommendation;
                pImpl->nSyncs++;
                rt_make_soft_real_time();
                cout << setprecision(2) << fixed
                     << "Average idle time in analog in: " << average << " ns" << endl
                     << "saturation " << 100.0*saturation << "%" << endl
                     << "spread: " << count2nano(worst) << " ns to " << count2nano(best) << " ns." << endl
                     << "total overruns, out of " << n << " acquisitions: " << overruns << endl
                     << "Maximum recommended supersampling rate: " << (int)(pImpl->supersamplingRecommend / pImpl->nSyncs) << endl;
                rt_make_hard_real_time();
            }
        }
        pImpl->in->reportQ.flush();

        // Load input assignment
        pImpl->in->load.signal();

        // Wait for load
        pImpl->in->prime();
        pImpl->out->prime();
        pImpl->in->presync.wait();
        pImpl->out->presync.wait();
        rt_sleep(nano2count(100000));

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
    pImpl->nSyncs = 0;
    pImpl->supersamplingRecommend = 0;
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

void RealtimeEnvironment::setIdleTimeReporting(bool yes)
{
    pImpl->in->reporting = yes;
    pImpl->nSyncs = 0;
    pImpl->supersamplingRecommend  = 0;
}

bool RealtimeEnvironment::idleTimeReporting() const
{
    return pImpl->in->reporting;
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
void RealtimeEnvironment::setIdleTimeReporting(bool) {}
bool RealtimeEnvironment::idleTimeReporting() const { return false; }

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
