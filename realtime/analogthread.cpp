/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-25

--------------------------------------------------------------------------*/
#ifdef CONFIG_RT

extern "C" {
#include <rtai_lxrt.h>
}
#include <iostream>
#include "analogthread.h"
#include "config.h"
#include "channel.h"


void *launchIn(void *_this)
{
    try {
        return ((AnalogThread *)_this)->inputFn();
    } catch ( RealtimeException &e ) {
        std::cerr << "An exception occurred in the analog input thread:" << std::endl << e.what() << std::endl;
        throw;
    }
}

void *launchOut(void *_this)
{
    try {
        return ((AnalogThread *)_this)->outputFn();
    } catch ( RealtimeException &e ) {
        std::cerr << "An exception occurred in the analog output thread:" << std::endl << e.what() << std::endl;
        throw;
    }
}


AnalogThread::AnalogThread(bool in, std::shared_ptr<RealtimeConditionVariable> sync) :
    sync(sync),
    reloadChannels(true),
    reporting(false),
    reportQ(Channel_MailboxSize, RealtimeQueue<RTIME>::Tenfold),
    running(false),
    exit(false),
    t(new RealtimeThread(in ? launchIn : launchOut, this, AnalogThread_Priority, AnalogThread_StackSize, SCHED_FIFO,
                         0, AnalogThread_CPUMask, nam2num(in ? "AIFUNC" : "AOFUNC")))
{}

AnalogThread::~AnalogThread()
{
    exit = true;
    running = false;
    load.broadcast();
    sync->broadcast();
    t->join();
}

void *AnalogThread::inputFn()
{
    int i, nchans=0, iter, nsum=0;
    RTIME now, expected, samp_ticks=0, tDiff;
    lsampl_t sample;
    std::vector<lsampl_t> sums;
    std::vector<Channel> chans;
    bool report;

    while ( !exit ) {
        // Load new channel config
        preload.signal();
        load.wait();

        if ( reloadChannels ) {
            rt_make_soft_real_time();
            chans = channels; // copy
            sums = std::vector<lsampl_t>(channels.size(), 0);
            nchans = (int) channels.size();
            reloadChannels = false;
            rt_make_hard_real_time();
        }
        nsum = supersampling;
        samp_ticks = nano2count(1e6 * config->io.dt / nsum);

        if ( (report = reporting) ) {
            reportQ.push(samp_ticks);
        }

        iter = 0;

        // Wait for sync
        presync.signal();
        if ( exit || !nchans ) {
            continue;
        }
        sync->wait();

        expected = rt_get_time() + samp_ticks;
        while ( running ) {
            // Read samples
            i = 0;
            for ( Channel &c : chans ) {
                if ( !c.read(sample, (nchans > 1)) ) { // Fatal: Read failed.
                    std::cerr << "Error reading from AI, thread exited" << std::endl;
                    return (void *)EXIT_FAILURE;
                }
                if ( nsum > 1 ) {
                    sums[i] += sample;
                    if ( (iter+1) % nsum == 0 ) {
                        sample = sums[i] / nsum;
                        sums[i] = 0;
                        c.put(sample);
                    }
                    ++i;
                } else {
                    c.put(sample);
                }
            }

            // Wait period
            iter++;
            now = rt_get_time();
            tDiff = expected - now;
            if ( tDiff > 0 ) {
                rt_sleep(tDiff);
            }
            expected += samp_ticks;

            if ( report ) {
                reportQ.push(tDiff, false);
            }
        }
    }

    return 0;
}

void *AnalogThread::outputFn()
{
    int step;
    RTIME now, expected, toff;
    Channel channel(Channel::AnalogOut);
    bool hasChannel = false;
    std::vector<lsampl_t> samples;
    std::vector<RTIME> times;

    while ( !exit ) {
        // Load new channel config
        preload.signal();
        load.wait();
        
        
        rt_make_soft_real_time();
        if ( reloadChannels ) {
            if ( (hasChannel = !channels.empty()) ) {
                channel = channels.at(0);
            }
            reloadChannels = false;
        }
        if ( hasChannel && channel.waveformChanged() ) {
            const inputSpec &w = channel.waveform();
            samples.clear();
            times.clear();
            samples.push_back(channel.convert(w.baseV));
            for ( int i = 0; i < w.N; i++ ) {
                samples.push_back(channel.convert(w.V.at(i)));
                times.push_back(nano2count((RTIME)(1e6 * w.st.at(i))));
            }
            times.push_back(nano2count((RTIME)(1e6 * w.t)));
        }
        rt_make_hard_real_time();

        // Wait for sync
        presync.signal();
        if ( !hasChannel || exit ) {
            continue;
        }
        sync->wait();

        toff = rt_get_time();
        expected = toff + times.at(step = 0);
        while ( running ) {
            if ( !channel.write(samples.at(step)) ) { // Fatal: Write failure
                std::cerr << "Error writing to AO, thread exited" << std::endl;
                return (void *)EXIT_FAILURE;
            }

            ++step;
            
            // Wait period
            now = rt_get_time();
            if ( now < expected ) {
                rt_sleep(expected-now);
            }

            if ( step == channel.waveform().N + 1 ) { // Return to base value before leaving
                if ( !channel.write(samples.front()) ) { // Fatal: Write failure
                    std::cerr << "Error writing to AO, thread exited" << std::endl;
                    return (void *)EXIT_FAILURE;
                }
                break;
            }

            expected = toff + times.at(step);
        }
    }

    return 0;
}

#endif
