/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#include "comedidaq.h"
#include <comedilib.h>
#include <QString>
#include <unistd.h>
#include "supportcode.h"
#include "session.h"

#define AIBUFSZ 2048

namespace RTMaybe {

ComediDAQ::ComediDAQ(Session &session, const Settings &settings) :
    DAQ(session, settings),
    live(true),
    ready(), set(), go(), finish(),
    qI(), qV(), qV2(),
    t(&ComediDAQ::launchStatic, this),
    conI(p.currentChn, &p, true),
    conV(p.voltageChn, &p, true),
    conV2(p.V2Chan, &p, true),
    conVC(p.vclampChan, &p, false),
    conCC(p.cclampChan, &p, false)
{

}

ComediDAQ::~ComediDAQ()
{
    live = false;
    running = false;
    ready.broadcast();
    set.broadcast();
    go.broadcast();
    finish.broadcast();
    reset();
}

int ComediDAQ::throttledFor(const Stimulation &)
{
    int diff = p.throttle - wallclock.elapsed();
    if ( wallclock.isNull() || diff <= 0 )
        return 0;
    else
        return diff;
}

void ComediDAQ::run(Stimulation s, double settleDuration)
{
    if ( running )
        return;

    extendStimulation(s, settleDuration);

    currentStim = s;
    samplesRemaining = nSamples();
    dispatch(samplesRemaining + 1);
}

void ComediDAQ::scope(int qSize)
{
    if ( running )
        return;
    currentStim.baseV = 0;
    samplesRemaining = 0;
    endless = true;
    dispatch(qSize);
}

void ComediDAQ::dispatch(int qSize)
{
    qI.flush();
    qV.flush();
    qV2.flush();
    qI.resize(qSize);
    qV.resize(qSize);
    qV2.resize(qSize);

    ready.signal();
    set.wait();
    running = true;
    go.signal();

    wallclock.restart();
}

void ComediDAQ::next()
{
    if ( running ) {
        lsampl_t c,v;
        if ( p.currentChn.active ) {
            qI.pop(c);
            current = conI.toPhys(c);
        }
        if ( p.voltageChn.active ) {
            qV.pop(v);
            voltage = conV.toPhys(v);
        }
        if ( p.V2Chan.active ) {
            qV2.pop(v);
            voltage_2 = conV2.toPhys(v);
        }

        if ( endless ) {
            if ( p.currentChn.active )
                samplesRemaining = qI.n_available_items();
            else if ( p.voltageChn.active )
                samplesRemaining = qV.n_available_items();
            else if ( p.V2Chan.active )
                samplesRemaining = qV2.n_available_items();
        } else
            --samplesRemaining;
    }
}

void ComediDAQ::reset()
{
    if ( !running )
        return;
    running = false;
    endless = false;
    finish.wait();
    qI.flush();
    qV.flush();
    qV2.flush();
    voltage = current = 0.0;
}

void *ComediDAQ::launchStatic(void *_this)
{
    return ((ComediDAQ*)_this)->launch();
}

void *ComediDAQ::launch()
{
    comedi_t *dev = comedi_open(p.devname().c_str());
    if ( !dev )
        throw std::runtime_error(QString("Failed to open device %1 in non-realtime mode")
                                 .arg(QString::fromStdString(p.devname()))
                                 .toStdString());

    int aidev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AI, 0);
    int aodev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AO, 0);
    if ( aidev < 0 || aodev < 0 )
        throw std::runtime_error(std::string("Failed to find subdevices: ") + comedi_strerror(comedi_errno()));

    unsigned int aiFlags = comedi_get_subdevice_flags(dev, aidev);
    unsigned int aoFlags = comedi_get_subdevice_flags(dev, aodev);
    if ( aiFlags & SDF_LSAMPL && aoFlags & SDF_LSAMPL )
        acquisitionLoop<lsampl_t, lsampl_t>(dev, aidev, aodev);
    else if ( aiFlags & SDF_LSAMPL )
        acquisitionLoop<lsampl_t, sampl_t>(dev, aidev, aodev);
    else if ( aoFlags & SDF_LSAMPL )
        acquisitionLoop<sampl_t, lsampl_t>(dev, aidev, aodev);
    else
        acquisitionLoop<sampl_t, sampl_t>(dev, aidev, aodev);

    return nullptr;
}

void doTrigger(comedi_t *dev, int aidev, int aodev)
{
    lsampl_t zero = 0;
    comedi_insn triggers[2];
    comedi_insnlist list;
    list.n_insns = 0;
    list.insns = triggers;

    if ( aidev >= 0 ) {
        triggers[list.n_insns].insn = INSN_INTTRIG;
        triggers[list.n_insns].subdev = aidev;
        triggers[list.n_insns].n = 1;
        triggers[list.n_insns].data = &zero;
        ++list.n_insns;
    }

    if ( aodev >= 0 ) {
        triggers[list.n_insns].insn = INSN_INTTRIG;
        triggers[list.n_insns].subdev = aodev;
        triggers[list.n_insns].n = 1;
        triggers[list.n_insns].data = &zero;
        ++list.n_insns;
    }

    int ret = comedi_do_insnlist(dev, &list);
    if ( ret < (int)list.n_insns )
        throw std::runtime_error(std::string("Failed start triggers: ") + comedi_strerror(comedi_errno()));
}


template <typename aisampl_t, typename aosampl_t>
void ComediDAQ::acquisitionLoop(void *vdev, int aidev, int aodev)
{
    comedi_t *dev = (comedi_t*)vdev;
    comedi_cmd aicmd, aocmd;
    ChnData V, V2, I, O;
    struct ChanPointer { ChnData *chan; RTMaybe::Queue<lsampl_t> *q; };
    ChanPointer aiChanPointers[3], allAiChans[] = {ChanPointer{&V, &qV}, ChanPointer{&V2, &qV2}, ChanPointer{&I, &qI}};
    ComediConverter *conO;
    lsampl_t V0;
    unsigned int aiDt;
    unsigned int aiChans[3], nAIChans, readOffset;
    unsigned int aoChan;
    int aiDataRemaining;
    std::vector<aosampl_t> aoData;
    char aiBuffer[AIBUFSZ];
    int ret;

    int aiBufSz = comedi_get_max_buffer_size(dev, aidev);
    if ( aiBufSz < 0 )
        throw std::runtime_error(std::string("Failed get AI max buffer size: ") + comedi_strerror(comedi_errno()));
    ret = comedi_set_buffer_size(dev, aidev, aiBufSz);
    aiBufSz = comedi_get_buffer_size(dev, aidev);
    if ( ret < 0 || aiBufSz < 0 )
        throw std::runtime_error(std::string("Failed set AI buffer size: ") + comedi_strerror(comedi_errno()));

    int aoBufSz = comedi_get_max_buffer_size(dev, aodev);
    if ( aoBufSz < 0 )
        throw std::runtime_error(std::string("Failed get AO max buffer size: ") + comedi_strerror(comedi_errno()));
    ret = comedi_set_buffer_size(dev, aodev, aoBufSz);
    aoBufSz = comedi_get_buffer_size(dev, aodev);
    if ( ret < 0 || aoBufSz < 0 )
        throw std::runtime_error(std::string("Failed set AO buffer size: ") + comedi_strerror(comedi_errno()));

    bool retry = false;
    while ( live ) {
        if ( !retry )
            ready.wait();
        if ( !live )
            break;

        V = p.voltageChn;
        V2 = p.V2Chan;
        I = p.currentChn;
        O = rund.VC ? p.vclampChan : p.cclampChan;
        conO =& (rund.VC ? conVC : conCC);

        // Set AO to baseV immediately
        V0 = conO->toSamp(currentStim.baseV);
        if ( O.active ) {
            ret = comedi_data_write(dev, aodev, O.idx, O.range, O.aref, V0);
            if ( ret < 0 )
                throw std::runtime_error(std::string("Failed synchronous write to AO: ") + comedi_strerror(comedi_errno()));
        }

        // AI setup
        if ( V.active || V2.active || I.active ) {
            comedi_set_read_subdevice(dev, aidev);
            ret = comedi_get_read_subdevice(dev);
            if ( ret != aidev )
                throw std::runtime_error("Failed to set AI read subdevice");

            aiDt = 1e6 * samplingDt();
            nAIChans = 0;
            for ( ChanPointer &ptr : allAiChans ) {
                if ( ptr.chan->active ) {
                    aiChanPointers[nAIChans] = ptr;
                    aiChans[nAIChans++] = CR_PACK(ptr.chan->idx, ptr.chan->range, ptr.chan->aref);
                }
            }
            ret = comedi_get_cmd_generic_timed(dev, aidev, &aicmd, nAIChans, aiDt);
            if ( ret < 0 )
                throw std::runtime_error(std::string("Failed AI command setup: ") + comedi_strerror(comedi_errno()));
            if ( aiDt != aicmd.scan_begin_arg ) {
                std::cout << "Warning: AI sampling interval is " << aicmd.scan_begin_arg << " ns instead of the requested " << aiDt << " ns." << std::endl;
                std::cout << "AI time scale is invalid; change (likely reduce) time step or oversampling rate to correct this issue." << std::endl;
            }
            aicmd.chanlist = aiChans;
            aicmd.start_src = TRIG_INT;
            aicmd.start_arg = 0;
            if ( endless ) {
                aicmd.stop_src = TRIG_NONE;
                aicmd.stop_arg = 0;
            } else {
                aicmd.stop_src = TRIG_COUNT;
                aicmd.stop_arg = nSamples();
            }
            aicmd.flags |= TRIG_DITHER;

            ret = comedi_command_test(dev, &aicmd);
            if ( ret < 0 )
                throw std::runtime_error(std::string("Failed AI command test: ") + comedi_strerror(comedi_errno()));
            ret = comedi_command_test(dev, &aicmd);
            if ( ret < 0 )
                throw std::runtime_error(std::string("Failed AI command test: ") + comedi_strerror(comedi_errno()));

            aiDataRemaining = aicmd.stop_arg;
            readOffset = 0;
        } else {
            nAIChans = readOffset = aiDataRemaining = 0;
        }

        // AO setup
        if ( O.active && !endless ) {
            comedi_set_write_subdevice(dev, aodev);
            ret = comedi_get_write_subdevice(dev);
            if ( ret != aodev )
                throw std::runtime_error("Failed to set AO write subdevice");

            int scan_ns = currentStim.duration * 1e6 / ((aoBufSz-sizeof(aosampl_t)) / sizeof(aosampl_t));
            do {
                ret = comedi_get_cmd_generic_timed(dev, aodev, &aocmd, 1, scan_ns);
                if ( ret < 0 )
                    throw std::runtime_error(std::string("Failed AO command setup: ") + comedi_strerror(comedi_errno()));
                scan_ns += 8;
                aocmd.stop_arg = currentStim.duration*1e6 / aocmd.scan_begin_arg
                        + 1 /* return to baseV upon completion */
                        + (p.filter.active ? p.filter.width/2 : 0); /* preload filter to give accurately filtered first sample */
                        // Note, AO does not need to be running past the stimulation's end even when filtering, as the output is V0 anyway.
            // Ensure the entire stimulation fits into one buffer write (can't read & write simultaneously):
            } while ( aocmd.stop_arg > aoBufSz/sizeof(aosampl_t) );
            aoChan = CR_PACK(O.idx, O.range, O.aref);
            aocmd.chanlist =& aoChan;
            aocmd.start_src = TRIG_INT;
            aocmd.start_arg = 0;
            aocmd.stop_src = TRIG_COUNT;
            aocmd.flags |= TRIG_DEGLITCH;

            ret = comedi_command_test(dev, &aicmd);
            if ( ret < 0 )
                throw std::runtime_error(std::string("Failed AO command test: ") + comedi_strerror(comedi_errno()));
            ret = comedi_command_test(dev, &aicmd);
            if ( ret < 0 )
                throw std::runtime_error(std::string("Failed AO command test: ") + comedi_strerror(comedi_errno()));

            aoData.resize(aocmd.stop_arg);
            for ( unsigned int i = 0, offset = (p.filter.active ? p.filter.width/2 : 0); i < aocmd.stop_arg-1; i++ )
                aoData[i] = conO->toSamp(getCommandVoltage(currentStim, 1e-6 * aocmd.scan_begin_arg * (i - offset)));
            aoData.back() = V0;

            outputResolution = 1e-6 * aocmd.scan_begin_arg;
        } else {
            aoData.clear();
            outputResolution = 0;
        }

        // Exit empty loop
        if ( !aiDataRemaining && aoData.empty() && !endless ) {
            set.signal();
            go.wait();
            running = false;
            finish.signal();
            continue;
        }

        // Send commands to device
        if ( aiDataRemaining || endless ) {
            ret = comedi_command(dev, &aicmd);
            if ( ret < 0 )
                throw std::runtime_error(std::string("Failed AI command: ") + comedi_strerror(comedi_errno()));
//            std::cout << "Read " << aiDataRemaining << " samples (" << 1e6/aicmd.scan_begin_arg << " kHz)\t";
        }
        if ( !aoData.empty() ) {
            ret = comedi_command(dev, &aocmd);
            if ( ret < 0 )
                throw std::runtime_error(std::string("Failed AO command: ") + comedi_strerror(comedi_errno()));
//            std::cout << "Write " << aoData.size() << " samples (" << 1e6/aocmd.scan_begin_arg << " kHz)";

            // Fill buffer
            int aoBytes = aoData.size() * sizeof(aosampl_t);
            ret = write(comedi_fileno(dev), aoData.data(), aoBytes);
            if ( ret != aoBytes )
                throw std::runtime_error(std::string("Failed AO buffer prefill: ") + strerror(errno));
        }
//        std::cout << std::endl;

        if ( !retry ) {
            set.signal();
            go.wait();
            if ( !live )
                break;
        }

        // Trigger start
        doTrigger(dev, (aiDataRemaining||endless) ? aidev : -1, (O.active&&!endless) ? aodev : -1);

        // Acquisition loop
        retry = false;
        while ( running && (aiDataRemaining > 0 || endless) ) {
            ret = read(comedi_fileno(dev), aiBuffer + readOffset, AIBUFSZ - readOffset);
            if ( ret < 0 ) {
                std::string err = std::string("Failed read: ") + strerror(errno);
                if ( endless ) {
                    std::cout << err << std::endl;
                    std::cout << "Reinitialising..." << std::endl;
                    retry = true;
                    break;
                }
                throw std::runtime_error(err);
            } else if ( ret > 0 ) {
                readOffset += ret;
                int i;
                for ( i = 0; readOffset >= nAIChans * sizeof(aisampl_t); readOffset -= nAIChans * sizeof(aisampl_t), i+= nAIChans ) {
                    for ( int j = 0; j < nAIChans; j++ )
                        aiChanPointers[j].q->push(((aisampl_t *)aiBuffer)[i+j]);
                    --aiDataRemaining;
                }
                if ( i > 0 && readOffset > 0 )
                    memcpy(aiBuffer, aiBuffer + i*sizeof(aisampl_t), readOffset);
            }
        }

        // Complete output before cancelling the commands
        while ( running && comedi_get_buffer_contents(dev, aodev) > 0 )
            usleep(20);
        comedi_cancel(dev, aodev);
        comedi_cancel(dev, aidev);

        if ( retry )
            continue;

        // Reset output
        comedi_data_write(dev, aodev, O.idx, O.range, O.aref, V0);

        finish.signal();
    }
}

}
