#include "comedidaq.h"
#include "RC_rtai_comedi.h"
#include <QString>

namespace RTMaybe {

ComediDAQ::ComediDAQ(ComediData *p) :
    DAQ(p),
    live(true),
    ready(), set(), go(), finish(),
    qI(), qV(),
    t(&ComediDAQ::launchStatic, this),
    conI(p->currentChn, p),
    conV(p->voltageChn, p),
    conO(p->stimChn, p)
{

}

ComediDAQ::~ComediDAQ()
{
    live = false;
    reset();
}

void ComediDAQ::run(Stimulation s)
{
    if ( running )
        return;
    currentStim = s;
    qI.flush();
    qV.flush();
    int qSize = currentStim.duration / p->dt + 1;
    qI.resize(qSize);
    qV.resize(qSize);
    ready.signal();
    set.wait();
    running = true;
    go.signal();
}

void ComediDAQ::next()
{
    if ( running ) {
        lsampl_t c,v;
        if ( p->currentChn.active ) {
            qI.pop(c);
            current = conI.toPhys(c);
        }
        if ( p->voltageChn.active ) {
            qV.pop(v);
            voltage = conV.toPhys(v);
        }
    }
}

void ComediDAQ::reset()
{
    if ( !running )
        return;
    running = false;
    finish.wait();
    qI.flush();
    qV.flush();
    voltage = current = 0.0;
}

void *ComediDAQ::launchStatic(void *_this)
{
    return ((ComediDAQ*)_this)->launch();
}

void *ComediDAQ::launch()
{
    ComediData *p = static_cast<ComediData*>(this->p);
    comedi_t *dev = RC_comedi_open(p->devname.c_str());
    if ( !dev )
        throw std::runtime_error(QString("Failed to open device %1 in realtime mode")
                                 .arg(QString::fromStdString(p->devname))
                                 .toStdString());
    int aidev = RC_comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AI, 0);
    int aodev = RC_comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AO, 0);
    RTIME dt, toffset, reltime;
    inChnData V, I;
    outChnData O;
    lsampl_t vSamp, iSamp, V0, VRamp, VRampDelta;
    struct Step {
        RTIME t;
        lsampl_t V;
        bool ramp;
    };
    std::vector<Step> steps;
    std::vector<Step>::iterator stepIter;
    while ( live ) {
        ready.wait();

        V = p->voltageChn;
        I = p->currentChn;
        O = p->stimChn;
        if ( (!V.active && !I.active) || !O.active ) {
            // Empty loop:
            set.signal();
            go.wait();
            running = false;
            finish.signal();
            continue;
        }
        V0 = conO.toSamp(currentStim.baseV);
        RC_comedi_data_write(dev, aodev, O.idx, O.range, O.aref, V0);
        rt_make_soft_real_time();

        // AI setup
        dt = nano2count((RTIME)(1e6 * p->dt));

        // AO setup
        steps.reserve(currentStim.steps.size() + 1);
        // Convert Stimulation::Step t/V to RTIME/lsampl_t
        for ( const Stimulation::Step &s : currentStim.steps ) {
            steps.push_back( Step {
                dt * (RTIME)(nano2count((RTIME)(1e6 * s.t)) / dt), // Round down
                conO.toSamp(s.V),
                s.ramp
            });
        }
        // Add an additional non-ramp step to return to base voltage:
        steps.push_back( Step {
            nano2count((RTIME)(1e6 * currentStim.duration)),
            conO.toSamp(currentStim.baseV),
            false
        });
        stepIter = steps.begin();
        if ( stepIter->ramp ) {
            VRamp = V0;
            VRampDelta = (stepIter->V - VRamp) / (stepIter->t / dt);
        }

        reltime = 0;
        rt_make_hard_real_time();
        set.signal();
        go.wait();
        toffset = rt_get_time();

        while ( running ) {
            // AI
            if ( I.active ) {
                if ( V.active )
                    RC_comedi_data_read_hint(dev, aidev, I.idx, I.range, I.aref);
                RC_comedi_data_read(dev, aidev, I.idx, I.range, I.aref, &iSamp);
                qI.push(iSamp);
            }
            if ( V.active ) {
                if ( I.active )
                    RC_comedi_data_read_hint(dev, aidev, V.idx, V.range, V.aref);
                RC_comedi_data_read(dev, aidev, V.idx, V.range, V.aref, &vSamp);
                qV.push(vSamp);
            }

            // AO
            if ( O.active ) {
                if ( stepIter->t == reltime ) {
                    RC_comedi_data_write(dev, aodev, O.idx, O.range, O.aref, stepIter->V);
                    ++stepIter;
                    if ( stepIter == steps.end() ) {
                        O.active = false;
                    } else if ( stepIter->ramp ) {
                        VRamp = (stepIter-1)->V;
                        VRampDelta = (stepIter->V - VRamp) / ((stepIter->t - reltime) / dt);
                    }
                } else if ( stepIter->ramp ) {
                    VRamp += VRampDelta;
                    RC_comedi_data_write(dev, aodev, O.idx, O.range, O.aref, VRamp);
                }
            }

            // Timing
            reltime += dt;
            if ( rt_get_time() < reltime + toffset )
                rt_sleep_until(reltime + toffset);
        }

        RC_comedi_data_write(dev, aodev, O.idx, O.range, O.aref, V0);
        finish.signal();
    }

    return nullptr;
}

}
