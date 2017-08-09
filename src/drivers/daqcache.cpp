#include "daqcache.h"
#include "comedidaq.h"
#include <algorithm>
#include "session.h"

DAQCache::DAQCache(Session &session) :
    DAQ(session),
    daq(p.simulate ? (DAQ*)(session.project.experiment().createSimulator(session, true)) : (DAQ*)(new RTMaybe::ComediDAQ(session)))
{

}

DAQCache::~DAQCache()
{
    session.project.experiment().destroySimulator(daq);
}

void DAQCache::setAdjustableParam(size_t idx, double value)
{
    daq->setAdjustableParam(idx, value);
}

int DAQCache::throttledFor(const Stimulation &s)
{
    int diff = daq->throttledFor(s);
    if ( diff > 0 )
        for ( const Cache &c : cache )
            if ( c.stim == s )
                return 0;
    return diff;
}

void DAQCache::run(Stimulation s)
{
    if ( running )
        return;
    currentStim = s;
    for ( iterC = cache.begin(); iterC != cache.end(); ++iterC ) {
        if ( iterC->stim == currentStim )
            break;
    }
    if ( iterC == cache.end() ) {
        cache.emplace_back(currentStim, p.cache.numTraces, currentStim.duration/samplingDt() + 1 + (p.filter.active ? p.filter.width : 0));
        iterC = --cache.end();
        iterI = iterC->sampI[0].begin();
        iterV = iterC->sampV[0].begin();
        collecting = true;
    } else {
        if ( iterC->trace == p.cache.numTraces-1 ) {
            iterI = iterC->medI.begin();
            iterV = iterC->medV.begin();
            collecting = false;
        } else if ( daq->throttledFor(s) > 0 ) {
            iterI = iterC->medI.begin();
            iterV = iterC->medV.begin();
            collecting = false;
        } else {
            ++iterC->trace;
            iterI = iterC->sampI[iterC->trace].begin();
            iterV = iterC->sampV[iterC->trace].begin();
            collecting = true;
        }
    }

    if ( collecting )
        daq->run(s);

    running = true;
}

void DAQCache::next()
{
    if ( !running )
        return;

    if ( collecting ) {
        daq->next();
        *iterI = daq->current;
        *iterV = daq->voltage;

        std::size_t offset = iterI - iterC->sampI[iterC->trace].begin();
        if ( p.cache.useMedian ) {
            std::vector<double> curr(iterC->trace + 1), volt(iterC->trace + 1);
            for ( std::size_t i = 0; i <= iterC->trace; i++ ) {
                curr[i] = iterC->sampI[i][offset];
                volt[i] = iterC->sampV[i][offset];
            }
            std::sort(curr.begin(), curr.end());
            std::sort(volt.begin(), volt.end());
            current = curr[iterC->trace / 2];
            voltage = volt[iterC->trace / 2];
        } else {
            current = (iterC->medI[offset] * iterC->trace + *iterI) / (iterC->trace+1);
            voltage = (iterC->medV[offset] * iterC->trace + *iterV) / (iterC->trace+1);
        }
        iterC->medI[offset] = current;
        iterC->medV[offset] = voltage;
    } else {
        current = *iterI;
        voltage = *iterV;
    }
    ++iterI;
    ++iterV;
}

void DAQCache::reset()
{
    if ( !running )
        return;
    if ( collecting )
        daq->reset();
    running = false;
}

DAQCache::Cache::Cache(Stimulation stim, std::size_t numTraces, std::size_t traceLen) :
    stim(stim),
    trace(0),
    sampI(numTraces, std::vector<double>(traceLen)),
    sampV(numTraces, std::vector<double>(traceLen)),
    medI(traceLen, 0.0),
    medV(traceLen, 0.0)
{

}
