#include "daqcache.h"
#include <algorithm>

DAQCache::DAQCache(DAQ *daq, CacheData *p) :
    DAQ(daq->p),
    daq(daq),
    cp(p)
{

}

DAQCache::~DAQCache()
{

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
        cache.emplace_back(currentStim, cp->numTraces, currentStim.duration/p->dt + 1);
        iterC = --cache.end();
        iterI = iterC->sampI[0].begin();
        iterV = iterC->sampV[0].begin();
        collecting = true;
    } else {
        if ( ++iterC->trace == cp->numTraces ) {
            iterI = iterC->medI.begin();
            iterV = iterC->medV.begin();
            collecting = false;
        } else {
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

        if ( cp->averageWhileCollecting ) {
            std::size_t offset = iterI - iterC->sampI[iterC->trace].begin();
            if ( cp->useMedian ) { // Terrible idea for performance!
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
            current = *(iterI++);
            voltage = *(iterV++);
        }
    } else {
        current = *(iterI++);
        voltage = *(iterV++);
    }
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
