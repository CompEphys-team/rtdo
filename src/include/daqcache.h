#ifndef DAQCACHE_H
#define DAQCACHE_H

#include "daq.h"
#include <vector>
#include <list>

/**
 * The DAQCache class provides a cached/caching interface to an actually acquiring DAQ object.
 * It takes control of the DAQ object it is given, exposing data either from it (if required)
 * or from cache (if possible).
 */
class DAQCache : public DAQ
{
public:
    DAQCache(DAQ *daq, CacheData *p);
    ~DAQCache();

    void run(Stimulation s);
    void next();
    void reset();

protected:
    DAQ *daq;
    CacheData *cp;

    struct Cache
    {
        Cache(Stimulation stim, std::size_t numTraces, std::size_t traceLen);
        Stimulation stim;
        std::size_t trace;
        std::vector<std::vector<double>> sampI, sampV;
        std::vector<double> medI, medV;
    };

    std::list<Cache> cache;
    std::list<Cache>::iterator iterC;
    std::vector<double>::iterator iterI, iterV;
    bool collecting, average;
};

#endif // DAQCACHE_H
