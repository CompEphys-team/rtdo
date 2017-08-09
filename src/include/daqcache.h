#ifndef DAQCACHE_H
#define DAQCACHE_H

#include "daq.h"
#include <vector>
#include <list>
#include <QTime>

/**
 * The DAQCache class provides a cached/caching interface to an actually acquiring DAQ object.
 * It hosts a ComediDAQ object, exposing data either from it (if required) or from cache (if possible).
 */
class DAQCache : public DAQ
{
public:
    DAQCache(Session &session);
    ~DAQCache();

    void setAdjustableParam(size_t idx, double value);
    int throttledFor(const Stimulation &s);
    void run(Stimulation s);
    void next();
    void reset();

protected:
    DAQ *daq;

    struct Cache
    {
        Cache(Stimulation stim, std::size_t numTraces, std::size_t traceLen);
        Stimulation stim;
        std::size_t trace, nCollected;
        std::vector<std::vector<double>> sampI, sampV;
        std::vector<double> medI, medV;
        std::vector<QTime> time;
    };

    std::list<Cache> cache;
    std::list<Cache>::iterator iterC;
    std::vector<double>::iterator iterI, iterV;
    bool collecting, average;
};

#endif // DAQCACHE_H
