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
    DAQCache(Session &session, const Settings &settings);
    ~DAQCache();

    double getAdjustableParam(size_t idx);
    void setAdjustableParam(size_t idx, double value);
    int throttledFor(const Stimulation &s);
    void run(Stimulation s, double settleDuration = 0);
    void next();
    void reset();

protected:
    DAQ *daq;

    struct Cache
    {
        Cache(Stimulation stim, bool VC, std::size_t numTraces, std::size_t traceLen);
        Stimulation stim;
        bool VC;
        std::size_t trace, nCollected;
        std::vector<std::vector<double>> sampI, sampV, sampV2;
        std::vector<double> medI, medV, medV2;
        std::vector<QTime> time;
        double outputResolution;
    };

    std::list<Cache> cache;
    std::list<Cache>::iterator iterC;
    std::vector<double>::iterator iterI, iterV, iterV2;
    bool collecting, average;
};

#endif // DAQCACHE_H
