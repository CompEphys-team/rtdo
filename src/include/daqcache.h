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
