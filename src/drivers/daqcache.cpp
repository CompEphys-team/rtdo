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


#include "daqcache.h"
#include "comedidaq.h"
#include <algorithm>
#include "session.h"

DAQCache::DAQCache(Session &session, const Settings &settings) :
    DAQ(session, settings),
    daq(p.simulate ? project.universal().createSimulator(p.simulate, session, settings, true) : new RTMaybe::ComediDAQ(session, settings))
{

}

DAQCache::~DAQCache()
{
    project.universal().destroySimulator(daq);
}

double DAQCache::getAdjustableParam(size_t idx)
{
    return getAdjustableParam(idx);
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

void DAQCache::run(Stimulation s, double settleDuration)
{
    if ( running )
        return;
    extendStimulation(s, settleDuration);
    currentStim = s;
    for ( iterC = cache.begin(); iterC != cache.end(); ++iterC ) {
        if ( iterC->stim == currentStim && iterC->VC == rund.VC )
            break;
    }
    if ( iterC == cache.end() ) {
        cache.emplace_back(currentStim, rund.VC, p.cache.numTraces, currentStim.duration/samplingDt() + 1 + (p.filter.active ? p.filter.width : 0));
        iterC = --cache.end();
        iterI = iterC->sampI[0].begin();
        iterV = iterC->sampV[0].begin();
        iterV2 = iterC->sampV2[0].begin();
        iterC->nCollected = 1;
        collecting = true;
    } else {
        // Throttled, use cached responses
        if ( daq->throttledFor(s) > 0 ) {
            collecting = false;
        // Cache full, refresh oldest response if necessary
        } else if ( iterC->nCollected == p.cache.numTraces ) {
            if ( p.cache.timeout > 0 ) {
                iterC->trace = 0;
                for ( size_t i = 1; i < iterC->nCollected; i++ )
                    if ( iterC->time[i] < iterC->time[iterC->trace] )
                        iterC->trace = i;
                collecting = ( iterC->time[iterC->trace].elapsed() > p.cache.timeout );
            } else {
                collecting = false;
            }
        // Cache not yet full, collect fresh response
        } else {
            ++iterC->trace;
            ++iterC->nCollected;
            collecting = true;
        }
    }

    if ( collecting ) {
        iterC->time[iterC->trace] = QTime::currentTime();
        iterI = iterC->sampI[iterC->trace].begin();
        iterV = iterC->sampV[iterC->trace].begin();
        iterV2 = iterC->sampV2[iterC->trace].begin();
        daq->run(s);
        samplesRemaining = daq->samplesRemaining;
        outputResolution = iterC->outputResolution = daq->outputResolution;
    } else {
        iterI = iterC->medI.begin();
        iterV = iterC->medV.begin();
        iterV2 = iterC->medV2.begin();
        samplesRemaining = iterC->medI.size();
        outputResolution = iterC->outputResolution;
    }

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
        *iterV2 = daq->voltage_2;

        std::size_t offset = iterI - iterC->sampI[iterC->trace].begin();
        if ( p.cache.useMedian ) {
            std::vector<double> curr(iterC->nCollected), volt(iterC->nCollected), volt2(iterC->nCollected);
            for ( std::size_t i = 0; i < iterC->nCollected; i++ ) {
                curr[i] = iterC->sampI[i][offset];
                volt[i] = iterC->sampV[i][offset];
                volt2[i] = iterC->sampV2[i][offset];
            }
            std::sort(curr.begin(), curr.end());
            std::sort(volt.begin(), volt.end());
            std::sort(volt2.begin(), volt2.end());
            if ( iterC->nCollected % 2 == 0 ) {
                current = (curr[iterC->nCollected/2] + curr[iterC->nCollected/2 - 1]) / 2;
                voltage = (volt[iterC->nCollected/2] + volt[iterC->nCollected/2 - 1]) / 2;
                voltage_2 = (volt2[iterC->nCollected/2] + volt2[iterC->nCollected/2 - 1]) / 2;
            } else {
                current = curr[iterC->nCollected/2];
                voltage = volt[iterC->nCollected/2];
                voltage_2 = volt2[iterC->nCollected/2];
            }
        } else {
            current = 0;
            voltage = 0;
            voltage_2 = 0;
            for ( std::size_t i = 0; i < iterC->nCollected; i++ ) {
                current += iterC->sampI[i][offset];
                voltage += iterC->sampV[i][offset];
                voltage_2 += iterC->sampV2[i][offset];
            }
            current /= iterC->nCollected;
            voltage /= iterC->nCollected;
            voltage_2 /= iterC->nCollected;
        }
        iterC->medI[offset] = current;
        iterC->medV[offset] = voltage;
        iterC->medV2[offset] = voltage_2;
    } else {
        current = *iterI;
        voltage = *iterV;
        voltage_2 = *iterV2;
    }
    ++iterI;
    ++iterV;
    ++iterV2;
    --samplesRemaining;
}

void DAQCache::reset()
{
    if ( !running )
        return;
    if ( collecting )
        daq->reset();
    running = false;
}

DAQCache::Cache::Cache(Stimulation stim, bool VC, std::size_t numTraces, std::size_t traceLen) :
    stim(stim),
    VC(VC),
    trace(0),
    nCollected(0),
    sampI(numTraces, std::vector<double>(traceLen)),
    sampV(numTraces, std::vector<double>(traceLen)),
    sampV2(numTraces, std::vector<double>(traceLen)),
    medI(traceLen, 0.0),
    medV(traceLen, 0.0),
    medV2(traceLen, 0.0),
    time(numTraces)
{

}
