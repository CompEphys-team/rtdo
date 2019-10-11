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


#ifndef DAQFILTER_H
#define DAQFILTER_H

#include "daq.h"
#include "filter.h"
#include "canneddaq.h"

class DAQFilter : public DAQ
{
public:
    DAQFilter(Session &s, const Settings &settings);
    ~DAQFilter();

    inline double getAdjustableParam(size_t idx) { return daq->getAdjustableParam(idx); }
    inline void setAdjustableParam(size_t idx, double value) { daq->setAdjustableParam(idx, value); }
    inline int throttledFor(const Stimulation &s) { return daq->throttledFor(s); }
    void run(Stimulation s, double settlingDuration = 0);
    void next();
    void reset();

    CannedDAQ *getCannedDAQ();

protected:
    DAQ *daq;
    bool initial;
    std::vector<double> currentBuffer, voltageBuffer, V2Buffer;
    int bufferIndex;
    Filter filter;
};

#endif // DAQFILTER_H
