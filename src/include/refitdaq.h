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

#ifndef REFITDAQ_H
#define REFITDAQ_H

#include "daq.h"
#include "gafitter.h"
#include <QFile>
#include <vector>

class RefitDAQ : public DAQ
{
public:
    RefitDAQ(Session &s, const Settings &settings, GAFitter::Output const& fit);
    ~RefitDAQ() = default;

    double getAdjustableParam(size_t);
    void setAdjustableParam(size_t, double) {}
    int throttledFor(const Stimulation &) { return 0; }
    void run(Stimulation s, double settleDuration = 0);
    void next();
    void reset() {}

    std::vector<iStimulation> getStims() { return stims; }

protected:
    Session *session;
    const GAFitter::Output &fit;
    std::vector<iStimulation> stims;
    std::vector<std::vector<double>> traces;
    std::vector<int> settleDur;
    double trace_dt;
    size_t trace_idx = 0;
    int nSkipSamples, sampleIdx;
};

#endif // REFITDAQ_H
