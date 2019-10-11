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


#include "daq.h"
#include "session.h"

DAQ::DAQ(Session &session, const Settings &settings) :
    current(0.0),
    voltage(0.0),
    voltage_2(0.0),
    project(session.project),
    p(settings.daqd),
    rund(settings.rund),
    RNG(session.RNG),
    running(false)
{

}

DAQ::~DAQ()
{
}

double DAQ::samplingDt() const
{
    return p.filter.active
            ? rund.dt / p.filter.samplesPerDt
            : rund.dt;
}

void DAQ::extendStimulation(Stimulation &stim, scalar settleDuration)
{
    stim.duration += settleDuration;
    for ( Stimulation::Step &step : stim ) {
        step.t += settleDuration;
    }
    if ( stim.begin()->ramp )
        stim.insert(stim.begin(), Stimulation::Step {settleDuration, stim.baseV, false});
}
