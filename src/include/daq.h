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


#ifndef DAQ_H
#define DAQ_H

#include "types.h"
#include "randutils.hpp"

class DAQ
{
public:
    DAQ(Session &session, const Settings &settings);
    virtual ~DAQ();

    /// For simulator instances only: Get/set the model parameters. For analog DAQs, this should be a return 0 / no-op.
    virtual double getAdjustableParam(size_t idx) = 0;
    virtual void setAdjustableParam(size_t idx, double value) = 0;

    /// For throttled instances: Returns the number of ms remaining until responses to @s can start to be provided
    virtual int throttledFor(const Stimulation &s) = 0;

    /// Run a stimulation/acquisition epoch, starting immediately
    virtual void run(Stimulation s, double settleDuration = 0) = 0;

    /// Advance to next set of inputs, populating DAQ::current and DAQ::voltage
    virtual void next() = 0;

    /// Returns the time between samples in ms
    double samplingDt() const;

    double current;
    double voltage;
    double voltage_2;
    int samplesRemaining;

    double outputResolution; //!< Time in ms between command output updates. Populated upon calling @fn run().

    /// Stop stimulation/acquisition, discarding any acquired inputs
    virtual void reset() = 0;

    Project &project;
    const DAQData &p;

    const RunData &rund;
    randutils::mt19937_rng &RNG;

protected:
    bool running;
    Stimulation currentStim;

    void extendStimulation(Stimulation &stim, scalar settleDuration);
};

#endif // DAQ_H
