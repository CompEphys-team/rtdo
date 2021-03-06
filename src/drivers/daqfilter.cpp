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


#include "daqfilter.h"
#include "comedidaq.h"
#include "daqcache.h"
#include "session.h"
#include "canneddaq.h"

DAQFilter::DAQFilter(Session &s, const Settings &settings) :
    DAQ(s, settings),
    currentBuffer(p.filter.width),
    voltageBuffer(p.filter.width),
    V2Buffer(p.filter.width),
    filter(p.filter.method, p.filter.width)
{
    if ( p.simulate == -1 )
        daq = new CannedDAQ(s, settings);
    else if ( p.cache.active )
        daq = new DAQCache(s, settings);
    else if ( p.simulate )
        daq = project.universal().createSimulator(p.simulate, s, settings, true);
    else
        daq = new RTMaybe::ComediDAQ(s, settings);

    if ( !p.filter.active )
        return;
}

DAQFilter::~DAQFilter()
{
    project.universal().destroySimulator(daq);
}

void DAQFilter::run(Stimulation s, double settlingDuration)
{
    daq->run(s, settlingDuration);
    currentStim = s;
    samplesRemaining = p.filter.active ? ((daq->samplesRemaining - p.filter.width) / p.filter.samplesPerDt) : (daq->samplesRemaining);
    outputResolution = daq->outputResolution;
    initial = true;

}

void DAQFilter::next()
{
    --samplesRemaining;
    if ( !p.filter.active ) {
        daq->next();
        current = daq->current;
        voltage = daq->voltage;
        voltage_2 = daq->voltage_2;
        return;
    }

    if ( initial ) {
        // Acquire into the buffer all samples necessary for the first actual time point
        // Note the assumption that daq acquire filterWidth/2 samples on either side of the actual stimulation.
        for ( int i = 0; i < p.filter.width; i++ ) {
            daq->next();
            currentBuffer[i] = daq->current;
            voltageBuffer[i] = daq->voltage;
            V2Buffer[i] = daq->voltage_2;
        }
        bufferIndex = 0;
        initial = false;
    } else {
        // Acquire into the buffer one dt's worth of fresh samples, discarding old data
        for ( int i = 0; i < p.filter.samplesPerDt; i++ ) {
            daq->next();
            currentBuffer[bufferIndex] = daq->current;
            voltageBuffer[bufferIndex] = daq->voltage;
            V2Buffer[bufferIndex] = daq->voltage_2;
            if ( ++bufferIndex == p.filter.width )
                bufferIndex = 0;
        }
    }

    current = voltage = voltage_2 = 0;
    // Convolve buffer (whose oldest sample is at [bufferIndex]) with the kernel.
    // Notice how bufferIndex returns to its original position, wrapping around the buffer.
    for ( int i = 0; i < p.filter.width; i++ ) {
        current += currentBuffer[bufferIndex] * filter.kernel[i];
        voltage += voltageBuffer[bufferIndex] * filter.kernel[i];
        voltage_2 += V2Buffer[bufferIndex] * filter.kernel[i];
        if ( ++bufferIndex == p.filter.width )
            bufferIndex = 0;
    }
}

void DAQFilter::reset()
{
    daq->reset();
}

CannedDAQ *DAQFilter::getCannedDAQ()
{
    if ( p.simulate < 0 )
        return static_cast<CannedDAQ*>(daq);
    else
        return nullptr;
}
