#include "daqfilter.h"
#include "comedidaq.h"
#include "daqcache.h"
#include "session.h"

DAQFilter::DAQFilter(Session &s) :
    DAQ(s),
    kernel(p.filter.width),
    currentBuffer(p.filter.width),
    voltageBuffer(p.filter.width)
{
    if ( p.cache.active )
        daq = new DAQCache(session);
    else
        daq = new RTMaybe::ComediDAQ(session);

    if ( !p.filter.active )
        return;

    double norm;
    switch ( p.filter.method ) {
    case FilterMethod::MovingAverage:
    {
        norm = 1.0/p.filter.width;
        for ( double &v : kernel )
            v = norm;
        break;
    }
    case FilterMethod::SavitzkyGolay23:
    {
        // See Madden 1978, doi:10.1021/ac50031a048
        // p_s = 3(3m^2 + 3m - 1 - 5s^2) / ((2m+3)(2m+1)(2m-1))
        //     = (3(3m^2 + 3m - 1) - 15s^2) / ((2m+3)(2m+1)(2m-1))
        // Note the use of double precision throughout to prevent integer overflows for large m
        int m = p.filter.width/2;
        double numerator = 3*(3.*m*m + 3*m - 1);
        norm = 1.0 / ((2.*m+3)*(2*m+1)*(2*m-1));
        for ( int i = 0, s = -m; i < p.filter.width; i++, s++ ) // s in [-m, m]
            kernel[i] = (numerator - 15.*s*s) * norm;
        break;
    }
    case FilterMethod::SavitzkyGolay45:
    {
        // p_s = 15/4 * ((15m^4 + 30m^3 - 35m^2 - 50m + 12) - 35s^2(2m^2 + 2m - 3) + 63s^4)
        //            / ((2m+5)(2m+3)(2m+1)(2m-1)(2m-3))
        int m = p.filter.width/2;
        double numerator = (15.*m*m*m*m + 30*m*m*m - 35*m*m - 50*m + 12);
        double ssquareFactor = 35 * (2.*m*m + 2*m - 3);
        norm = 15. / 4. / ((2.*m+5)*(2*m+3)*(2*m+1)*(2*m-1)*(2*m-3));
        for ( int i = 0, s = -m; i < p.filter.width; i++, s++ ) // s in [-m, m]
            kernel[i] = (numerator - ssquareFactor*s*s + 63.*s*s*s*s) * norm;
        break;
    }
    }
}

DAQFilter::~DAQFilter()
{
    delete daq;
}

void DAQFilter::run(Stimulation s)
{
    daq->run(s);
    currentStim = s;
    initial = true;

}

void DAQFilter::next()
{
    if ( !p.filter.active ) {
        daq->next();
        current = daq->current;
        voltage = daq->voltage;
        return;
    }

    if ( initial ) {
        // Acquire into the buffer all samples necessary for the first actual time point
        // Note the assumption that daq acquire filterWidth/2 samples on either side of the actual stimulation.
        for ( int i = 0; i < p.filter.width; i++ ) {
            daq->next();
            currentBuffer[i] = daq->current;
            voltageBuffer[i] = daq->voltage;
        }
        bufferIndex = 0;
        initial = false;
    } else {
        // Acquire into the buffer one dt's worth of fresh samples, discarding old data
        for ( int i = 0; i < p.filter.samplesPerDt; i++ ) {
            daq->next();
            currentBuffer[bufferIndex] = daq->current;
            voltageBuffer[bufferIndex] = daq->voltage;
            if ( ++bufferIndex == p.filter.width )
                bufferIndex = 0;
        }
    }

    current = voltage = 0;
    // Convolve buffer (whose oldest sample is at [bufferIndex]) with the kernel.
    // Notice how bufferIndex returns to its original position, wrapping around the buffer.
    for ( int i = 0; i < p.filter.width; i++ ) {
        current += currentBuffer[bufferIndex] * kernel[i];
        voltage += voltageBuffer[bufferIndex] * kernel[i];
        if ( ++bufferIndex == p.filter.width )
            bufferIndex = 0;
    }
}

void DAQFilter::reset()
{
    daq->reset();
}
