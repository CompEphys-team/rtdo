#include "calibrator.h"
#include "daqfilter.h"

Calibrator::Calibrator(Session &s, QObject *parent) :
    QObject(parent),
    session(s)
{
    session.appropriate(this);
}

void Calibrator::zeroV1(DAQData p)
{
    emit zeroingV1(false);

    // Remove existing offset
    p.voltageChn.offset = 0;
    session.setDAQData(p);

    // Initialise
    DAQFilter daq(session);
    Stimulation stim;
    stim.duration = 1000;
    stim.baseV = session.daqData().vclampChan.offset; // Zero output
    stim.clear();

    // Measure the average input
    daq.run(stim);
    double voltage = 0;
    int nSamples = daq.samplesRemaining;
    while ( daq.samplesRemaining > 0 ) {
        daq.next();
        voltage += daq.voltage;
    }
    voltage /= nSamples;

    // Set the new offset to the measured value (see ComediConverter for sign rationale; offset is *added* to the raw value)
    p.voltageChn.offset = -voltage;
    session.setDAQData(p);

    emit zeroingV1(true);
}

void Calibrator::zeroVout(DAQData p)
{
    emit zeroingVout(false);

    // Remove existing offset - output /should/ already be zeroed from zeroing input values
    p.vclampChan.offset = 0;
    session.setDAQData(p);

    // Initialise
    DAQFilter daq(session);
    Stimulation stim;
    stim.duration = 1000;
    stim.baseV = 0;
    stim.clear();

    // Measure -- twice, in case output was not, in fact, zero to start with.
    double voltage;
    for ( int i = 0; i < 2; i++ ) {
        daq.run(stim);
        voltage = 0;
        int nSamples = daq.samplesRemaining;
        while ( daq.samplesRemaining > 0 ) {
            daq.next();
            voltage += daq.voltage;
        }
        daq.reset();
        voltage /= nSamples;
    }

    // Set the new offset
    p.vclampChan.offset = voltage;
    session.setDAQData(p);

    emit zeroingVout(true);
}
