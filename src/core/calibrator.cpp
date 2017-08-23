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
    daq.VC = true;
    Stimulation stim;
    stim.duration = 1000;
    stim.baseV = session.daqData().vclampChan.offset; // Zero output
    // The above isn't technically required (V1 calibration happens in CC & saline and is unaffected by either Vcmd or I2 output),
    // but it's an opportune moment to reset Vcmd to (true) zero before the experimenter drops into VC.
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

void Calibrator::zeroV2(DAQData p)
{
    emit zeroingV2(false);

    // Remove existing offset
    p.V2Chan.offset = 0;
    session.setDAQData(p);

    // Initialise
    DAQFilter daq(session);
    daq.VC = false;
    Stimulation stim;
    stim.duration = 1000;
    stim.baseV = 0; // Zero Iout to (only manually defined) offset
    stim.clear();

    // Measure the average input
    daq.run(stim);
    double voltage = 0;
    int nSamples = daq.samplesRemaining;
    while ( daq.samplesRemaining > 0 ) {
        daq.next();
        voltage += daq.voltage_2;
    }
    voltage /= nSamples;

    // Set the new offset to the measured value (see ComediConverter for sign rationale; offset is *added* to the raw value)
    p.V2Chan.offset = -voltage;
    session.setDAQData(p);

    emit zeroingV2(true);
}

void Calibrator::zeroIin(DAQData p)
{
    emit zeroingIin(false);

    // Remove existing offset
    p.currentChn.offset = 0;
    session.setDAQData(p);

    // Initialise
    DAQFilter daq(session);
    daq.VC = false;
    Stimulation stim;
    stim.duration = 1000;
    stim.baseV = 0; // Zero Iout to (only manually defined) offset
    stim.clear();

    // Measure the average input
    daq.run(stim);
    double current = 0;
    int nSamples = daq.samplesRemaining;
    while ( daq.samplesRemaining > 0 ) {
        daq.next();
        current += daq.current;
    }
    current /= nSamples;

    // Set the new offset to the measured value (see ComediConverter for sign rationale; offset is *added* to the raw value)
    p.currentChn.offset = -current;
    session.setDAQData(p);

    emit zeroingIin(true);
}

void Calibrator::zeroVout(DAQData p)
{
    emit zeroingVout(false);

    // Remove existing offset - output /should/ already be zeroed from zeroing V1
    p.vclampChan.offset = 0;
    session.setDAQData(p);

    // Initialise
    DAQFilter daq(session);
    daq.VC = true;
    Stimulation stim;
    stim.duration = 1000;
    stim.baseV = 0; // Offset-free true zero
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
