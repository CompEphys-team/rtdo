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
    Settings settings = session.qSettings();
    settings.rund.VC = true;
    DAQFilter daq(session, settings);
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
    Settings settings = session.qSettings();
    settings.rund.VC = false;
    DAQFilter daq(session, settings);
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
    Settings settings = session.qSettings();
    settings.rund.VC = false;
    DAQFilter daq(session, settings);
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
    Settings settings = session.qSettings();
    settings.rund.VC = true;
    DAQFilter daq(session, settings);
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

void Calibrator::findAccessResistance()
{
    emit findingAccessResistance(false);

    Settings settings = session.qSettings();
    settings.rund.VC = false;
    DAQFilter daq(session, settings);
    Stimulation stim;
    stim.duration = 2000;
    stim.baseV = 0;
    stim.clear();
    int j = 0;
    for ( int i : {1, 0, -1, 0, 1, 0, -1} )
        stim.insert(stim.end(), Stimulation::Step {250 * (++j), i, false});

    double base = 0, step = 0;
    daq.run(stim);
    int samplesPerStep = daq.samplesRemaining / 8;
    int margin = samplesPerStep/10;
    int stepNo = 0;
    while ( daq.samplesRemaining > 0 ) {
        daq.next();
        int sampleNo = daq.samplesRemaining % samplesPerStep;
        if ( sampleNo == 0 )
            ++stepNo;
        if ( sampleNo > margin && sampleNo < samplesPerStep-margin ) {
            if ( stepNo % 2 == 0 ) // out: 0
                base += daq.voltage_2;
            else if ( stepNo % 4 == 1 ) // out: 1
                step += daq.voltage_2;
            else // out: -1
                step -= daq.voltage_2;
        }
    }

    base /= 4 * (samplesPerStep - 2*margin);
    step /= 4 * (samplesPerStep - 2*margin);

    RunData rd = session.qRunData();
    rd.accessResistance = step - base;
    // Note, R [MOhm] = V [mV] / I [nA], with I = 1 nA
    session.setRunData(rd);

    emit findingAccessResistance(true);
}
