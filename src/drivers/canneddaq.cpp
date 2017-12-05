#include "canneddaq.h"
#include <iostream>
#include <fstream>
#include "session.h"

int CannedDAQ::Iidx = 0;
int CannedDAQ::Vidx = -1;
int CannedDAQ::V2idx = -1;

double CannedDAQ::Iscale = 1;
double CannedDAQ::Vscale = 1;
double CannedDAQ::V2scale = 1;

CannedDAQ::CannedDAQ(Session &s) :
    DAQ(s)
{

}

void CannedDAQ::setRecord(std::vector<Stimulation> stims, QString record, bool readData, bool useQueuedSettings)
{
    std::ifstream is(record.toStdString());
    std::string str;
    is >> str;
    if ( str != "ATF" ) {
        std::cerr << "Invalid file (start should be 'ATF', not '" << str << "'" << std::endl;
        return;
    }
    std::getline(is, str);
    int nHeaderLines, nColumns, nChannels, nSweeps = stims.size();
    char cr;
    is >> nHeaderLines >> nColumns >> cr;
    nChannels = (nColumns-1) / nSweeps;

    // First column is time. Subsequent columns should be n channels by m sweeps, clustered by channel.
    if ( nColumns <= 1 || (nColumns-1) % nSweeps != 0 ) {
        std::cerr << "Wrong number of columns: " << nColumns << ", expected 1 + nChannels*" << nSweeps << std::endl;
        return;
    }

    for ( int i = 0; i < nHeaderLines; i++ )
        std::getline(is, str);

    channelNames.resize(nChannels);
    QuotedString qstr;
    is >> qstr; // Time
    for ( int i = 0; i < nChannels; i++ )
        is >> channelNames[i];
    std::getline(is, str);

    if ( readData && (Iidx >= 0 || Vidx >= 0 || V2idx >= 0) ) {
        int recStart = records.size();
        int nRead = 0, nTotal;
        nTotal = prepareRecords(stims, useQueuedSettings);
        std::vector<double> samples(nChannels);
        double t;
        is >> t;
        while ( is.good() ) {
            ++nRead;
            for ( int i = 0, r = recStart; i < nSweeps; i++, r++ ) {
                for ( double &sample : samples )
                    is >> sample;
                if ( Iidx >= 0 )    records[r].I.push_back(samples[Iidx] * Iscale);
                if ( Vidx >= 0 )    records[r].V.push_back(samples[Vidx] * Vscale);
                if ( V2idx >= 0 )   records[r].V2.push_back(samples[V2idx] * V2scale);
            }
            is >> t; // advance to next line
        }

        if ( nRead != nTotal ) {
            std::cerr << "Warning: Data size mismatched. Expected " << nTotal << " samples, have " << nRead << std::endl;
            for ( int r = recStart; r < recStart+nSweeps; r++ )
                records[r].nTotal = nRead;
        }
    }
}

void CannedDAQ::getSampleNumbers(const std::vector<Stimulation> &stims, double dt, int *nTotal, int *nBuffer, int *nSamples)
{
    scalar duration = 0;
    for ( const Stimulation &stim : stims )
        duration = std::max(duration, stim.tObsEnd);
    int _nSamples = duration / dt;
    int _nBuffer = _nSamples/62;
    int _nTotal = _nSamples + 2*_nBuffer;

    if ( nTotal )   *nTotal   = _nTotal;
    if ( nSamples ) *nSamples = _nSamples;
    if ( nBuffer )  *nBuffer  = _nBuffer;
}

int CannedDAQ::prepareRecords(std::vector<Stimulation> stims, bool useQueuedSettings)
{
    int nTotal, nBuffer;
    const DAQData &dd = useQueuedSettings ? session.qDaqData() : p;
    double dt = session.project.dt();
    if ( dd.filter.active )
        dt /= dd.filter.samplesPerDt;
    getSampleNumbers(stims, dt, &nTotal, &nBuffer);
    for ( size_t i = 0; i < stims.size(); i++ ) {
        Record rec;
        rec.stim = stims[i];
        rec.nBuffer = nBuffer;
        rec.nTotal = nTotal;
        if ( Iidx >= 0 )    rec.I.reserve(nTotal);
        if ( Vidx >= 0 )    rec.V.reserve(nTotal);
        if ( V2idx >= 0 )   rec.V2.reserve(nTotal);
        records.push_back(rec);
    }

    if ( nBuffer < dd.filter.width/2 )
        std::cerr << "Warning: Filter width exceeds available data by " << 2*(dd.filter.width/2 - nBuffer) << " points." << std::endl;

    return nTotal;
}

void CannedDAQ::run(Stimulation s)
{
    currentStim = s;
    for ( size_t i = 0; i < records.size(); i++ ) {
        if ( s == records[i].stim ) {
            currentRecord = i;
            reset();
            return;
        }
    }
    std::cerr << "Warning: No record found for stimulation: " << s << std::endl;
}

void CannedDAQ::next()
{
    int idx = recordIndex++;
    if ( idx < 0 )
        idx = 0;
    else if ( idx >= records[currentRecord].nTotal )
        idx = records[currentRecord].nTotal-1;

    if ( records[currentRecord].I.empty() )
        current = 0;
    else
        current = records[currentRecord].I[idx];

    if ( records[currentRecord].V.empty() )
        voltage = 0;
    else
        voltage = records[currentRecord].V[idx];

    if ( records[currentRecord].V2.empty() )
        voltage_2 = 0;
    else
        voltage_2 = records[currentRecord].V2[idx];
}

void CannedDAQ::reset()
{
    if ( p.filter.active ) {
        recordIndex = records[currentRecord].nBuffer - int(p.filter.width/2);
    } else {
        recordIndex = records[currentRecord].nBuffer;
    }
}
