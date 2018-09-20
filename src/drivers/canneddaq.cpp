#include "canneddaq.h"
#include <iostream>
#include <fstream>
#include "session.h"
#include <QFileInfo>

CannedDAQ::CannedDAQ(Session &s, const Settings &settings) :
    DAQ(s, settings),
    currentRecord(0),
    settleDuration(0)
{

}

double CannedDAQ::getAdjustableParam(size_t idx)
{
    return ref_params[idx];
}

void CannedDAQ::setRecord(std::vector<Stimulation> stims, QString record, bool readData)
{
    QFileInfo finfo(record);
    bool backcompat_noise = finfo.baseName().startsWith("2017");

    std::ifstream is(record.toStdString());
    std::string str;
    is >> str;
    if ( str != "ATF" ) {
        std::cerr << "Invalid file (start should be 'ATF', not '" << str << "'" << std::endl;
        return;
    }
    std::getline(is, str);
    int nHeaderLines, nColumns, nChannels, nSweeps = stims.size() - backcompat_noise;
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

    if ( readData && (assoc.Iidx >= 0 || assoc.Vidx >= 0 || assoc.V2idx >= 0) ) {
        int recStart = records.size();
        int nRead = 0, nTotal;
        nTotal = prepareRecords(stims);
        std::vector<double> samples(nChannels);
        double t;
        is >> t;
        while ( is.good() ) {
            ++nRead;
            for ( int i = 0, r = recStart; i < nSweeps; i++, r++ ) {
                for ( double &sample : samples )
                    is >> sample;
                if ( assoc.Iidx >= 0 )    records[r].I.push_back(samples[assoc.Iidx] * assoc.Iscale);
                if ( assoc.Vidx >= 0 )    records[r].V.push_back(samples[assoc.Vidx] * assoc.Vscale);
                if ( assoc.V2idx >= 0 )   records[r].V2.push_back(samples[assoc.V2idx] * assoc.V2scale);
            }
            is >> t; // advance to next line
        }

        if ( nRead != nTotal ) {
            std::cerr << "Warning: Data size mismatched. Expected " << nTotal << " samples, have " << nRead << std::endl;
            std::cerr << "Expected buffer: " << records[recStart].nBuffer << " lines; likely buffer in file: " << int(nRead/64) << std::endl;
            for ( int r = recStart; r < recStart+nSweeps; r++ )
                records[r].nTotal = nRead;
        }

        // 2017 recordings did not have a noise sample. Use initial buffers and any flat initial stim segments to replace this shortfall.
        if ( backcompat_noise ) {
            Record &compat = records.back();
            std::vector<Record> sortedRecs = records;
            sortedRecs.pop_back(); // Remove the noise-associated, but not yet initialised record
            // Sort stimulations by length of undisturbed initial segment, descending
            std::sort(sortedRecs.begin(), sortedRecs.end(), [](const Record &a, const Record &b){
                if ( a.stim.begin()->ramp )
                    return false;
                if ( b.stim.begin()->ramp )
                    return true;
                return a.stim.begin()->t > b.stim.begin()->t;
            });

            int i, iMax = compat.nTotal - compat.nBuffer;
            for ( i = 0; i < compat.nBuffer && i < iMax; i++ )
                compat.I.push_back(0);

            while ( i < iMax ) {
                for ( const Record &rec: sortedRecs ) {
                    int jMax = rec.nBuffer + (rec.stim.begin()->ramp ? 0 : rec.stim.begin()->t/samplingDt());
                    for ( int j = 0; j < jMax && i < iMax; i++, j++ ) {
                        compat.I.push_back(rec.I[j]);
                    }
                }
            }
        }
    }

    getReferenceParams(record);
}

void CannedDAQ::getReferenceParams(QString record)
{
    ref_params.resize(project.model().adjustableParams.size());
    for ( size_t i = 0; i < ref_params.size(); i++ )
        ref_params[i] = project.model().adjustableParams[i].initial;

    QString refname = record.replace(".atf", ".params");
    if ( !QFileInfo(refname).exists() )
        return;

    std::ifstream is(refname.toStdString());
    while ( is.good() ) {
        std::string name, unit;
        double value;
        is >> name >> value >> unit;
        for ( size_t i = 0; i < ref_params.size(); i++ ) {
            if ( name == project.model().adjustableParams[i].name + ':' ) {
                ref_params[i] = value;
                break;
            }
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

int CannedDAQ::prepareRecords(std::vector<Stimulation> stims)
{
    int nTotal, nBuffer;
    double dt = rund.dt;
    if ( p.filter.active )
        dt /= p.filter.samplesPerDt;
    getSampleNumbers(stims, dt, &nTotal, &nBuffer);
    for ( size_t i = 0; i < stims.size(); i++ ) {
        Record rec;
        rec.stim = stims[i];
        rec.nBuffer = nBuffer;
        rec.nTotal = nTotal;
        if ( assoc.Iidx >= 0 )    rec.I.reserve(nTotal);
        if ( assoc.Vidx >= 0 )    rec.V.reserve(nTotal);
        if ( assoc.V2idx >= 0 )   rec.V2.reserve(nTotal);
        records.push_back(rec);
    }

    if ( nBuffer < p.filter.width/2 )
        std::cerr << "Warning: Filter width exceeds available data by " << 2*(p.filter.width/2 - nBuffer) << " points." << std::endl;

    return nTotal;
}

void CannedDAQ::run(Stimulation s, double settleDuration)
{
    currentStim = s;
    this->settleDuration = settleDuration;
    for ( currentRecord = 0; currentRecord < records.size(); currentRecord++ ) {
        if ( s == records[currentRecord].stim ) {
            reset();
            return;
        }
    }
    std::cerr << "Warning: No record found for stimulation: " << s << ". Defaulting to record 0." << std::endl;
    currentRecord = 0;
    reset();
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

    --samplesRemaining;
}

void CannedDAQ::reset()
{
    recordIndex = records[currentRecord].nBuffer;
    if ( p.filter.active )
        recordIndex -= int(p.filter.width/2);
    if ( settleDuration > 0 )
        recordIndex -= settleDuration / samplingDt();

    samplesRemaining = records[currentRecord].nTotal - recordIndex - records[currentRecord].nBuffer;
    outputResolution = samplingDt();
}
