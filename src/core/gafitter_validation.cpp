#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include <QDataStream>

void GAFitter::record_validation(QFile &base)
{
    if ( output.stimSource.type == WaveSource::Empty )
        return;

    const RunData &rd = session.runData();

    QFile traceFile(QString("%1.validation.ep_%2.bin").arg(base.fileName()).arg(epoch, 4, 10, QChar('0')));
    if ( !traceFile.open(QIODevice::WriteOnly) ) {
        std::cerr << "Failed to open file " << traceFile.fileName().toStdString() << " for writing.";
        return;
    }
    QDataStream os(&traceFile);

    std::vector<iStimulation> val_iStims = output.stimSource.iStimulations(rd.dt);
    std::vector<Stimulation> val_aStims = output.stimSource.stimulations();
    for ( size_t i = 0; i < val_iStims.size(); i++ ) {
        const Stimulation &aI = val_aStims[i];
        const iStimulation &I = val_iStims[i];

        os << qint32(I.duration);

        // Initiate DAQ stimulation
        daq->reset();
        daq->run(aI, rd.settleDuration);

        // Step DAQ through full stimulation
        for ( int iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ ) {
            daq->next();
            pushToQ(qT + iT*rd.dt, daq->voltage, daq->current, I.baseV);
        }
        for ( int iT = 0; iT < I.duration; iT++ ) {
            daq->next();
            scalar t = rd.settleDuration + iT*rd.dt;
            scalar command = getCommandVoltage(aI, iT*rd.dt);
            pushToQ(qT + t, daq->voltage, daq->current, command);
            os << (rd.VC ? daq->current : daq->voltage);
        }
        daq->reset();

        qT += rd.settleDuration + I.duration * rd.dt;
    }
}

std::vector<std::vector<double>> load_validation(QFile &base, int ep)
{
    std::vector<std::vector<double>> traces;

    QFile traceFile(QString("%1.validation.ep_%2.bin").arg(base.fileName()).arg(ep, 4, 10, QChar('0')));
    if ( !traceFile.exists() )
        traceFile.setFileName(QString("%1.validation.ep_%2.bin").arg(base.fileName()).arg(ep, 2, 10, QChar('0'))); // Legacy
    if ( !traceFile.open(QIODevice::ReadOnly) ) {
        std::cerr << "Failed to open file " << traceFile.fileName().toStdString() << " for reading.";
        return traces;
    }
    QDataStream is(&traceFile);

    qint32 duration;
    int stimIdx = 0;

    while ( !is.atEnd() ) {
        is >> duration;
        traces.emplace_back(duration);
        for ( int i = 0; i < duration; i++ )
            is >> traces[stimIdx][i];
        ++stimIdx;
    }
    return traces;
}
