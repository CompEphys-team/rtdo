#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include <QDataStream>

void GAFitter::record_validation(QFile &base)
{
    const RunData &rd = session.runData();

    std::vector<std::vector<double>> traces;
    int maxDuration = 0;

    std::vector<iStimulation> val_iStims = output.stimSource.iStimulations(rd.dt);
    std::vector<Stimulation> val_aStims = output.stimSource.stimulations();
    for ( size_t i = 0; i < val_iStims.size(); i++ ) {
        const Stimulation &aI = val_aStims[i];
        const iStimulation &I = val_iStims[i];

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
            traces[i][iT] = rd.VC ? daq->current : daq->voltage;
        }
        daq->reset();

        qT += rd.settleDuration + I.duration * rd.dt;

        if ( maxDuration < I.duration )
            maxDuration = I.duration;
    }

    QFile traceFile(QString("%1.validation.ep_%2.bin").arg(base.fileName()).arg(epoch, 2, 10, QChar('0')));
    if ( !traceFile.open(QIODevice::WriteOnly) ) {
        std::cerr << "Failed to open file " << traceFile.fileName().toStdString() << " for writing.";
        return;
    }
    QDataStream os(&traceFile);

    for ( int iT = 0; iT < maxDuration; iT++ ) {
        for ( const std::vector<double> &trace : traces ) {
            if ( iT < int(trace.size()) )
                os << trace[iT];
            else
                os << double(0);
        }
    }
}

std::vector<std::vector<double>> load_validation(QFile &base, int ep, int nStims, int maxDuration)
{
    std::vector<std::vector<double>> traces(nStims, std::vector<double>(maxDuration));

    QFile traceFile(QString("%1.validation.ep_%2.bin").arg(base.fileName()).arg(ep, 4, 10, QChar('0')));
    if ( !traceFile.open(QIODevice::ReadOnly) ) {
        std::cerr << "Failed to open file " << traceFile.fileName().toStdString() << " for reading.";
        return traces;
    }
    QDataStream is(&traceFile);

    for ( int iT = 0; iT < maxDuration; iT++ ) {
        for ( std::vector<double> &trace : traces ) {
            is >> trace[iT];
        }
    }
    return traces;
}
