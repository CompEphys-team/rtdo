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


#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include "populationsaver.h"
#include <QDataStream>

const QString GAFitter::validate_action = QString("validate");
const quint32 GAFitter::validate_magic = 0xaa3e9ef0;
const quint32 GAFitter::validate_version = 101;

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

std::vector<std::vector<double>> GAFitter::load_validation(QFile &base, int ep)
{
    std::vector<std::vector<double>> traces;

    QFile traceFile(QString("%1.validation.ep_%2.bin").arg(base.fileName()).arg(ep, 4, 10, QChar('0')));
    if ( !traceFile.exists() )
        traceFile.setFileName(QString("%1.validation.ep_%2.bin").arg(base.fileName()).arg(ep, 2, 10, QChar('0'))); // Legacy
    if ( !traceFile.open(QIODevice::ReadOnly) ) {
        std::cerr << "Failed to open file " << traceFile.fileName().toStdString() << " for reading." << std::endl;
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

void GAFitter::validate(size_t fitIdx)
{
    Validation *tmp = new Validation;
    tmp->fitIdx = fitIdx;
    session.queue(actorName(), validate_action, QString("Fit %1").arg(fitIdx), tmp);
}

bool GAFitter::exec_validation(Result *res, QFile &file)
{
    Validation val = std::move(*static_cast<Validation*>(res));
    delete res;

    const Output &ancestor = m_results[val.fitIdx];
    Settings cfg = session.getSettings(ancestor.resultIndex);
    QFile basefile(session.resultFilePath(ancestor.resultIndex, actorName(), ancestor.closedLoop ? cl_action : action));
    PopLoader population(basefile, lib);
    if ( !population.nEpochs )
        return false;

    {
        QMutexLocker locker(&mutex);
        doFinish = false;
        aborted = false;
        running = true;
    }

    emit starting();

    std::ofstream ascii(QString("%1.tsv").arg(file.fileName()).toStdString());

    stims = ancestor.stimSource.iStimulations(cfg.rund.dt);

    lib.setSingularRund();
    lib.simCycles = session.runData().simCycles;
    lib.integrator = session.runData().integrator;
    lib.setRundata(0, session.runData());
    lib.setSingularStim();
    lib.obs[0] = {{}, {}};
    lib.setSingularTarget();
    lib.summaryOffset = 0;

    lib.SDF_size = settings.SDF_size;
    lib.SDF_decay = settings.SDF_decay;

    const unsigned int assignment = lib.assignment_base
            | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_SQUARED | ASSIGNMENT_SUMMARY_COMPARE_TARGET
            | (ancestor.closedLoop ? ASSIGNMENT_SUMMARY_ERRFN : 0)
            | (cfg.rund.VC ? 0 : ASSIGNMENT_CURRENTCLAMP);

    std::vector<std::vector<double>> traces;
    val.error.resize(ancestor.epochs);
    val.mean.resize(ancestor.epochs);
    val.sd.resize(ancestor.epochs);

    int nSamples = 0;
    for ( const iStimulation &I : stims )
        nSamples += I.duration;
    lib.resizeTarget(1, nSamples);

    for ( epoch = 0; epoch < ancestor.epochs && population.load(epoch, lib); epoch++ ) {
        if ( epoch % cfg.gafs.cl_validation_interval == 0 ) {
            int nSamplesDone = 0;
            traces = load_validation(basefile, epoch);
            for ( size_t i = 0; i < stims.size(); i++ ) {
                for ( int j = 0; j < stims[i].duration; j++ )
                    lib.target[nSamplesDone + j] = traces[i][j];
                nSamplesDone += stims[i].duration;
            }
            lib.pushTarget();
        }

        lib.push();

        lib.targetOffset[0] = 0;
        for ( size_t i = 0; i < stims.size(); i++ ) {
            lib.stim[0] = stims[i];
            lib.obs[0].stop[0] = stims[i].duration;
            lib.push(lib.stim);
            lib.push(lib.obs);

            lib.push(lib.targetOffset);
            lib.targetOffset[0] += stims[i].duration;

            lib.assignment = assignment | (i ? ASSIGNMENT_SUMMARY_PERSIST : 0);
            lib.run();
        }

        val.error[epoch].resize(lib.NMODELS);
        lib.pullSummary();
        for ( size_t i = 0; i < lib.NMODELS; i++ ) {
            double rmse = std::sqrt(lib.summary[i]);
            val.error[epoch][i] = rmse;
            val.mean[epoch] += rmse;
            ascii << rmse << '\t';
        }
        val.mean[epoch] /= lib.NMODELS;

        for ( size_t i = 0; i < lib.NMODELS; i++ ) {
            double diff = val.mean[epoch] - std::sqrt(lib.summary[i]);
            val.sd[epoch] += diff*diff;
        }
        val.sd[epoch] = std::sqrt(val.sd[epoch] / lib.NMODELS);

        std::cout << epoch << '\t' << val.mean[epoch] << '\t' << val.sd[epoch] << std::endl;
        ascii << '\n';
    }

    {
        QMutexLocker locker(&mutex);
        m_validations.push_back(std::move(val));
        running = false;
    }
    emit done();

    // Save
    save_validation_result(file);

    return true;
}

void GAFitter::save_validation_result(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, validate_magic, validate_version) )
        return;
    const Validation &val = m_validations.back();
    os << val.error << val.mean << val.sd << qint32(val.fitIdx);
}

Result *GAFitter::load_validation_result(Result r, QFile &results, const QString &args)
{
    QDataStream is;
    quint32 ver = openLoadStream(results, is, validate_magic);
    if ( ver < 100 || ver > validate_version )
        throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

    Validation *p_out;
    if ( r.dryrun ) {
        p_out = new Validation(r);
    } else {
        m_validations.emplace_back(r);
        p_out =& m_validations.back();
    }
    Validation &val = *p_out;

    is >> val.error >> val.mean >> val.sd;

    if ( ver < 101 ) {
        val.fitIdx = args.split(' ').back().toInt();
    } else {
        qint32 fitIdx;
        is >> fitIdx;
        val.fitIdx = fitIdx;
    }

    return p_out;
}
