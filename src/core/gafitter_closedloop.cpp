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
#include <QtConcurrent/QtConcurrent>

void GAFitter::cl_run(WaveSource src)
{
    session.queue(actorName(), cl_action, QString(), new Output(session, src, QString("")));
}

void GAFitter::cl_resume(size_t fitIdx, WaveSource src)
{
    if ( fitIdx >= results().size() )
        return;
    Output *out = new Output(m_results[fitIdx]);
    out->stimSource = src;
    session.queue(actorName(), cl_action, QString("Resume %1").arg(m_results[fitIdx].resultIndex, 4, 10, QChar('0')), out);
}

bool GAFitter::cl_exec(Result *res, QFile &file)
{
    if ( settings.mutationSelectivity != 0 ) {
        std::cerr << "Error: Use non-selective mutation scheme for closed-loop experiments." << std::endl;
        return false;
    }

    output = std::move(*static_cast<Output*>(res));
    delete res;
    output.closedLoop = true;

    {
        QMutexLocker locker(&mutex);
        doFinish = false;
        aborted = false;
        running = true;
    }

    emit starting();

    qT = 0;

    astims.assign(1, {}); // Required for setup only

    daq = new DAQFilter(session, session.getSettings());

    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
        output.targets[i] = lib.adjustableParams[i].initial;

    setup(true);

    populate();

    cl_fit(file);

    // Resumability
    output.resume.population.assign(lib.adjustableParams.size(), std::vector<scalar>(lib.NMODELS));
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
        for ( size_t j = 0; j < lib.NMODELS; j++ )
            output.resume.population[i][j] = lib.adjustableParams[i][j];
    output.resume.bias = bias;
    output.resume.DEMethodSuccess = DEMethodSuccess;
    output.resume.DEMethodFailed = DEMethodFailed;
    output.resume.DEpX = DEpX;

    // Save
    output.epochs = epoch;
    save(file);

    {
        QMutexLocker locker(&mutex);
        m_results.push_back(std::move(output));
        running = false;
    }
    emit done();

    delete daq;
    daq = nullptr;

    return true;
}

void GAFitter::cl_fit(QFile &file)
{
    if ( settings.num_populations > 1 )
        std::cerr << "Warning: Multi-population closed loop fitting is not supported and may behave in unexpected ways." << std::endl;

    // Set up complete observation once (saves doing it over and over in cl_findStims())
    lib.setSingularStim(false);
    for ( int i = 0; i < settings.cl_nStims; i++ )
        lib.obs[i] = obs[0];
    lib.push(lib.obs);
    lib.resizeOutput(obs[0].duration());

    lib.SDF_size = settings.SDF_size;
    lib.SDF_decay = settings.SDF_decay;

    PopSaver pop(file);

    for ( epoch = 0; !finished(); epoch++ ) {
        cl_settle();

        if ( epoch % settings.cl_validation_interval == 0 )
            record_validation(file);
        pop.savePop(lib);

        if ( finished() )
            break;

        // Find some stimulations
        std::vector<iStimulation> selectedStims = cl_findStims(file);

        int i = 0;
        for ( iStimulation &pick : selectedStims ) {
            stims[0] = pick;
            astims[0] = Stimulation(pick, session.runData().dt);

            cl_stimulate(file, i++);

            if ( finished() )
                break;
        }

        // Advance
        lib.pullSummary();
        pop.saveErr(lib);

        {
            QString epoch_file = QString("%1.%2.models").arg(file.fileName()).arg(epoch);
            std::ofstream os(epoch_file.toStdString());
            os << "idx\terr";
            for ( const AdjustableParam &p : lib.adjustableParams )
                os << '\t' << p.name;
            os << '\n';
            for ( size_t i = 0; i < lib.NMODELS; i++ ) {
                os << i << '\t' << lib.summary[i];
                for ( const AdjustableParam &p : lib.adjustableParams )
                    os << '\t' << p[i];
                os << '\n';
            }
        }

        if ( settings.useDE ) {
            procreateDE();
        } else {
            std::vector<errTupel> p_err = procreate();
            cl_relegate_reinitialised(p_err);
        }

        emit progress(epoch);
    }
}

void GAFitter::cl_settle()
{
    const RunData &rd = session.runData();

    lib.setSingularRund();
    lib.simCycles = rd.simCycles;
    lib.integrator = rd.integrator;
    lib.setRundata(0, rd);

    lib.setSingularTarget();
    lib.resizeTarget(1, iStimData(session.stimulationData(), rd.dt).iDuration);
    lib.targetOffset[0] = 0;

    lib.setSingularStim();
    lib.stim[0] = stims[0];
    lib.obs[0] = obs[0];

    lib.assignment = lib.assignment_base | ASSIGNMENT_SETTLE_ONLY | ASSIGNMENT_MAINTAIN_STATE;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_CURRENTCLAMP;

    lib.summaryOffset = 0;

    lib.push();
    lib.run();
}

std::vector<iStimulation> GAFitter::cl_findStims(QFile &base)
{
    lib.cl_blocksize = lib.NMODELS / settings.cl_nStims;

    QString epoch_file = QString("%1.%2.stims").arg(base.fileName()).arg(epoch);
    std::ofstream os(epoch_file.toStdString());

    const RunData &rd = session.runData();
    iStimData istimd(session.stimulationData(), rd.dt);

    // Put in some random stims
    lib.setSingularStim(false);
    for ( int i = 0; i < settings.cl_nStims; i++ ) {
        lib.stim[i] = session.wavegen().getRandomStim(session.stimulationData(), istimd);
    }
    lib.push(lib.stim);

    // Set up library
    lib.iSettleDuration[0] = 0;
    lib.push(lib.iSettleDuration);
    lib.assignment = lib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_NONE | ASSIGNMENT_SUBSET_MUX;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_CURRENTCLAMP;

    lib.run();
    std::vector<scalar> mean_cost = lib.cl_get_mean_cost(settings.cl_nStims, lib.stim[0].duration, settings.spike_threshold * session.runData().dt, 0.99, settings.SDF_size, settings.SDF_decay);

    std::vector<errTupel> cost(settings.cl_nStims);
    for ( int stimIdx = 0; stimIdx < settings.cl_nStims; stimIdx++ ) {
        cost[stimIdx].idx = stimIdx;
        cost[stimIdx].err = mean_cost[stimIdx];
        os << stimIdx << '\t' << mean_cost[stimIdx] << '\n' << lib.stim[stimIdx] << '\n' << '\n';
    }

    // Pick the nSelect highest variance stimulations to return. Note, ascending sort
    std::sort(cost.begin(), cost.end(), &errTupelSort);
    std::vector<iStimulation> ret(settings.cl_nSelect);
    for ( int i = 0; i < settings.cl_nSelect; i++ ) {
        int stimIdx = cost[settings.cl_nStims - i - 1].idx;
        ret[i] = lib.stim[stimIdx];
        os << "Picked stim # " << stimIdx << '\n';
    }

    lib.cl_blocksize = lib.NMODELS;

    return ret;
}

void GAFitter::cl_stimulate(QFile &file, int stimIdx)
{
    QFuture<void> pca_future;
    if ( stimIdx == 0 )
        pca_future = QtConcurrent::run(this, &GAFitter::cl_pca);

    const RunData &rd = session.runData();
    const Stimulation &aI = astims[targetStim];
    iStimulation I = stims[targetStim];

    QString epoch_file = QString("%1.%2.stim_%3.trace").arg(file.fileName()).arg(epoch).arg(stimIdx);
    std::ofstream os(epoch_file.toStdString());

    double dV = 0, Vprev = 0;
    constexpr scalar SDF_dV_DECAY = 0.99;
    scalar spike_threshold = settings.spike_threshold*rd.dt;
    lib.spike_threshold = spike_threshold;

    // Set up library
    lib.setSingularStim();
    lib.stim[0] = I;
    lib.push(lib.stim);

    lib.iSettleDuration[0] = 0;
    lib.push(lib.iSettleDuration);
    lib.assignment = lib.assignment_base | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_TARGET | ASSIGNMENT_SUMMARY_SQUARED | ASSIGNMENT_SUMMARY_ERRFN;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_CURRENTCLAMP;
    if ( stimIdx > 0 )
        lib.assignment |= ASSIGNMENT_SUMMARY_PERSIST;

    // Initiate DAQ stimulation
    daq->reset();
    daq->run(aI, rd.settleDuration);

    // Step DAQ through full stimulation
    for ( int iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ ) {
        daq->next();
        pushToQ(qT + iT*rd.dt, daq->voltage, daq->current, I.baseV);
        os << iT*rd.dt << '\t' << I.baseV << '\t' << daq->voltage << '\n';

        if ( qV2 )
            qV2->push({qT + iT*rd.dt, dV - spike_threshold});
        dV = dV*SDF_dV_DECAY + (daq->voltage - Vprev);
        Vprev = daq->voltage;
    }
    for ( int iT = 0; iT < I.duration; iT++ ) {
        daq->next();
        scalar t = rd.settleDuration + iT*rd.dt;
        scalar command = getCommandVoltage(aI, iT*rd.dt);
        pushToQ(qT + t, daq->voltage, daq->current, command);
        os << t << '\t' << command << '\t' << daq->voltage << '\n';
        lib.target[iT] = daq->voltage;

        if ( qV2 )
            qV2->push({qT + t, dV - spike_threshold});
        dV = dV*SDF_dV_DECAY + (daq->voltage - Vprev);
        Vprev = daq->voltage;
    }
    daq->reset();

    if ( stimIdx == 0 )
        pca_future.waitForFinished();

    lib.pushTarget();
    lib.run();

    qT += rd.settleDuration + I.duration * rd.dt;
}

void GAFitter::cl_pca()
{
    QTime wallclock = QTime::currentTime();
    std::vector<scalar> singular_values = lib.get_params_principal_components(2, settings.nReinit);
    for ( const scalar &s : singular_values )
        std::cout << s << '\t';
    lib.sync();
    std::cout << "\nCompleted in " << wallclock.msecsTo(QTime::currentTime()) << " ms" << std::endl;

    emit pca_complete();
}

void GAFitter::cl_relegate_reinitialised(std::vector<errTupel> &p_err)
{
    // Sort reinit subpopulation by idx, descending
    std::sort(p_err.end() - settings.nReinit, p_err.end(), [](const errTupel &lhs, const errTupel &rhs){ return lhs.idx < rhs.idx; });

    // Move subpop to end of population (tail first)
    using std::swap;
    size_t i = lib.NMODELS - 1;
    for ( auto it = p_err.rbegin(); it != p_err.rbegin() + settings.nReinit; ++it, --i ) {
        for ( AdjustableParam &p : lib.adjustableParams ) {
            swap(p[i], p[it->idx]);
        }
    }
}
