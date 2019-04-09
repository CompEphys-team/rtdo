#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include <QTextStream>

void GAFitter::cl_run()
{
    WaveSource src;
    std::vector<WaveSource> allSources = session.wavesets().sources();
    if ( allSources.size() ) {
        src = allSources[0];
        src.waveno = 0;
    } else {
        WavesetCreator &creator = session.wavesets();
        session.queue(creator.actorName(), creator.actionManual, "ClosedLoop dummy", new ManualWaveset({iStimulation()}, {iObservations()}));
        src = WaveSource(session, WaveSource::Manual, 0, 0);
    }
    session.queue(actorName(), cl_action, QString(), new Output(session, src, QString("")));
}

void GAFitter::cl_resume(size_t fitIdx)
{
    if ( fitIdx >= results().size() )
        return;
    Output *out = new Output(m_results[fitIdx]);
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

    astims = output.stimSource.stimulations(); // Required for setup only

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
    output.resume.DEMethodUsed = DEMethodUsed;
    output.resume.DEMethodSuccess = DEMethodSuccess;
    output.resume.DEMethodFailed = DEMethodFailed;
    output.resume.DEpX = DEpX;

    // Finish
    output.epochs = epoch;
    {
        QMutexLocker locker(&mutex);
        m_results.push_back(std::move(output));
        running = false;
    }
    emit done();

    // Save
    save(file);

    delete daq;
    daq = nullptr;

    return true;
}

void GAFitter::cl_fit(QFile &file)
{
    // Set up complete observation once (saves doing it over and over in cl_findStims())
    lib.setSingularStim(false);
    for ( int i = 0; i < settings.cl_nStims; i++ )
        lib.obs[i] = obs[0];
    lib.push(lib.obs);
    stims[0].baseV = session.stimulationData().baseV;

    for ( epoch = 0; !finished(); epoch++ ) {
        cl_settle();

        if ( finished() )
            break;

        // Find some stimulations
        std::vector<iStimulation> selectedStims = cl_findStims(file);

        int i = 0;
        for ( iStimulation &pick : selectedStims ) {
            stims[0] = pick;
            astims[0] = Stimulation(pick, session.runData().dt);

            cl_stimulate(i++);

            if ( finished() )
                break;
        }

        // Advance
        lib.pullSummary();

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

        if ( settings.useDE )
            procreateDE();
        else
            procreate();

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
    lib.assignment = lib.assignment_base | ASSIGNMENT_REPORT_FIRST_SPIKE;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_CURRENTCLAMP;

    // Target spike time is 0:
    lib.target[0] = 0;
    lib.pushTarget(-1, 1);

    lib.run();
    lib.pullSummary();

    std::vector<errTupel> variance(settings.cl_nStims);
    for ( int stimIdx = 0; stimIdx < settings.cl_nStims; stimIdx++ ) {
        double mean = 0;
        for ( int i = 0; i < lib.cl_blocksize; i++ )
            mean += lib.summary[stimIdx*lib.cl_blocksize + i];
        mean /= lib.cl_blocksize;

        variance[stimIdx].idx = stimIdx;
        variance[stimIdx].err = 0;
        for ( int i = 0; i < lib.cl_blocksize; i++ ) {
            double delta = mean - lib.summary[stimIdx*lib.cl_blocksize + i];
            variance[stimIdx].err += delta*delta;
        }
        variance[stimIdx].err /= lib.cl_blocksize;

        os << stimIdx << '\t' << mean << '\t' << variance[stimIdx].err << '\n' << lib.stim[stimIdx] << '\n' << '\n';
    }

    // Pick the nSelect highest variance stimulations to return. Note, ascending sort
    std::sort(variance.begin(), variance.end(), &errTupelSort);
    std::vector<iStimulation> ret(settings.cl_nSelect);
    for ( int i = 0; i < settings.cl_nSelect; i++ ) {
        int stimIdx = variance[settings.cl_nStims - i - 1].idx;
        ret[i] = lib.stim[stimIdx];
        os << "Picked stim # " << stimIdx << '\n';
    }

    lib.cl_blocksize = lib.NMODELS;

    return ret;
}

void GAFitter::cl_stimulate(bool summaryPersist)
{
    const RunData &rd = session.runData();
    const Stimulation &aI = astims[targetStim];
    iStimulation I = stims[targetStim];

    // Set up library
    lib.setSingularStim();
    lib.stim[0] = I;
    lib.push(lib.stim);

//    lib.assignment = lib.assignment_base
//            | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_TARGET | ASSIGNMENT_SUMMARY_SQUARED;
//    if ( !rd.VC )
//        lib.assignment |= ASSIGNMENT_PATTERNCLAMP | ASSIGNMENT_PC_REPORT_PIN;

    lib.assignment = lib.assignment_base | ASSIGNMENT_REPORT_FIRST_SPIKE | ASSIGNMENT_SUMMARY_SQUARED;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_CURRENTCLAMP;
    if ( summaryPersist )
        lib.assignment |= ASSIGNMENT_SUMMARY_PERSIST;

    lib.target[0] = -1;

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
        pushToQ(qT + rd.settleDuration + iT*rd.dt, daq->voltage, daq->current, getCommandVoltage(aI, iT*rd.dt));
//        lib.target[iT] = rd.VC ? daq->current : daq->voltage;
        if ( daq->voltage > -10. && lib.target[0] < 0 ) {
            // Run lib to first spike
            lib.target[0] = iT*rd.dt;
            lib.pushTarget(-1, 1);
            lib.run();
        }
    }
    daq->reset();

    // No spike detected, assume first spike is at I.duration
    if ( lib.target[0] < 0 ) {
        lib.target[0] = aI.duration;
        lib.pushTarget(-1, 1);
        lib.run();
    }

    qT += rd.settleDuration + I.duration * rd.dt;
}