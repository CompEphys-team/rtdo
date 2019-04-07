#include "gafitter.h"
#include "session.h"
#include "supportcode.h"

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

bool GAFitter::cl_exec(Result *res, QFile &file)
{
    if ( settings.mutationSelectivity != 0 ) {
        std::cerr << "Error: Use non-selective mutation scheme for closed-loop experiments." << std::endl;
        return false;
    }

    {
        QMutexLocker locker(&mutex);
        doFinish = false;
        aborted = false;
    }

    output = std::move(*static_cast<Output*>(res));
    delete res;

    emit starting();

    astims = output.stimSource.stimulations();

    daq = new DAQFilter(session, session.getSettings());

    setup(true);

    populate();

    cl_fit(file);

    return true;
}

void GAFitter::cl_fit(QFile &file)
{
    std::vector<iStimulation> selectedStims;

    // Set up complete observation once (saves doing it over and over in cl_findStims())
    lib.setSingularStim(false);
    for ( int i = 0; i < settings.cl_nStims; i++ )
        lib.obs[i] = obs[0];
    stims[0].baseV = session.stimulationData().baseV;

    for ( epoch = 0; !finished(); epoch++ ) {
        cl_settle();

        // Find some stimulations
        selectedStims = cl_findStims();

        int i = 0;
        for ( iStimulation &pick : selectedStims ) {
            stims[0] = pick;
            astims[0] = Stimulation(pick, session.runData().dt);

            cl_stimulate(i++);
        }

        // Advance
        lib.pullSummary();
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

std::vector<iStimulation> GAFitter::cl_findStims()
{
    lib.cl_blocksize = lib.NMODELS / settings.cl_nStims;

    const RunData &rd = session.runData();
    iStimData istimd(session.stimulationData(), rd.dt);

    // Set up library
    lib.assignment = lib.assignment_base | ASSIGNMENT_REPORT_FIRST_SPIKE;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_CURRENTCLAMP;

    // Put in some random stims
    lib.setSingularStim(false);
    for ( int i = 0; i < settings.cl_nStims; i++ ) {
        lib.stim[i] = session.wavegen().getRandomStim(session.stimulationData(), istimd);
    }
    lib.push(lib.stim);

    // Target spike time is 0:
    lib.target[0] = 0;
    lib.pushTarget(-1, 1);

    lib.run();
    lib.pullSummary();

    errTupel foo;
    foo.err = 0;
    std::vector<errTupel> variance(settings.cl_nStims, foo);
    for ( int stimIdx = 0; stimIdx < settings.cl_nStims; stimIdx++ ) {
        double mean = 0;
        for ( int i = 0; i < lib.cl_blocksize; i++ )
            mean += lib.summary[stimIdx*lib.cl_blocksize + i];
        mean /= lib.cl_blocksize;

        for ( int i = 0; i < lib.cl_blocksize; i++ ) {
            double delta = mean - lib.summary[stimIdx*lib.cl_blocksize + i];
            variance[stimIdx].err += delta*delta;
        }
        variance[stimIdx].err /= lib.cl_blocksize;
        variance[stimIdx].idx = stimIdx;
    }

    // Pick the nSelect highest variance stimulations to return. Note, ascending sort
    std::sort(variance.begin(), variance.end(), &errTupelSort);
    std::vector<iStimulation> ret(settings.cl_nSelect);
    for ( int i = 0; i < settings.cl_nSelect; i++ ) {
        ret[i] = lib.stim[variance[settings.cl_nStims - i - 1].idx];
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
        if ( daq->voltage > -10. ) {
            lib.target[0] = iT*rd.dt;
            break;
        }
    }
    daq->reset();

    // Stimulate lib - no settling, this is done externally
    lib.pushTarget(-1, 1);
    lib.run();

    qT += rd.settleDuration + I.duration * rd.dt;
}
