#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include "populationsaver.h"

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
    lib.assignment = lib.assignment_base | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_PREVTHREAD | ASSIGNMENT_SUMMARY_SQUARED | ASSIGNMENT_SUMMARY_ERRFN;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_CURRENTCLAMP;

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

void GAFitter::cl_stimulate(QFile &file, int stimIdx)
{
    const RunData &rd = session.runData();
    const Stimulation &aI = astims[targetStim];
    iStimulation I = stims[targetStim];

    QString epoch_file = QString("%1.%2.stim_%3.trace").arg(file.fileName()).arg(epoch).arg(stimIdx);
    std::ofstream os(epoch_file.toStdString());

    // Set up library
    lib.setSingularStim();
    lib.stim[0] = I;
    lib.push(lib.stim);

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
    }
    for ( int iT = 0; iT < I.duration; iT++ ) {
        daq->next();
        scalar t = rd.settleDuration + iT*rd.dt;
        scalar command = getCommandVoltage(aI, iT*rd.dt);
        pushToQ(qT + t, daq->voltage, daq->current, command);
        os << t << '\t' << command << '\t' << daq->voltage << '\n';
        lib.target[iT] = daq->voltage;
    }
    daq->reset();

    lib.pushTarget();
    lib.run();

    qT += rd.settleDuration + I.duration * rd.dt;
}
