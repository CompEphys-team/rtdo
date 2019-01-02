#include "wavegen.h"
#include "session.h"

Wavegen::Wavegen(Session &session) :
    SessionWorker(session),
    searchd(session.wavegenData()),
    stimd(session.stimulationData()),
    ulib(session.project.universal())
{
}

const QString Wavegen::cluster_action = QString("cluster");
const QString Wavegen::bubble_action = QString("bubble");

void Wavegen::search(const QString &action)
{
    if ( action != cluster_action && action != bubble_action )
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
    session.queue(actorName(), action, "", new Result());
}

bool Wavegen::execute(QString action, QString, Result *result, QFile &file)
{
    clearAbort();
    istimd = iStimData(stimd, session.runData().dt);
    if ( action != cluster_action && action != bubble_action )
        return false;

    bool dryrun = result->dryrun;
    current = Archive(action, searchd, *result);
    delete result;
    result = nullptr;

    if ( dryrun ) {
        save(file, action);
        return true;
    }

    emit startedSearch(action);

    const int nModelsPerStim = searchd.trajectoryLength * searchd.nTrajectories;
    const int nStimsPerEpoch = ulib.NMODELS / nModelsPerStim;
    const int blankCycles = session.gaFitterSettings().cluster_blank_after_step / session.runData().dt;
    const int sectionLength = session.gaFitterSettings().cluster_fragment_dur / session.runData().dt;
    const scalar dotp_threshold = session.gaFitterSettings().cluster_threshold;
    const int minClusterLen = session.gaFitterSettings().cluster_min_dur / session.runData().dt;

    double run_maxCurrent = 0, sum_maxCurrents = 0;

    std::vector<MAPEDimension> dims = session.wavegenData().mapeDimensions;
    std::vector<size_t> variablePrecisionDims;
    for ( size_t i = 0; i < dims.size(); i++ ) {
        if ( dims[i].hasVariableResolution() )
            variablePrecisionDims.push_back(i);
        else if ( action == bubble_action && dims[i].func == MAPEDimension::Func::EE_ClusterIndex )
            std::cerr << "Warning: ClusterIndex dimension invalid for bubble search" << std::endl;
        else if ( action == bubble_action && dims[i].func == MAPEDimension::Func::EE_NumClusters )
            std::cerr << "Warning: NumClusters dimension invalid for bubble search" << std::endl;
    }

    ulib.resizeOutput(istimd.iDuration);
    prepare_EE_models();
    settle_EE_models();

    // Use the settling period to generate the first two sets of stims
    std::vector<iStimulation> waves_ep1(nStimsPerEpoch), waves_ep2(nStimsPerEpoch);
    for ( iStimulation &w : waves_ep1 )
        w = getRandomStim(stimd, istimd);
    for ( iStimulation &w : waves_ep2 )
        w = getRandomStim(stimd, istimd);
    size_t nInitialStims = 2*nStimsPerEpoch;
    std::vector<iStimulation> *returnedWaves = &waves_ep1, *newWaves = &waves_ep2;

    // Run a set of stims through a deltabar finding mission
    current.deltabar = getDeltabar();
    std::vector<double> deltabar = current.deltabar.toStdVector();

    std::function<void(void)> evaluate, pull;
    std::function<scalar(std::vector<iStimulation>&)> score_and_insert;
    if ( action == cluster_action ) {
        evaluate = [&](){ ulib.cluster(searchd.trajectoryLength, searchd.nTrajectories, istimd.iDuration,
                                       sectionLength, dotp_threshold, minClusterLen, deltabar, false); };
        pull = [&](){ ulib.pullClusters(nStimsPerEpoch); };
        score_and_insert = [&](std::vector<iStimulation> &waves){ return cluster_scoreAndInsert(waves, nStimsPerEpoch, dims); };
    } else if ( action == bubble_action ) {
        evaluate = [&](){ ulib.bubble(searchd.trajectoryLength, searchd.nTrajectories, istimd.iDuration,
                                      sectionLength, deltabar, false); };
        pull = [&](){ ulib.pullBubbles(nStimsPerEpoch); };
        score_and_insert = [&](std::vector<iStimulation> &waves){ return bubble_scoreAndInsert(waves, nStimsPerEpoch, dims); };
    }

    // Queue the first set of stims
    pushStimsAndObserve(*returnedWaves, nModelsPerStim, blankCycles);
    ulib.assignment = ulib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_NONE;
    ulib.run();
    evaluate();

    for ( size_t epoch = 0; epoch < searchd.maxIterations; epoch++ ) {
        // Copy completed bubbles for returnedWaves to host (sync)
        pull();

        bool done = (++current.iterations == searchd.maxIterations) || isAborted();

        // Push and run newWaves (async)
        if ( !done ) {
            pushStimsAndObserve(*newWaves, nModelsPerStim, blankCycles);
            ulib.run();
            evaluate();
        }

        // score and insert returnedWaves into archive
        scalar epoch_maxCurrent = score_and_insert(*returnedWaves);
        sum_maxCurrents += epoch_maxCurrent;
        if ( epoch_maxCurrent > run_maxCurrent )
            run_maxCurrent = epoch_maxCurrent;

        // Swap waves: After this, returnedWaves points at those in progress, and newWaves are ready for new adventures
        using std::swap;
        swap(newWaves, returnedWaves);

        // Aborting or complete: No further processing.
        if ( done )
            break;

        // Increase precision
        if ( current.precision < searchd.precisionIncreaseEpochs.size() &&
             current.iterations == searchd.precisionIncreaseEpochs[current.precision] ) {
            current.precision++;
            // Only rebin/resort if necessary
            if ( !variablePrecisionDims.empty() ) {
                for ( size_t i : variablePrecisionDims )
                    dims[i].resolution *= 2;
                for ( MAPElite &e : current.elites )
                    for ( size_t i : variablePrecisionDims )
                        e.bin[i] = dims.at(i).bin(e, session.runData().dt);
                current.elites.sort();
            }
        }

        emit searchTick(current.iterations);

        // Prepare the next set of stims
        if ( nInitialStims < searchd.nInitialWaves ) {
            for ( iStimulation &w : *newWaves )
                w = getRandomStim(stimd, istimd);
            nInitialStims += nStimsPerEpoch;
        } else {
            construct_next_generation(*newWaves);
        }
    }

    std::cout << "Overall maximum observed current: " << run_maxCurrent
              << " nA; average maximum across epochs: " << (sum_maxCurrents/current.iterations) << " nA."
              << std::endl;

    current.nCandidates.squeeze();
    current.nInsertions.squeeze();
    current.nReplacements.squeeze();
    current.nElites.squeeze();

    save(file, action);
    m_archives.push_back(std::move(current));

    emit done();

    return true;
}

void Wavegen::save(QFile &file, const QString &action)
{
    if ( action != cluster_action && action != bubble_action )
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());

    QDataStream os;
    if ( !openSaveStream(file, os, search_magic, search_version) )
        return;
    size_t maxObs = action==bubble_action ? 2 : iObservations::maxObs;

    os << quint32(current.precision);
    os << quint32(current.iterations);
    os << current.nCandidates << current.nInsertions << current.nReplacements << current.nElites;
    os << current.deltabar;

    os << quint32(current.elites.size());
    os << quint32(current.elites.front().bin.size());
    os << quint32(current.elites.front().deviations.size());
    os << quint32(maxObs);

    // Separate waves into unique and shared to reduce the number of lookups
    std::vector<iStimulation*> w_unique, w_shared;
    for ( MAPElite const& e : current.elites ) {
        os << e.fitness;
        for ( size_t b : e.bin )
            os << quint32(b);
        for ( scalar d : e.deviations )
            os << d;
        for ( size_t i = 0; i < maxObs; i++ )
            os << quint32(e.obs.start[i]) << quint32(e.obs.stop[i]);
        os << e.current;

        if ( e.wave.unique() ) {
            w_unique.push_back(e.wave.get());
            os << qint32(-w_unique.size());
        } else {
            size_t i;
            for ( i = 0; i < w_shared.size(); i++ )
                if ( e.wave.get() == w_shared[i] )
                    break;
            if ( i == w_shared.size() )
                w_shared.push_back(e.wave.get());
            os << qint32(i);
        }
    }
    os << quint32(w_unique.size());
    for ( const iStimulation *pstim : w_unique )
        os << *pstim;
    os << quint32(w_shared.size());
    for ( const iStimulation *pstim : w_shared )
        os << *pstim;
}

Result *Wavegen::load(const QString &action, const QString &, QFile &file, Result r)
{
    if ( action != cluster_action && action != bubble_action )
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());

    QDataStream is;
    quint32 version = openLoadStream(file, is, search_magic);
    if ( version < 100 || version > search_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());

    Archive *arch;
    if ( r.dryrun )
        arch = new Archive(action, searchd, r);
    else {
        m_archives.emplace_back(action, searchd, r);
        arch =& m_archives.back();
    }

    quint32 precision, iterations, archSize, nBins, nParams, maxObs;
    is >> precision >> iterations;
    arch->precision = precision;
    arch->iterations = iterations;
    is >> arch->nCandidates >> arch->nInsertions >> arch->nReplacements >> arch->nElites;
    is >> arch->deltabar;

    is >> archSize >> nBins >> nParams >> maxObs;
    arch->elites.resize(archSize);
    std::vector<qint32> stimIdx(archSize);
    auto idxIt = stimIdx.begin();
    quint32 tmp, start, stop;
    for ( auto el = arch->elites.begin(); el != arch->elites.end(); el++, idxIt++ ) {
        is >> el->fitness;
        el->bin.resize(nBins);
        for ( size_t &b : el->bin ) {
            is >> tmp;
            b = size_t(tmp);
        }
        el->deviations.resize(nParams);
        for ( scalar &d : el->deviations )
            is >> d;

        el->obs = {{}, {}};
        for ( size_t i = 0; i < maxObs; i++ ) {
            is >> start >> stop;
            if ( i < iObservations::maxObs ) {
                el->obs.start[i] = start;
                el->obs.stop[i] = stop;
            }
        }
        is >> el->current;

        is >> *idxIt;
    }

    quint32 uniqueSize, sharedSize;
    is >> uniqueSize;
    std::vector<std::shared_ptr<iStimulation>> w_unique(uniqueSize);
    for ( std::shared_ptr<iStimulation> &ptr : w_unique ) {
        ptr.reset(new iStimulation);
        is >> *ptr;
    }
    is >> sharedSize;
    std::vector<std::shared_ptr<iStimulation>> w_shared(sharedSize);
    for ( std::shared_ptr<iStimulation> &ptr : w_shared ) {
        ptr.reset(new iStimulation);
        is >> *ptr;
    }

    idxIt = stimIdx.begin();
    for ( auto el = arch->elites.begin(); el != arch->elites.end(); el++, idxIt++ ) {
        if ( *idxIt < 0 )
            el->wave = w_unique[-1 - *idxIt];
        else
            el->wave = w_shared[*idxIt];
    }

    return arch;
}
