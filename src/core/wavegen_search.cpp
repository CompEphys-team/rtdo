#include "wavegen.h"
#include "session.h"

QString Wavegen::bubble_action = QString("bubble");
quint32 Wavegen::bubble_magic = 0xdd76dc0e;
quint32 Wavegen::bubble_version = 100;

void Wavegen::bubbleSearch()
{
    if ( session.qWavegenData().mapeDimensions[0].func != MAPEDimension::Func::EE_ParamIndex ) {
        WavegenData d = session.qWavegenData();
        d.mapeDimensions.insert(d.mapeDimensions.begin(), MAPEDimension {
                                    MAPEDimension::Func::EE_ParamIndex,
                                    0,
                                    scalar(ulib.adjustableParams.size()),
                                    ulib.adjustableParams.size()});
        session.setWavegenData(d);
    }
    session.queue(actorName(), bubble_action, "", new Result());
}

bool Wavegen::bubble_exec(QFile &file, Result *result)
{
    bool dryrun = result->dryrun;
    current = Archive(-2, searchd, *result);
    delete result;
    result = nullptr;

    if ( dryrun ) {
        bubble_save(file);
        return true;
    }

    emit startedSearch(-2);

    const int nModelsPerStim = searchd.trajectoryLength * searchd.nTrajectories;
    const int nStimsPerEpoch = ulib.NMODELS / nModelsPerStim;
    const int blankCycles = session.gaFitterSettings().cluster_blank_after_step / session.runData().dt;
    const int sectionLength = session.gaFitterSettings().cluster_fragment_dur / session.runData().dt;

    double run_maxCurrent = 0, sum_maxCurrents = 0;

    std::vector<MAPEDimension> dims = session.wavegenData().mapeDimensions;
    std::vector<size_t> variablePrecisionDims;
    for ( size_t i = 0; i < dims.size(); i++ ) {
        if ( dims[i].hasVariableResolution() )
            variablePrecisionDims.push_back(i);
        else if ( dims[i].func == MAPEDimension::Func::EE_ClusterIndex )
            std::cerr << "Warning: ClusterIndex dimension invalid for bubble search" << std::endl;
        else if ( dims[i].func == MAPEDimension::Func::EE_NumClusters )
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

    // Queue the first set of stims
    pushStimsAndObserve(*returnedWaves, nModelsPerStim, blankCycles);
    ulib.assignment = ulib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_NONE;
    ulib.run();
    ulib.bubble(searchd.trajectoryLength, searchd.nTrajectories, istimd.iDuration,
                sectionLength, deltabar, false);

    for ( size_t epoch = 0; epoch < searchd.maxIterations; epoch++ ) {
        // Copy completed bubbles for returnedWaves to host (sync)
        ulib.pullBubbles(nStimsPerEpoch);

        bool done = (++current.iterations == searchd.maxIterations) || isAborted();

        // Push and run newWaves (async)
        if ( !done ) {
            pushStimsAndObserve(*newWaves, nModelsPerStim, blankCycles);
            ulib.run();
            ulib.bubble(searchd.trajectoryLength, searchd.nTrajectories, istimd.iDuration,
                        sectionLength, deltabar, false);
        }

        // score and insert returnedWaves into archive
        scalar epoch_maxCurrent = bubble_scoreAndInsert(*returnedWaves, nStimsPerEpoch, dims);
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

    std::cout << "Overall maximum observed bubble current: " << run_maxCurrent
              << " nA; average maximum across epochs: " << (sum_maxCurrents/current.iterations) << " nA."
              << std::endl;

    current.nCandidates.squeeze();
    current.nInsertions.squeeze();
    current.nReplacements.squeeze();
    current.nElites.squeeze();
    current.meanFitness.squeeze();
    current.maxFitness.squeeze();

    bubble_save(file);
    m_archives.push_back(std::move(current));

    emit done();

    return true;
}

void Wavegen::bubble_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, bubble_magic, bubble_version) )
        return;

    os << quint32(current.precision);
    os << quint32(current.iterations);
    os << current.nCandidates << current.nInsertions << current.nReplacements << current.nElites;
    os << current.deltabar;

    os << quint32(current.elites.size());
    os << quint32(current.elites.front().bin.size());
    os << quint32(current.elites.front().deviations.size());

    // Separate waves into unique and shared to reduce the number of lookups
    std::vector<iStimulation*> w_unique, w_shared;
    for ( MAPElite const& e : current.elites ) {
        os << e.fitness;
        for ( size_t b : e.bin )
            os << quint32(b);
        for ( scalar d : e.deviations )
            os << d;
        os << quint32(e.obs.start[0]) << quint32(e.obs.stop[0]);
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

Result *Wavegen::bubble_load(QFile &file, const QString &, Result r)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, bubble_magic);
    if ( version < 100 || version > bubble_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());

    Archive *arch;
    if ( r.dryrun )
        arch = new Archive(-2, searchd, r);
    else {
        m_archives.emplace_back(-2, searchd, r);
        arch =& m_archives.back();
    }

    quint32 precision, iterations, archSize, nBins, nParams;
    is >> precision >> iterations;
    arch->precision = precision;
    arch->iterations = iterations;
    is >> arch->nCandidates >> arch->nInsertions >> arch->nReplacements >> arch->nElites;
    is >> arch->deltabar;

    is >> archSize >> nBins >> nParams;
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
        is >> start >> stop;
        el->obs.start[0] = start;
        el->obs.stop[0] = stop;

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

scalar Wavegen::bubble_scoreAndInsert(const std::vector<iStimulation> &stims, const int nStims, const std::vector<MAPEDimension> &dims)
{
    const double dt = session.runData().dt;
    const size_t nParams = ulib.adjustableParams.size();

    const int nBins = dims.size();
    std::vector<size_t> bins(nBins, 0);
    std::vector<std::forward_list<MAPElite>> candidates_by_param(nParams);
    int nCandidates = 0;
    scalar maxCurrent = 0;

    std::vector<size_t> stim_bins, bubble_bins;
    constexpr size_t paramIdx_bin = 0;
    for ( size_t i = 0; i < dims.size(); i++ ) {
        switch ( dims[i].func ) {
        case MAPEDimension::Func::BestBubbleDuration:
        case MAPEDimension::Func::BestBubbleTime:
        case MAPEDimension::Func::EE_MeanCurrent:
            bubble_bins.push_back(i);
            break;
        case MAPEDimension::Func::VoltageDeviation:
        case MAPEDimension::Func::VoltageIntegral:
            stim_bins.push_back(i);
        default: // ignore paramIdx (dealt with individually through paramIdx_bin) and clusterIdx/numClusters (left at 0)
            break;
        }
    }

    for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
        std::shared_ptr<iStimulation> stim = std::make_shared<iStimulation>(stims[stimIdx]);

        // Populate stim-wide bins
        for ( size_t binIdx : stim_bins )
            bins[binIdx] = dims[binIdx].bin(*stim, 1, dt);

        // Construct a MAPElite for each bubble (one per target parameter)
        for ( size_t targetParamIdx = 0; targetParamIdx < nParams; targetParamIdx++ ) {
            const Bubble &bubble = ulib.bubbles[stimIdx * nParams + targetParamIdx];
            if ( !bubble.cycles )
                continue;

            candidates_by_param[targetParamIdx].emplace_front(MAPElite {bins, stim, bubble.value, std::vector<scalar>(nParams), iObservations {{},{}}});
            ++nCandidates;
            MAPElite &el = candidates_by_param[targetParamIdx].front();

            for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
                el.deviations[paramIdx] = ulib.clusters[stimIdx * nParams * nParams + targetParamIdx * nParams + paramIdx];

            el.current = ulib.clusterCurrent[stimIdx * nParams + targetParamIdx];
            if ( el.current > maxCurrent )
                maxCurrent = el.current;

            el.obs.start[0] = bubble.startCycle;
            el.obs.stop[0] = bubble.startCycle + bubble.cycles;

            // Populate bubble-level bins
            for ( size_t binIdx : bubble_bins )
                el.bin[binIdx] = dims[binIdx].bin(el, dt);
            el.bin[paramIdx_bin] = targetParamIdx;
        }
    }

    // Sort and consolidate the lists
    std::forward_list<MAPElite> candidates = sortCandidates(candidates_by_param, dims);

    // Insert into archive
    int nInserted = 0, nReplaced = 0;
    auto archIter = current.elites.begin();
    for ( auto candIter = candidates.begin(); candIter != candidates.end(); candIter++ ) {
        while ( archIter != current.elites.end() && *archIter < *candIter ) // Advance to the first archive element with coords >= candidate
            ++archIter;
        if ( archIter == current.elites.end() || *candIter < *archIter ) { // No elite at candidate's coords, insert implicitly
            archIter = current.elites.insert(archIter, std::move(*candIter));
            ++nInserted;
        } else { // preexisting elite at the candidate's coords, compete
            nReplaced += archIter->compete(*candIter);
        }
    }

    current.nCandidates.push_back(nCandidates);
    current.nInsertions.push_back(nInserted);
    current.nReplacements.push_back(nReplaced);
    current.nElites.push_back(current.elites.size());

    return maxCurrent;
}

