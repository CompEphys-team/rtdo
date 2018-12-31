#include "wavegen.h"
#include "session.h"

QString Wavegen::cluster_action = QString("cluster");
quint32 Wavegen::cluster_magic = 0xc9fd545f;
quint32 Wavegen::cluster_version = 100;

void Wavegen::clusterSearch()
{
    if ( session.qWavegenData().mapeDimensions[0].func != MAPEDimension::Func::EE_ParamIndex ) {
        WavegenData d = session.qWavegenData();
        d.mapeDimensions.insert(d.mapeDimensions.begin(), MAPEDimension {
                                    MAPEDimension::Func::EE_ParamIndex,
                                    0,
                                    scalar(lib.adjustableParams.size()),
                                    lib.adjustableParams.size()});
        session.setWavegenData(d);
    }
    session.queue(actorName(), cluster_action, "", new Result());
}

bool Wavegen::cluster_exec(QFile &file, Result *result)
{
    bool dryrun = result->dryrun;
    current = Archive(-1, searchd, *result);
    delete result;
    result = nullptr;

    if ( dryrun ) {
        cluster_save(file);
        return true;
    }

    emit startedSearch(-1);

    const int nModelsPerStim = searchd.trajectoryLength * searchd.nTrajectories;
    const int nStimsPerEpoch = ulib.NMODELS / nModelsPerStim;
    const int blankCycles = session.gaFitterSettings().cluster_blank_after_step / session.runData().dt;
    const int sectionLength = session.gaFitterSettings().cluster_fragment_dur / session.runData().dt;
    const scalar dotp_threshold = session.gaFitterSettings().cluster_threshold;
    const int minClusterLen = session.gaFitterSettings().cluster_min_dur / session.runData().dt;

    double run_maxCurrent = 0, sum_maxCurrents = 0;

    std::vector<MAPEDimension> dims = session.wavegenData().mapeDimensions;
    std::vector<size_t> variablePrecisionDims;
    int meanCurrentDim = -1;
    for ( size_t i = 0; i < dims.size(); i++ ) {
        if ( dims[i].hasVariableResolution() )
            variablePrecisionDims.push_back(i);
        if ( dims[i].func == MAPEDimension::Func::EE_MeanCurrent )
            meanCurrentDim = i;
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
    ulib.cluster(searchd.trajectoryLength, searchd.nTrajectories, istimd.iDuration,
                 sectionLength, dotp_threshold, minClusterLen, deltabar, false);

    for ( size_t epoch = 0; epoch < searchd.maxIterations; epoch++ ) {
        // Copy completed clusters for returnedWaves to host (sync)
        ulib.pullClusters(nStimsPerEpoch);

        bool done = (++current.iterations == searchd.maxIterations) || isAborted();

        // Push and run newWaves (async)
        if ( !done ) {
            pushStimsAndObserve(*newWaves, nModelsPerStim, blankCycles);
            ulib.run();
            ulib.cluster(searchd.trajectoryLength, searchd.nTrajectories, istimd.iDuration,
                         sectionLength, dotp_threshold, minClusterLen, deltabar, false);
        }

        // score and insert returnedWaves into archive
        scalar epoch_maxCurrent = cluster_scoreAndInsert(*returnedWaves, nStimsPerEpoch, dims);
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
                    for ( size_t i : variablePrecisionDims ) {
                        if ( int(i) == meanCurrentDim )
                            e.bin[i] = dims.at(i).bin(e.current, session.runData().dt);
                        else
                            e.bin[i] = dims.at(i).bin(*e.wave, 1, session.runData().dt);
                    }
                current.elites.sort(); // TODO: radix sort
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

    std::cout << "Overall maximum observed cluster current: " << run_maxCurrent
              << " nA; average maximum across epochs: " << (sum_maxCurrents/current.iterations) << " nA."
              << std::endl;

    current.nCandidates.squeeze();
    current.nInsertions.squeeze();
    current.nReplacements.squeeze();
    current.nElites.squeeze();
    current.meanFitness.squeeze();
    current.maxFitness.squeeze();

    cluster_save(file);
    m_archives.push_back(std::move(current));

    emit done();

    return true;
}

void Wavegen::cluster_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, cluster_magic, cluster_version) )
        return;

    os << quint32(current.precision);
    os << quint32(current.iterations);
    os << current.nCandidates << current.nInsertions << current.nReplacements << current.nElites;
    os << current.deltabar;

    os << quint32(current.elites.size());
    os << quint32(current.elites.front().bin.size());
    os << quint32(current.elites.front().deviations.size());
    os << quint32(iObservations::maxObs);

    // Separate waves into unique and shared to reduce the number of lookups
    std::vector<iStimulation*> w_unique, w_shared;
    for ( MAPElite const& e : current.elites ) {
        os << e.fitness;
        for ( size_t b : e.bin )
            os << quint32(b);
        for ( scalar d : e.deviations )
            os << d;
        for ( size_t i = 0; i < iObservations::maxObs; i++ )
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

Result *Wavegen::cluster_load(QFile &file, const QString &, Result r)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, cluster_magic);
    if ( version < 100 || version > cluster_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());

    Archive *arch;
    if ( r.dryrun )
        arch = new Archive(-1, searchd, r);
    else {
        m_archives.emplace_back(-1, searchd, r);
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

scalar Wavegen::cluster_scoreAndInsert(const std::vector<iStimulation> &stims, const int nStims, const std::vector<MAPEDimension> &dims)
{
    const double dt = session.runData().dt;
    const int minLength = session.gaFitterSettings().cluster_min_dur / dt;
    const size_t nParams = ulib.adjustableParams.size();

    const int nBins = dims.size();
    std::vector<size_t> bins(nBins);
    std::vector<std::forward_list<MAPElite>> candidates_by_param(nParams);
    int nCandidates = 0;
    scalar maxCurrent = 0;

    // Note the dimensions that can't be computed once for an entire stim
    // NOTE: This expects that the first dimension is always EE_ParamIdx.
    constexpr int bin_for_paramIdx = 0;
    int bin_for_clusterIdx = -1, bin_for_clusterDuration = -1, bin_for_current = -1;
    for ( int i = 0; i < nBins; i++ ) {
        if ( dims[i].func == MAPEDimension::Func::EE_ClusterIndex )
            bin_for_clusterIdx = i;
        else if ( dims[i].func == MAPEDimension::Func::BestBubbleDuration )
            bin_for_clusterDuration = i;
        else if ( dims[i].func == MAPEDimension::Func::EE_MeanCurrent )
            bin_for_current = i;
    }

    for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
        std::shared_ptr<iStimulation> stim = std::make_shared<iStimulation>(stims[stimIdx]);
        stim->tObsBegin = 0;

        // Find number of valid clusters
        size_t nClusters = 0, nValidClusters = 0;
        for ( size_t clusterIdx = 0; clusterIdx < ulib.maxClusters; clusterIdx++ ) {
            const iObservations &obs = ulib.clusterObs[stimIdx * ulib.maxClusters + clusterIdx];
            int len = 0;
            for ( size_t i = 0; i < iObservations::maxObs; i++ )
                len += obs.stop[i] - obs.start[i];
            if ( len >= minLength )
                ++nValidClusters;
            if ( len == 0 )
                break;
            ++nClusters;
        }

        // Populate all bins (with some garbage for clusterIdx, paramIdx, clusterDuration)
        for ( int binIdx = 0; binIdx < nBins; binIdx++ )
            bins[binIdx] = dims[binIdx].bin(*stim, 0, 0, nValidClusters, 1, dt);

        // Construct a MAPElite for each non-zero parameter contribution of each valid cluster
        for ( size_t clusterIdx = 0; clusterIdx < nClusters; clusterIdx++ ) {
            const iObservations &obs = ulib.clusterObs[stimIdx * ulib.maxClusters + clusterIdx];

            // Check for valid length
            int len = 0;
            for ( size_t i = 0; i < iObservations::maxObs; i++ )
                len += obs.stop[i] - obs.start[i];
            if ( len >= minLength ) {
                scalar meanCurrent = ulib.clusterCurrent[stimIdx * ulib.maxClusters + clusterIdx];
                if ( meanCurrent > maxCurrent )
                    maxCurrent = meanCurrent;

                // Populate cluster-level bins
                if ( bin_for_clusterDuration > 0 ) {
                    stim->tObsEnd = len;
                    bins[bin_for_clusterDuration] = dims[bin_for_clusterDuration].bin(*stim, 1, dt);
                }
                if ( bin_for_clusterIdx > 0 )
                    bins[bin_for_clusterIdx] = dims[bin_for_clusterIdx].bin(*stim, 0, clusterIdx, 0, 1, dt);
                if ( bin_for_current > 0 )
                    bins[bin_for_current] = dims[bin_for_current].bin(meanCurrent, 1);

                // One entry for each parameter
                std::vector<scalar> contrib(nParams);
                for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
                    contrib[paramIdx] = ulib.clusters[stimIdx*ulib.maxClusters*nParams + clusterIdx*nParams + paramIdx];
                for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ ) {
                    if ( contrib[paramIdx] > 0 ) {
                        bins[bin_for_paramIdx] = dims[bin_for_paramIdx].bin(*stim, paramIdx, 0, 0, 1, dt);
                        candidates_by_param[paramIdx].emplace_front(MAPElite {bins, stim, contrib[paramIdx], contrib, obs});
                        candidates_by_param[paramIdx].front().current = meanCurrent;
                        ++nCandidates;
                    }
                }
            }
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
