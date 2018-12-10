#include "wavegen.h"
#include "session.h"
#include <forward_list>

QString Wavegen::ee_action = QString("ee_search");
quint32 Wavegen::ee_magic = 0xc9fd545f;
quint32 Wavegen::ee_version = 100;

void Wavegen::elementaryEffects()
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
    session.queue(actorName(), ee_action, "", new Result());
}

void prepareModels(Session &session, UniversalLibrary &ulib, const WavegenData &searchd)
{
    int nParams = ulib.adjustableParams.size(), nextParam = 0;
    bool initial = true;
    std::vector<scalar> values(nParams);
    size_t nModelsPerStim = searchd.nTrajectories * searchd.trajectoryLength;
    for ( size_t i = 0; i < ulib.NMODELS; i++ ) {
        if ( i % searchd.trajectoryLength == 0 ) { // Pick a starting point
            // For the benefit of rerandomiseParameters && useBaseParameters, provide each new stim with a trajectory from base model
            if ( i % nModelsPerStim == 0 )
                initial = true;
            if ( searchd.useBaseParameters && initial ) // Reset to base model
                for ( int j = 0; j < nParams; j++ )
                    values[j] = ulib.adjustableParams[j].initial;
            else if ( searchd.rerandomiseParameters || i < nModelsPerStim ) // Randomise uniformly
                for ( int j = 0; j < nParams; j++ )
                    values[j] = session.RNG.uniform(ulib.adjustableParams[j].min, ulib.adjustableParams[j].max);
            else // Copy from first stim
                for ( int j = 0; j < nParams; j++ )
                    values[j] = ulib.adjustableParams[j][i % nModelsPerStim];
        } else {
            // Add a sigma-sized step to one parameter at a time
            values[nextParam] += ulib.adjustableParams[nextParam].sigma;
            nextParam = (nextParam+1) % nParams;
            if ( nextParam == 0 )
                initial = false;
        }

        for ( size_t j = 0; j < values.size(); j++ )
            ulib.adjustableParams[j][i] = values[j];
    }
}

void settleModels(const Session &session, UniversalLibrary &ulib)
{
    const RunData &rd = session.runData();

    ulib.setSingularRund();
    ulib.setRundata(0, rd);
    ulib.integrator = rd.integrator;
    ulib.simCycles = rd.simCycles;

    ulib.setSingularTarget();
    ulib.targetOffset[0] = 0;

    ulib.setSingularStim();
    ulib.stim[0].baseV = session.stimulationData().baseV;

    ulib.assignment = ulib.assignment_base | ASSIGNMENT_SETTLE_ONLY | ASSIGNMENT_MAINTAIN_STATE;
    ulib.push();
    ulib.run();
}

void pushStimsAndObserve(const std::vector<iStimulation>&stims, UniversalLibrary &ulib, int nModelsPerStim, int blankCycles)
{
    ulib.setSingularStim(false);
    for ( size_t i = 0; i < ulib.NMODELS; i++ ) {
        ulib.stim[i] = stims[i / nModelsPerStim];
    }
    ulib.push(ulib.stim);
    ulib.observe_no_steps(blankCycles);
}

QVector<double> getDeltabar(Session &session, UniversalLibrary &ulib)
{
    const WavegenData &searchd = session.wavegenData();
    int nModelsPerStim = searchd.nTrajectories * searchd.trajectoryLength;
    std::vector<iStimulation> stims(ulib.NMODELS / nModelsPerStim);
    for ( iStimulation &stim : stims )
        stim = session.wavegen().getRandomStim();

    ulib.iSettleDuration[0] = 0;
    ulib.push(ulib.iSettleDuration);

    pushStimsAndObserve(stims, ulib, nModelsPerStim, session.gaFitterSettings().cluster_blank_after_step / session.runData().dt);

    ulib.assignment = ulib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_PREVTHREAD;
    ulib.run();

    std::vector<double> dbar = ulib.find_deltabar(searchd.trajectoryLength, searchd.nTrajectories, stims[0].duration);
    return QVector<double>::fromStdVector(dbar);
}

/// Radix sort by MAPElite::bin, starting from dimension @a firstDimIdx upwards.
/// Returns: An iterator to the final element of the sorted list.
std::forward_list<MAPElite>::iterator radixSort(std::forward_list<MAPElite> &list, const std::vector<MAPEDimension> &dims, int firstDimIdx = 0)
{
    auto tail = list.before_begin();
    for ( int dimIdx = dims.size()-1; dimIdx >= firstDimIdx; dimIdx-- ) {
        size_t nBuckets = dims[dimIdx].resolution;
        std::vector<std::forward_list<MAPElite>> buckets(nBuckets);
        std::vector<std::forward_list<MAPElite>::iterator> tails(nBuckets);
        for ( size_t bucketIdx = 0; bucketIdx < nBuckets; bucketIdx++ )
            tails[bucketIdx] = buckets[bucketIdx].before_begin();

        // Sort into buckets, maintaining existing order
        while ( !list.empty() ) {
            const size_t bucketIdx = list.begin()->bin[dimIdx];
            buckets[bucketIdx].splice_after(tails[bucketIdx], list, list.before_begin());
            ++tails[bucketIdx];
        }

        // Consolidate back into a contiguous list
        tail = list.before_begin();
        for ( size_t bucketIdx = 0; bucketIdx < nBuckets; bucketIdx++ ) {
            if ( !buckets[bucketIdx].empty() ) {
                list.splice_after(tail, buckets[bucketIdx]);
                tail = tails[bucketIdx];
            }
        }
    }

    // Guard against loop not being entered at all (1D archive?)
    if ( tail == list.before_begin() && !list.empty() ) {
        auto after_tail = tail;
        for ( ++after_tail; after_tail != list.end(); ++after_tail )
            ++tail;
    }
    return tail;
}

std::forward_list<MAPElite> sortCandidates(std::vector<std::forward_list<MAPElite>> &candidates_by_param, const std::vector<MAPEDimension> &dims)
{
    std::forward_list<MAPElite> ret;
    auto tail = ret.before_begin();
    for ( std::forward_list<MAPElite> l : candidates_by_param ) {
        auto next_tail = radixSort(l, dims, 1); // NOTE: Expects EE_ParamIdx as first dimension.
        ret.splice_after(tail, l);
        using std::swap;
        swap(tail, next_tail);
    }
    return ret;
}

void scoreAndInsert(const std::vector<iStimulation> &stims, UniversalLibrary &ulib,
                    Session &session, Wavegen::Archive &current, const int nStims,
                    const std::vector<MAPEDimension> &dims)
{
    const double dt = session.runData().dt;
    const int minLength = session.gaFitterSettings().cluster_min_dur / dt;
    const size_t nParams = ulib.adjustableParams.size();

    const int nBins = dims.size();
    std::vector<size_t> bins(nBins);
    std::vector<std::forward_list<MAPElite>> candidates_by_param(nParams);
    int nCandidates = 0;

    // Note the dimensions that can't be computed once for an entire stim
    // NOTE: This expects that the first dimension is always EE_ParamIdx.
    constexpr int bin_for_paramIdx = 0;
    int bin_for_clusterIdx = -1, bin_for_clusterDuration = -1;
    for ( int i = 0; i < nBins; i++ ) {
        if ( dims[i].func == MAPEDimension::Func::EE_ClusterIndex )
            bin_for_clusterIdx = i;
        else if ( dims[i].func == MAPEDimension::Func::BestBubbleDuration )
            bin_for_clusterDuration = i;
    }

    for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
        std::shared_ptr<iStimulation> stim = std::make_shared<iStimulation>(stims[stimIdx]);
        stim->tObsBegin = 0;

        // Find number of valid clusters
        size_t nClusters = 0;
        for ( int clusterIdx = 0; clusterIdx < ulib.maxClusters; clusterIdx++ )
            if ( ulib.clusterLen[stimIdx*ulib.maxClusters + clusterIdx] >= minLength )
                ++nClusters;

        // Populate all bins (with some garbage for clusterIdx, paramIdx, clusterDuration)
        for ( int binIdx = 0; binIdx < nBins; binIdx++ )
            bins[binIdx] = dims[binIdx].bin(*stim, 0, 0, nClusters, 1, dt);

        // Construct a MAPElite for each non-zero parameter contribution of each valid cluster
        for ( int clusterIdx = 0; clusterIdx < ulib.maxClusters; clusterIdx++ ) {

            // Check for valid length
            int len = ulib.clusterLen[stimIdx*ulib.maxClusters + clusterIdx];
            if ( len >= minLength ) {

                // Populate cluster-level bins
                if ( bin_for_clusterDuration > 0 ) {
                    stim->tObsEnd = len;
                    bins[bin_for_clusterDuration] = dims[bin_for_clusterDuration].bin(*stim, 1, dt);
                }
                if ( bin_for_clusterIdx > 0 )
                    bins[bin_for_clusterIdx] = dims[bin_for_clusterIdx].bin(*stim, 0, clusterIdx, 0, 1, dt);

                // One entry for each parameter
                std::vector<scalar> contrib(nParams);
                for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
                    contrib[paramIdx] = ulib.clusters[stimIdx*ulib.maxClusters*nParams + clusterIdx*nParams + paramIdx];
                for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ ) {
                    if ( contrib[paramIdx] > 0 ) {
                        bins[bin_for_paramIdx] = dims[bin_for_paramIdx].bin(*stim, paramIdx, 0, 0, 1, dt);
                        candidates_by_param[paramIdx].emplace_front(MAPElite {bins, stim, contrib[paramIdx], contrib});
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
}

bool Wavegen::ee_exec(QFile &file, Result *result)
{
    current = Archive(-1, searchd, *result);
    delete result;
    result = nullptr;

    emit startedSearch(-1);

    UniversalLibrary &ulib = session.project.universal();
    const int nModelsPerStim = searchd.trajectoryLength * searchd.nTrajectories;
    const int nStimsPerEpoch = ulib.NMODELS / nModelsPerStim;
    const int blankCycles = session.gaFitterSettings().cluster_blank_after_step / session.runData().dt;
    const int sectionLength = session.gaFitterSettings().cluster_fragment_dur / session.runData().dt;
    const scalar dotp_threshold = session.gaFitterSettings().cluster_threshold;
    const int minClusterLen = session.gaFitterSettings().cluster_min_dur / session.runData().dt;

    std::vector<MAPEDimension> dims = session.wavegenData().mapeDimensions;
    std::vector<size_t> variablePrecisionDims;
    for ( size_t i = 0; i < dims.size(); i++ )
        if ( dims[i].hasVariableResolution() )
            variablePrecisionDims.push_back(i);

    ulib.resizeOutput(istimd.iDuration);
    prepareModels(session, ulib, searchd);
    settleModels(session, ulib);

    // Use the settling period to generate the first two sets of stims
    std::vector<iStimulation> waves_ep1(nStimsPerEpoch), waves_ep2(nStimsPerEpoch);
    for ( iStimulation &w : waves_ep1 )
        w = getRandomStim();
    for ( iStimulation &w : waves_ep2 )
        w = getRandomStim();
    size_t nInitialStims = 2*nStimsPerEpoch;
    std::vector<iStimulation> *returnedWaves = &waves_ep1, *newWaves = &waves_ep2;

    // Run a set of stims through a deltabar finding mission
    current.deltabar = getDeltabar(session, ulib);
    std::vector<double> deltabar = current.deltabar.toStdVector();

    // Queue the first set of stims
    pushStimsAndObserve(*returnedWaves, ulib, nModelsPerStim, blankCycles);
    ulib.assignment = ulib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_PREVTHREAD;
    ulib.run();
    ulib.cluster(searchd.trajectoryLength, searchd.nTrajectories, istimd.iDuration, sectionLength, dotp_threshold, minClusterLen, deltabar, false);

    for ( size_t epoch = 0; epoch < searchd.maxIterations; epoch++ ) {
        // Copy completed clusters for returnedWaves to host (sync)
        ulib.pullClusters(nStimsPerEpoch);

        bool done = (++current.iterations == searchd.maxIterations) || isAborted();

        // Push and run newWaves (async)
        if ( !done ) {
            pushStimsAndObserve(*newWaves, ulib, nModelsPerStim, blankCycles);
            ulib.run();
            ulib.cluster(searchd.trajectoryLength, searchd.nTrajectories, istimd.iDuration, sectionLength, dotp_threshold, minClusterLen, deltabar, false);
        }

        // score and insert returnedWaves into archive
        scoreAndInsert(*returnedWaves, ulib, session, current, nStimsPerEpoch, dims);

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
                        e.bin[i] = dims.at(i).bin(*e.wave, 1, session.runData().dt);
                current.elites.sort(); // TODO: radix sort
            }
        }

        emit searchTick(current.iterations);

        // Prepare the next set of stims
        if ( nInitialStims < searchd.nInitialWaves ) {
            for ( iStimulation &w : *newWaves )
                w = getRandomStim();
            nInitialStims += nStimsPerEpoch;
        } else {
            construct_next_generation(*newWaves);
        }
    }

    current.nCandidates.squeeze();
    current.nInsertions.squeeze();
    current.nReplacements.squeeze();
    current.nElites.squeeze();
    current.meanFitness.squeeze();
    current.maxFitness.squeeze();
    m_archives.push_back(std::move(current));

    ee_save(file);

    emit done();

    return true;
}

void Wavegen::ee_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, ee_magic, ee_version) )
        return;
    Archive &arch = m_archives.back();
    os << quint32(arch.precision);
    os << quint32(arch.iterations);
    os << arch.nCandidates << arch.nInsertions << arch.nReplacements << arch.nElites;
    os << arch.deltabar;

    os << quint32(arch.elites.size());
    os << quint32(arch.elites.front().bin.size());
    os << quint32(arch.elites.front().deviations.size());

    // Separate waves into unique and shared to reduce the number of lookups
    std::vector<iStimulation*> w_unique, w_shared;
    for ( MAPElite const& e : arch.elites ) {
        os << e.fitness;
        for ( size_t b : e.bin )
            os << quint32(b);
        for ( scalar d : e.deviations )
            os << d;

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

void Wavegen::ee_load(QFile &file, const QString &, Result r)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, ee_magic);
    if ( version < 100 || version > ee_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());

    m_archives.emplace_back(-1, searchd, r);
    Archive &arch = m_archives.back();

    quint32 precision, iterations, archSize, nBins, nParams;
    is >> precision >> iterations;
    arch.precision = precision;
    arch.iterations = iterations;
    is >> arch.nCandidates >> arch.nInsertions >> arch.nReplacements >> arch.nElites;
    is >> arch.deltabar;

    is >> archSize >> nBins >> nParams;
    arch.elites.resize(archSize);
    std::vector<qint32> stimIdx(archSize);
    auto idxIt = stimIdx.begin();
    quint32 tmp;
    for ( auto el = arch.elites.begin(); el != arch.elites.end(); el++, idxIt++ ) {
        is >> el->fitness;
        el->bin.resize(nBins);
        for ( size_t &b : el->bin ) {
            is >> tmp;
            b = size_t(tmp);
        }
        el->deviations.resize(nParams);
        for ( scalar &d : el->deviations )
            is >> d;
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
    for ( auto el = arch.elites.begin(); el != arch.elites.end(); el++, idxIt++ ) {
        if ( *idxIt < 0 )
            el->wave = w_unique[-1 - *idxIt];
        else
            el->wave = w_shared[*idxIt];
    }
}
