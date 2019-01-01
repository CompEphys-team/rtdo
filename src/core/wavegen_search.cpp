#include "wavegen.h"
#include "cuda_helper.h"
#include <cassert>
#include <QDataStream>
#include "session.h"

void Wavegen::search(int param)
{
    WavegenData d = session.qWavegenData();
    bool changed = false;
    for ( int i = d.mapeDimensions.size()-1; i >= 0; i-- ) {
        if ( d.mapeDimensions[i].func == MAPEDimension::Func::EE_ClusterIndex
             || d.mapeDimensions[i].func == MAPEDimension::Func::EE_NumClusters
             || d.mapeDimensions[i].func == MAPEDimension::Func::EE_ParamIndex ) {
            d.mapeDimensions.erase(d.mapeDimensions.begin() + i);
            changed = true;
        }
    }
    if ( changed )
        session.setWavegenData(d);

    assert(param >= 0 && param < (int)lib.adjustableParams.size());
    Archive *a = new Archive;
    a->param = param;
    session.queue(actorName(), search_action, QString::fromStdString(lib.adjustableParams.at(param).name), a);
}

bool Wavegen::search_exec(QFile &file, Result *result)
{
    bool dryrun = result->dryrun;
    int param = static_cast<Archive*>(result)->param;
    current = Archive(param, searchd, *result);
    delete result;
    result = nullptr;

    if ( dryrun ) {
        search_save(file);
        return true;
    }

    emit startedSearch(param);

    // Note: Episode == a single call to stimulate and lib.step; Epoch == a full iteration, containing one or more episode.
    const int numWavesPerEpisode = lib.numGroups / searchd.nGroupsPerWave;
    const int numEpisodesPerEpoch = searchd.nWavesPerEpoch / numWavesPerEpisode;
    int episodeCounter = 0;

    std::vector<iStimulation> waves_ep1(numWavesPerEpisode), waves_ep2(numWavesPerEpisode);

    // Initialise the population for episode 1:
    for ( iStimulation &w : waves_ep1 )
        w = getRandomStim(stimd, istimd);
    size_t nInitialWaves = numWavesPerEpisode;

    // Initiate a first stimulation with nothing going on in parallel:
    initModels();
    detune();
    settle();

    std::vector<double> meanErr = getMeanParamError();
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
        lib.adjustableParams[i].deltaBar = meanErr[i];

    lib.targetParam = param+1;
    lib.deltaBar = lib.adjustableParams[param].deltaBar;
    lib.ext_variance = searchd.noise_sd * searchd.noise_sd;
    lib.getErr = true;
    lib.nStim = numWavesPerEpisode;
    pushStims(waves_ep1);
    lib.generateBubbles(istimd.iDuration);

    // Prepare the second episode:
    for ( iStimulation &w : waves_ep2 )
        w = getRandomStim(stimd, istimd);
    nInitialWaves += numWavesPerEpisode;
    bool initialising = nInitialWaves < searchd.nInitialWaves;

    // Initialise waves pointer to episode 1, which is currently being stimulated
    std::vector<iStimulation> *returnedWaves = &waves_ep1, *newWaves = &waves_ep2;

    while ( true ) {
        lib.pullBubbles();
        pushStims(*newWaves);
        lib.generateBubbles(stimd.duration); // Initiate next stimulation episode

        // Calculate fitness & MAPE coordinates of the previous episode's waves, and compete with the elite
        mape_tournament(*returnedWaves);

        // Swap pointers: After this, returnedWaves points at the waves currently being stimulated,
        // while newWaves points at the already handled waves of the previous episode,
        // which can be safely overwritten with a new preparation.
        using std::swap;
        swap(newWaves, returnedWaves);

        if ( ++episodeCounter == numEpisodesPerEpoch ) {
            episodeCounter = 0;
            ++current.iterations;

            if ( current.iterations == searchd.maxIterations || isAborted() )
                break;

            if ( current.precision < searchd.precisionIncreaseEpochs.size() &&
                 current.iterations == searchd.precisionIncreaseEpochs[current.precision] ) {
                current.precision++;
                double sumFitness = 0;
                for ( MAPElite &e : current.elites ) {
                    e.bin = mape_bin(*e.wave);
                    sumFitness += e.fitness;
                }
                current.elites.sort();
                current.meanFitness.last() = sumFitness / current.elites.size();
            }
        } else if ( isAborted() ) {
            break;
        }
        emit searchTick(current.iterations);

        // Prepare next episode's waves
        if ( initialising ) {
            // Generate at random
            for ( iStimulation &w : *newWaves )
                w = getRandomStim(stimd, istimd);
            nInitialWaves += numWavesPerEpisode;
            initialising = nInitialWaves < searchd.nInitialWaves;
        } else {
            construct_next_generation(*newWaves);
        }

        if ( episodeCounter == 0 && searchd.rerandomiseParameters ) {
            initModels();
            detune();
            settle();
        }
    }

    // Pull and evaluate the last episode
    lib.pullBubbles();
    mape_tournament(*returnedWaves);

    current.nCandidates.squeeze();
    current.nInsertions.squeeze();
    current.nReplacements.squeeze();
    current.nElites.squeeze();
    current.meanFitness.squeeze();
    current.maxFitness.squeeze();

    search_save(file);
    m_archives.push_back(std::move(current));

    emit done(param);

    return true;
}

void Wavegen::mape_tournament(std::vector<iStimulation> &waves)
{
    // Gather candidates
    std::vector<MAPElite> candidates;
    candidates.reserve(waves.size());
    for ( size_t i = 0; i < waves.size(); i++ ) {
        if ( lib.bubbles[i].cycles > 0 ) {
            waves[i].tObsBegin = lib.bubbles[i].startCycle;
            waves[i].tObsEnd = lib.bubbles[i].startCycle + lib.bubbles[i].cycles;
            candidates.emplace_back(mape_bin(waves[i]), std::make_shared<iStimulation>(waves[i]), lib.bubbles[i].value);
        }
    }

    // Compare to elite & insert
    mape_insert(candidates);
}

void Wavegen::mape_insert(std::vector<MAPElite> &candidates)
{
    int nInserted = 0, nReplaced = 0;
    double addedFitness = 0, maxFitness = current.maxFitness.size() ? current.maxFitness.last() : 0;
    std::sort(candidates.begin(), candidates.end()); // Lexical sort by MAPElite::bin
    auto archIter = current.elites.begin();
    for ( auto candIter = candidates.begin(); candIter != candidates.end(); candIter++ ) {
        while ( archIter != current.elites.end() && *archIter < *candIter ) // Advance to the first archive element with coords >= candidate
            ++archIter;
        if ( archIter == current.elites.end() || *candIter < *archIter ) { // No elite at candidate's coords, insert implicitly
            archIter = current.elites.insert(archIter, *candIter);
            ++nInserted;
            addedFitness += archIter->fitness;
            if ( archIter->fitness > maxFitness )
                maxFitness = archIter->fitness;
        } else { // preexisting elite at the candidate's coords, compete
            double prevFitness = archIter->fitness;
            bool replaced = archIter->compete(*candIter);
            if ( replaced ) {
                ++nReplaced;
                addedFitness += archIter->fitness - prevFitness;
                if ( archIter->fitness > maxFitness )
                    maxFitness = archIter->fitness;
            }
        }
    }

    double meanFitness = ((current.meanFitness.size() ? current.meanFitness.last()*current.nElites.last() : 0) + addedFitness) / current.elites.size();
    current.nCandidates.push_back(candidates.size());
    current.nInsertions.push_back(nInserted);
    current.nReplacements.push_back(nReplaced);
    current.nElites.push_back(current.elites.size());
    current.meanFitness.push_back(meanFitness);
    current.maxFitness.push_back(maxFitness);
}

std::vector<size_t> Wavegen::mape_bin(const iStimulation &I)
{
    size_t mult = mape_multiplier(current.precision);
    std::vector<size_t> bin(searchd.mapeDimensions.size());
    for ( size_t i = 0; i < searchd.mapeDimensions.size(); i++ ) {
        bin[i] = searchd.mapeDimensions.at(i).bin(I, mult, session.runData().dt);
    }

    return bin;
}

void Wavegen::search_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, search_magic, search_version) )
        return;

    os << quint32(current.param);
    os << quint32(current.precision);
    os << quint32(current.iterations);
    os << quint32(current.elites.size());
    for ( MAPElite const& e : current.elites )
        os << e;
    os << current.nCandidates << current.nInsertions << current.nReplacements << current.nElites;
    os << current.meanFitness << current.maxFitness;
}

Result *Wavegen::search_load(QFile &file, const QString &args, Result r)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, search_magic);
    if ( version < 100 || version > search_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());

    quint32 param, precision, iterations, archSize;
    if ( version >= 102 )
        is >> param;
    else
        param = args.toInt();
    is >> precision >> iterations >> archSize;

    Archive *arch;
    if ( r.dryrun )
        arch = new Archive(param, searchd, r);
    else {
        m_archives.push_back(Archive(param, searchd, r)); // Note: this->searchd is correctly set up assuming sequential result loading
        arch =& m_archives.back();
    }

    arch->precision = precision;
    arch->iterations = iterations;
    arch->elites.resize(archSize);

    if ( version >= 110 ) {
        if ( version >= 112 ) {
            for ( MAPElite &e : arch->elites )
                is >> e;
        } else {
            for ( MAPElite &e : arch->elites ) {
                is >> *e.wave >> e.fitness;
                quint32 size, val;
                is >> size;
                e.bin.resize(size);
                for ( size_t &b : e.bin ) {
                    is >> val;
                    b = size_t(val);
                }
            }
        }
        if ( version == 110 ) {
            // iStimulation probably generated at WavegenData::dt=0.01; convert to RunData::dt stepping
            // There may be some loss of precision, but that's unavoidable. >.<
            double factor = session.runData().dt / 0.01;
            for ( MAPElite &e : arch->elites ) {
                e.wave->duration = lrint(e.wave->duration / factor);
                e.wave->tObsBegin = lrint(e.wave->tObsBegin / factor);
                e.wave->tObsEnd = lrint(e.wave->tObsEnd / factor);
                for ( iStimulation::Step &step : *e.wave )
                    step.t = lrint(step.t / factor);
            }
        }
    } else {
        for ( MAPElite &e : arch->elites ) {
            MAPElite__scalarStim old;
            is >> old;
            e.bin = old.bin;
            e.wave.reset(new iStimulation(old.wave, session.runData().dt));
            e.fitness = old.fitness;
        }
    }
    if ( version < 112 ) {
        for ( MAPElite &e : arch->elites ) {
            e.obs.start[0] = e.wave->tObsBegin;
            e.obs.stop[0] = e.wave->tObsEnd;
        }
    }

    if ( version >= 101 ) {
        is >> arch->nCandidates >> arch->nInsertions >> arch->nReplacements >> arch->nElites;
        is >> arch->meanFitness >> arch->maxFitness;
    }

    return arch;
}



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

