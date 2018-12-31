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

void Wavegen::construct_next_generation(std::vector<iStimulation> &stims)
{
    const int nStims = stims.size();
    std::vector<std::list<MAPElite>::const_iterator> parents(2 * nStims);

    // Sample the archive space with a bunch of random indices
    std::vector<size_t> idx(2 * nStims);
    session.RNG.generate(idx, size_t(0), current.elites.size() - 1);
    std::sort(idx.begin(), idx.end());

    // Insert the sampling into parents in a single run through
    auto archIter = current.elites.begin();
    size_t pos = 0;
    for ( int i = 0; i < 2*nStims; i++ ) {
        std::advance(archIter, idx[i] - pos);
        pos = idx[i];
        parents[i] = archIter;
    }

    // Shuffle the parents, but ensure that xover isn't incestuous
    session.RNG.shuffle(parents);
    for ( int i = 0; i < nStims; i++ ) {
        if ( parents[2*i] == parents[2*i + 1] ) {
            while ( true ) {
                int otherPair = session.RNG.uniform(0, nStims-2);
                otherPair += (otherPair >= i);
                if ( parents[2*otherPair] != parents[2*i] && parents[2*otherPair+1] != parents[2*i] ) {
                    // Both of the other pair are different from the incestuous couple => swap one of them over
                    parents[2*i+1] = parents[2*otherPair];
                    parents[2*otherPair] = parents[2*i];
                    break;
                }
            }
        }
    }

    // Mutate
    for ( int i = 0; i < nStims; i++ ) {
        stims[i] = mutate(*parents[2*i]->wave, *parents[2*i + 1]->wave);
    }
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
