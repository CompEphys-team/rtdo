#include "wavegen.h"
#include "cuda_helper.h"
#include <cassert>
#include <QDataStream>
#include "session.h"

void Wavegen::search(int param)
{
    if ( aborted )
        return;
    emit startedSearch(param);

    assert(param >= 0 && param < (int)lib.adjustableParams.size());

    // Note: Episode == a single call to stimulate and lib.step; Epoch == a full iteration, containing one or more episode.
    const int numWavesPerEpisode = lib.numGroups / searchd.nGroupsPerWave;
    const int numEpisodesPerEpoch = searchd.nWavesPerEpoch / numWavesPerEpisode;
    int episodeCounter = 0;

    std::vector<Stimulation> waves_ep1(numWavesPerEpisode), waves_ep2(numWavesPerEpisode);

    // Initialise the population for episode 1:
    for ( Stimulation &w : waves_ep1 )
        w = getRandomStim();
    size_t nInitialWaves = numWavesPerEpisode;

    // Initiate a first stimulation with nothing going on in parallel:
    initModels();
    detune();
    settle();
    lib.targetParam = param+1;
    lib.getErr = true;
    lib.nStim = numWavesPerEpisode;
    pushStims(waves_ep1);
    lib.generateBubbles(stimd.duration);

    // Prepare the second episode:
    for ( Stimulation &w : waves_ep2 )
        w = getRandomStim();
    nInitialWaves += numWavesPerEpisode;
    bool initialising = nInitialWaves < searchd.nInitialWaves;

    // Initialise waves pointer to episode 1, which is currently being stimulated
    std::vector<Stimulation> *returnedWaves = &waves_ep1, *newWaves = &waves_ep2;

    current = Archive(param, searchd);

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

            if ( current.iterations == searchd.maxIterations || aborted )
                break;

            if ( current.precision < searchd.precisionIncreaseEpochs.size() &&
                 current.iterations == searchd.precisionIncreaseEpochs[current.precision] ) {
                current.precision++;
                for ( MAPElite &e : current.elites )
                    e.bin = mape_bin(e.wave);
                current.elites.sort();
            }
        } else if ( aborted ) {
            break;
        }
        emit searchTick(current.iterations);

        // Prepare next episode's waves
        if ( initialising ) {
            // Generate at random
            for ( Stimulation &w : *newWaves )
                w = getRandomStim();
            nInitialWaves += numWavesPerEpisode;
            initialising = nInitialWaves < searchd.nInitialWaves;
        } else {
            std::vector<std::list<MAPElite>::const_iterator> parents(2 * numWavesPerEpisode);

            // Sample the archive space with a bunch of random indices
            std::vector<size_t> idx(2 * numWavesPerEpisode);
            session.RNG.generate(idx, size_t(0), current.elites.size() - 1);
            std::sort(idx.begin(), idx.end());

            // Insert the sampling into parents in a single run through
            auto archIter = current.elites.begin();
            size_t pos = 0;
            for ( int i = 0; i < 2*numWavesPerEpisode; i++ ) {
                std::advance(archIter, idx[i] - pos);
                pos = idx[i];
                parents[i] = archIter;
            }

            // Shuffle the parents, but ensure that xover isn't incestuous
            bool wellShuffled;
            int shuffleFailures = 0;
            do {
                session.RNG.shuffle(parents);
                wellShuffled = true;
                for ( int i = 0; i < numWavesPerEpisode; i++ ) {
                    if ( parents[2*i] == parents[2*i + 1] ) {
                        if ( i < numWavesPerEpisode - 1 ) {
                            session.RNG.shuffle(parents.begin() + 2*i, parents.end());
                            --i;
                        } else {
                            wellShuffled = false;
                            ++shuffleFailures;
                            break;
                        }
                    }
                }
            } while ( !wellShuffled && shuffleFailures < 10 );

            // Mutate
            for ( int i = 0; i < numWavesPerEpisode; i++ ) {
                (*newWaves)[i] = mutate(parents[2*i]->wave, parents[2*i + 1]->wave);
            }
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
    m_archives.push_back(std::move(current));

    QFile file(session.log(this, search_action, QString::number(param)));
    search_save(file);

    emit done(param);
}

void Wavegen::mape_tournament(std::vector<Stimulation> &waves)
{
    // Gather candidates
    std::vector<MAPElite> candidates;
    candidates.reserve(waves.size());
    scalar msPerCycle = lib.project.dt() / lib.simCycles;
    for ( size_t i = 0; i < waves.size(); i++ ) {
        if ( lib.bubbles[i].cycles > 0 ) {
            waves[i].tObsBegin = lib.bubbles[i].startCycle * msPerCycle;
            waves[i].tObsEnd = (lib.bubbles[i].startCycle + lib.bubbles[i].cycles) * msPerCycle;
            candidates.emplace_back(mape_bin(waves[i]), waves[i], lib.bubbles[i].value);
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

std::vector<size_t> Wavegen::mape_bin(const Stimulation &I)
{
    size_t mult = mape_multiplier(current.precision);
    std::vector<size_t> bin(searchd.mapeDimensions.size());
    for ( size_t i = 0; i < searchd.mapeDimensions.size(); i++ ) {
        bin[i] = searchd.mapeDimensions.at(i).bin(I, mult);
    }

    return bin;
}

void Wavegen::search_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, search_magic, search_version) )
        return;
    Archive &arch = m_archives.back();
    os << quint32(arch.precision);
    os << quint32(arch.iterations);
    os << quint32(arch.elites.size());
    for ( MAPElite const& e : arch.elites )
        os << e;
    os << arch.nCandidates << arch.nInsertions << arch.nReplacements << arch.nElites;
    os << arch.meanFitness << arch.maxFitness;
}

void Wavegen::search_load(QFile &file, const QString &args)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, search_magic);
    if ( version < 100 || version > search_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());

    quint32 precision, iterations, archSize;
    is >> precision >> iterations >> archSize;
    m_archives.push_back(Archive(args.toInt(), searchd)); // Note: this->searchd is correctly set up assuming sequential result loading
    Archive &arch = m_archives.back();
    arch.precision = precision;
    arch.iterations = iterations;
    arch.elites.resize(archSize);
    for ( MAPElite &e : arch.elites )
        is >> e;
    if ( version >= 101 ) {
        is >> arch.nCandidates >> arch.nInsertions >> arch.nReplacements >> arch.nElites;
        is >> arch.meanFitness >> arch.maxFitness;
    }
}
