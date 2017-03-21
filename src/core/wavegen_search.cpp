#include "wavegen.h"
#include "cuda_helper.h"
#include <cassert>
#include <QDataStream>
#include "session.h"

void Wavegen::search(int param)
{
    if ( aborted )
        return;
    completedArchives[param].clear();
    emit startedSearch(param);

    assert(param >= 0 && param < (int)lib.adjustableParams.size());

    const int numWavesPerEpisode = lib.project.wgPermute() ? 1 : lib.numGroups;
    std::vector<Stimulation> waves_ep1(numWavesPerEpisode), waves_ep2(numWavesPerEpisode);

    // Initialise the population for episode 1:
    for ( Stimulation &w : waves_ep1 )
        w = getRandomStim();
    size_t nInitialWaves = numWavesPerEpisode;

    // Initiate a first stimulation with nothing going on in parallel:
    restoreSettled();
    detune();
    lib.targetParam = param+1;
    lib.getErr = true;
    lib.clearStats();
    stimulate(waves_ep1);

    // Prepare the second episode:
    for ( Stimulation &w : waves_ep2 )
        w = getRandomStim();
    nInitialWaves += numWavesPerEpisode;
    bool initialising = nInitialWaves < searchd.nInitialWaves;

    // Initialise waves pointer to episode 1, which is currently being stimulated
    std::vector<Stimulation> *returnedWaves = &waves_ep1, *newWaves = &waves_ep2;

    mapeArchive.clear();
    mapeStats = MAPEStats(searchd.historySize, mapeArchive.end());
    mapeStats.histIter = mapeStats.history.begin();

    while ( true ) {
        lib.pullStats(); // Pull stats from the previous episode (returnedWaves' performance) to host memory
        lib.clearStats(); // Reset device memory stats
        restoreSettled(); // Reset state
        stimulate(*newWaves); // Initiate next stimulation episode

        // Calculate fitness & MAPE coordinates of the previous episode's waves, and compete with the elite
        mape_tournament(*returnedWaves);

        // Swap pointers: After this, returnedWaves points at the waves currently being stimulated,
        // while newWaves points at the already handled waves of the previous episode,
        // which can be safely overwritten with a new preparation.
        using std::swap;
        swap(newWaves, returnedWaves);

        emit searchTick(mapeStats.iterations);

        if ( mapeStats.iterations == searchd.maxIterations-1 || aborted )
            break;

        if ( mapeStats.precision < searchd.precisionIncreaseEpochs.size() &&
             mapeStats.iterations == searchd.precisionIncreaseEpochs[mapeStats.precision] ) {
            mapeStats.precision++;
            for ( MAPElite &e : mapeArchive )
                e.bin = mape_bin(e.wave, e.stats);
        }

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
            RNG.generate(idx, size_t(0), mapeArchive.size() - 1);
            std::sort(idx.begin(), idx.end());

            // Insert the sampling into parents in a single run through
            auto archIter = mapeArchive.begin();
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
                RNG.shuffle(parents);
                wellShuffled = true;
                for ( int i = 0; i < numWavesPerEpisode; i++ ) {
                    if ( parents[2*i] == parents[2*i + 1] ) {
                        if ( i < numWavesPerEpisode - 1 ) {
                            RNG.shuffle(parents.begin() + 2*i, parents.end());
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
    }

    // Pull and evaluate the last episode
    lib.pullStats();
    mape_tournament(*returnedWaves);

    // Correct observation periods
    for ( MAPElite &e : mapeArchive ) {
        e.wave.tObsBegin = e.stats.best.tEnd - e.stats.best.cycles*lib.project.dt()/lib.simCycles;
        e.wave.tObsEnd = e.stats.best.tEnd;
    }

    QFile file(session.log(this, search_action, QString::number(param)));
    search_save(file);

    completedArchives[param] = std::move(mapeArchive);
    archivePrecision[param] = mapeStats.precision;
    mapeArchive.clear();
    emit done(param);
}

void Wavegen::mape_tournament(const std::vector<Stimulation> &waves)
{
    // Advance statistics history and prepare stats page for this iteration
    auto prev = mapeStats.histIter;
    double prevInsertions = mapeStats.insertions;
    if ( ++mapeStats.histIter == mapeStats.history.end() )
        mapeStats.histIter = mapeStats.history.begin();
    mapeStats.historicInsertions -= mapeStats.histIter->insertions;
    *mapeStats.histIter = {};

    if ( lib.project.wgPermute() ) {
        // For both fitness and bin coordinates, take the mean fitness/behaviour of the stim across all evaluated model groups

        // Accumulate
        WaveStats meanStats = WaveStats();
        for ( int group = 0; group < lib.numGroups; group++ ) {
            meanStats += lib.wavestats[group];
        }

        // Average
        meanStats /= lib.numGroups;

        // Compare averages to elite & insert
        std::vector<MAPElite> candidate(1, MAPElite{mape_bin(waves[0], meanStats), waves[0], meanStats});
        mape_insert(candidate);

    } else {
        std::vector<MAPElite> candidates;
        candidates.reserve(lib.numGroups);
        for ( int group = 0; group < lib.numGroups; group++ ) {
            candidates.push_back(MAPElite {
                                     mape_bin(waves[group], lib.wavestats[group]),
                                     waves[group],
                                     lib.wavestats[group]
                                 });
        }

        // Compare to elite & insert
        mape_insert(candidates);
    }

    // Record statistics
    if ( !mapeStats.histIter->insertions ) // Set to 1 by mape_insert when a new best is found
        mapeStats.histIter->bestFitness = prev->bestFitness;
    mapeStats.histIter->insertions = mapeStats.insertions - prevInsertions;
    mapeStats.histIter->population = mapeStats.population;
    mapeStats.historicInsertions += mapeStats.histIter->insertions;
    ++mapeStats.iterations;
}

void Wavegen::mape_insert(std::vector<MAPElite> &candidates)
{
    std::sort(candidates.begin(), candidates.end()); // Lexical sort by MAPElite::bin
    auto archIter = mapeArchive.begin();
    for ( auto candIter = candidates.begin(); candIter != candidates.end(); candIter++ ) {
        while ( archIter != mapeArchive.end() && *archIter < *candIter ) // Advance to the first archive element with coords >= candidate
            ++archIter;
        if ( archIter == mapeArchive.end() || *candIter < *archIter ) { // No elite at candidate's coords, insert implicitly
            ++mapeStats.population;
            ++mapeStats.insertions;
            archIter = mapeArchive.insert(archIter, *candIter);
        } else { // preexisting elite at the candidate's coords, compete
            mapeStats.insertions += archIter->compete(*candIter);
        }

        if ( mapeStats.bestWave == mapeArchive.end() || archIter->stats.fitness > mapeStats.bestWave->stats.fitness ) {
            mapeStats.bestWave = archIter;
            mapeStats.histIter->bestFitness = archIter->stats.fitness;
            mapeStats.histIter->insertions = 1; // Code for "bestFitness has been set", see mape_tournament
            std::cout << "New best wave: " << archIter->wave << ", binned at ";
            for ( const size_t &x : archIter->bin )
                std::cout << x << ",";
            std::cout << " for its stats: " << archIter->stats << std::endl;
        }
    }
}

std::vector<size_t> Wavegen::mape_bin(const Stimulation &I, const WaveStats &S)
{
    size_t mult = mape_multiplier(mapeStats.precision);
    std::vector<size_t> bin(searchd.mapeDimensions.size());
    for ( size_t i = 0; i < searchd.mapeDimensions.size(); i++ ) {
        bin[i] = searchd.mapeDimensions.at(i).bin(I, S, mult);
    }

    return bin;
}

void Wavegen::search_save(QFile &file)
{
    QDataStream os;
    if ( !Session::openSaveStream(file, os, search_magic, search_version) )
        return;
    os << quint32(mapeStats.precision);
    os << quint32(mapeArchive.size());
    for ( MAPElite const& e : mapeArchive )
        os << e;
}

void Wavegen::search_load(QFile &file, const QString &args)
{
    QDataStream is;
    quint32 version = Session::openLoadStream(file, is, search_magic);
    if ( version < 100 || version > 100 )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());
    quint32 precision, archSize;
    is >> precision >> archSize;
    mapeStats.precision = size_t(precision);
    std::list<MAPElite> archive(archSize);
    for ( MAPElite &e : archive )
        is >> e;
    int param = args.toInt();
    completedArchives[param] = std::move(archive);
}
