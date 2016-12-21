#include "wavegen.h"
#include "cuda_helper.h"
#include <cassert>

void Wavegen::search(int param)
{
    assert(param >= 0 && param < (int)adjustableParams.size());
    targetParam = param+1;
    getErr = true;

    const int numWavesPerEpisode = m.cfg.permute ? 1 : numGroups;
    std::vector<Stimulation> waves_ep1(numWavesPerEpisode), waves_ep2(numWavesPerEpisode);

    // Initialise the population for episode 1:
    for ( Stimulation &w : waves_ep1 )
        w = getRandomStim();
    size_t nInitialWaves = numWavesPerEpisode;

    // Initiate a first stimulation with nothing going on in parallel:
    restoreSettled();
    clearStats();
    stimulate(waves_ep1);

    // Prepare the second episode:
    for ( Stimulation &w : waves_ep2 )
        w = getRandomStim();
    nInitialWaves += numWavesPerEpisode;
    bool initialising = nInitialWaves < r.nInitialWaves;

    // Initialise waves pointer to episode 1, which is currently being stimulated
    std::vector<Stimulation> *returnedWaves = &waves_ep1, *newWaves = &waves_ep2;

    mapeArchive.clear();
    mapeStats = MAPEStats(r.historySize, mapeArchive.end());
    mapeStats.histIter = mapeStats.history.begin();

    while ( true ) {
        pullStats(); // Pull stats from the previous episode (returnedWaves' performance) to host memory
        clearStats(); // Reset device memory stats
        restoreSettled(); // Reset state
        stimulate(*newWaves); // Initiate next stimulation episode

        // Calculate fitness & MAPE coordinates of the previous episode's waves, and compete with the elite
        mape_tournament(*returnedWaves);

        // Swap pointers: After this, returnedWaves points at the waves currently being stimulated,
        // while newWaves points at the already handled waves of the previous episode,
        // which can be safely overwritten with a new preparation.
        using std::swap;
        swap(newWaves, returnedWaves);

        if ( r.stopFunc(mapeStats) )
            break;

        if ( r.increasePrecision(mapeStats) ) {
            mapeStats.precision++;
            for ( MAPElite &e : mapeArchive )
                e.bin = r.binFunc(e.wave, e.stats, mapeStats.precision);
        }

        // Prepare next episode's waves
        if ( initialising ) {
            // Generate at random
            for ( Stimulation &w : *newWaves )
                w = getRandomStim();
            nInitialWaves += numWavesPerEpisode;
            initialising = nInitialWaves < r.nInitialWaves;
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
            } while ( !wellShuffled || shuffleFailures > 10 );

            // Mutate
            for ( int i = 0; i < numWavesPerEpisode; i++ ) {
                (*newWaves)[i] = mutate(parents[2*i]->wave, parents[2*i + 1]->wave);
            }
        }
    }

    // Pull and evaluate the last episode
    pullStats();
    mape_tournament(*returnedWaves);
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

    if ( m.cfg.permute ) {
        // For both fitness and bin coordinates, take the mean fitness/behaviour of the stim across all evaluated model groups

        // Accumulate
        WaveStats meanStats = WaveStats();
        for ( int group = 0; group < numGroups; group++ ) {
            meanStats += wavestats[group];
        }

        // Average
        meanStats /= numGroups;

        // Compare averages to elite & insert
        std::vector<MAPElite> candidate(1, MAPElite{r.binFunc(waves[0], meanStats, mapeStats.precision), waves[0], meanStats});
        mape_insert(candidate);

    } else {
        std::vector<MAPElite> candidates;
        candidates.reserve(numGroups);
        for ( int group = 0; group < numGroups; group++ ) {
            candidates.push_back(MAPElite {
                                     r.binFunc(waves[group], wavestats[group], mapeStats.precision),
                                     waves[group],
                                     wavestats[group]
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
