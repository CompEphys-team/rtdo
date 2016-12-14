#include "wavegen.h"
#include "cuda_helper.h"
#include "kernelhelper.h"
#include "mapedimension.h"
#include <cassert>

using namespace GeNN_Bridge;

void Wavegen::search(int param)
{
    assert(param >= 0 && param < (int)m.adjustableParams.size());
    *targetParam = param+1;
    *getErr = true;

    const int numWavesPerEpisode = m.cfg.permute ? 1 : m.numGroups;
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
        double fitness = 0.0;
        std::vector<double> behaviour(r.dim.size());
        std::vector<size_t> coords(r.dim.size());
        for ( int group = 0; group < m.numGroups; group++ ) {
            // Calculate fitness value for this group:
            fitness += r.fitnessFunc(wavestats[group]);

            // Gather bin coordinate data for this group:
            for ( size_t dim = 0; dim < r.dim.size(); dim++ )
                behaviour[dim] += r.dim[dim]->behaviour(waves[0], wavestats[group]);
        }

        // Average
        fitness /= m.numGroups;
        for ( size_t dim = 0; dim < r.dim.size(); dim++ )
            coords[dim] = r.dim[dim]->bin(behaviour[dim] / m.numGroups);

        // Compare averages to elite & insert
        std::vector<MAPElite> tmp(1, MAPElite{coords, fitness, waves[0]});
        mape_insert(tmp);

    } else {
        std::vector<MAPElite> candidates;
        candidates.reserve(m.numGroups);
        for ( int group = 0; group < m.numGroups; group++ ) {
            // Gather bin coordinate data for this group:
            std::vector<size_t> coords(r.dim.size());
            for ( size_t dim = 0; dim < r.dim.size(); dim++ )
                coords[dim] = r.dim[dim]->bin(waves[group], wavestats[group]);

            double fitness = r.fitnessFunc(wavestats[group]);
            candidates.push_back(MAPElite(coords, fitness, waves[group]));
            candidates.back().stats =& wavestats[group];
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

        if ( mapeStats.bestWave == mapeArchive.end() || archIter->fitness > mapeStats.bestWave->fitness ) {
            mapeStats.bestWave = archIter;
            mapeStats.histIter->bestFitness = archIter->fitness;
            mapeStats.histIter->insertions = 1; // Code for "bestFitness has been set", see mape_tournament
            std::cout << "New best wave: " << archIter->wave << ", Fitness: " << archIter->fitness << ", binned at "
                      << archIter->bin[0] << "," << archIter->bin[1] << "," << archIter->bin[2] << "," << archIter->bin[3];
            if ( archIter->stats ) {
                mapeStats.bestStats = *(archIter->stats);
                std::cout << " for its stats:" << std::endl << *(archIter->stats);
            }
            std::cout << std::endl;
        }
    }
}
