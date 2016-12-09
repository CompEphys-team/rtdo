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
            // Pick offspring of existing pool of elites
            std::vector<size_t> parentIndices(numWavesPerEpisode);
            RNG.generate(parentIndices, size_t(0), mapeArchive.size()-1);
            std::sort(parentIndices.begin(), parentIndices.end());
            auto it = mapeArchive.begin();
            size_t pos = 0;
            for ( int i = 0; i < numWavesPerEpisode; i++ ) {
                std::advance(it, parentIndices[i] - pos);
                (*newWaves)[i] = mutate(it->wave);
                pos = parentIndices[i];
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
        mape_insert(MAPElite{coords, fitness, waves[0]});

    } else {
        for ( int group = 0; group < m.numGroups; group++ ) {
            // Calculate fitness value for this group:
            double fitness = r.fitnessFunc(wavestats[group]);

            // Gather bin coordinate data for this group:
            std::vector<size_t> coords(r.dim.size(), 0);
            for ( size_t dim = 0; dim < r.dim.size(); dim++ )
                coords[dim] = r.dim[dim]->bin(waves[group], wavestats[group]);

            // Compare to elite & insert
            mape_insert(MAPElite{coords, fitness, waves[group]});
        }
    }

    // Record statistics
    if ( !mapeStats.histIter->insertions ) // Set to 1 by mape_insert when a new best is found
        mapeStats.histIter->bestFitness = prev->bestFitness;
    mapeStats.histIter->insertions = mapeStats.insertions - prevInsertions;
    mapeStats.histIter->population = mapeStats.population;
    mapeStats.historicInsertions += mapeStats.histIter->insertions;
    ++mapeStats.iterations;
}

void Wavegen::mape_insert(MAPElite &&candidate)
{
    bool inserted = true;
    auto insertedIterator = mapeArchive.end();
    for ( auto it = mapeArchive.begin(); it != mapeArchive.end(); it++ ) {
        char comp = it->compare(candidate);
        if ( comp == 0 ) { // Found an existing elite at these coordinates, compete
            inserted = it->compete(std::move(candidate));
            insertedIterator = it;
            break;
        } else if ( comp > 0 ) { // No existing elite at these coordinates
            ++mapeStats.population;
            insertedIterator = mapeArchive.insert(it, std::move(candidate));
            break;
        }
    }
    if ( insertedIterator == mapeArchive.end() ) { // No existing elite at these coordinates
        ++mapeStats.population;
        mapeArchive.push_back(std::move(candidate));
        --insertedIterator;
    }

    if ( inserted ) {
        if ( mapeStats.bestWave == mapeArchive.end() || insertedIterator->fitness > mapeStats.bestWave->fitness ) {
            mapeStats.bestWave = insertedIterator;
            mapeStats.histIter->bestFitness = insertedIterator->fitness;
            mapeStats.histIter->insertions = 1; // Code for "bestFitness has been set", see mape_tournament
        }
        ++mapeStats.insertions;
    }
}
