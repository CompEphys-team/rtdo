#include "wavegen.h"
#include "cuda_helper.h"
#include "kernelhelper.h"
#include "mapedimension.h"

using namespace GeNN_Bridge;

std::vector<size_t> Wavegen::getMAPEDimensions()
{
    std::vector<size_t> dim;
    dim.reserve(r.dim.size());
    for ( const std::shared_ptr<MAPEDimension> &d : r.dim )
        dim.push_back(d->size());
    return dim;
}

void Wavegen::search(int param)
{
    assert(param >= 0 && param < (int)m.adjustableParams.size());
    *targetParam = param+1;
    for ( int i = 0; i < nModels; i++ )
        getErr[i] = true;

    const int numWavesPerEpisode = m.cfg.permute ? 1 : m.numGroups;
    std::vector<Stimulation> waves_ep1(numWavesPerEpisode), waves_ep2(numWavesPerEpisode);

    // Initialise the population for episode 1:
    for ( Stimulation &w : waves_ep1 )
        w = getRandomStim();
    size_t nInitialWaves = numWavesPerEpisode;

    // Initiate a first stimulation with nothing going on in parallel:
    restoreSettled();
    clearStats();
    stimulate(waves_ep1, true);

    // Prepare the second episode:
    for ( Stimulation &w : waves_ep2 )
        w = getRandomStim();
    nInitialWaves += numWavesPerEpisode;
    bool initialising = nInitialWaves < r.nInitialWaves;

    // Initialise waves pointer to episode 1, which is currently being stimulated
    std::vector<Stimulation> *returnedWaves = &waves_ep1, *newWaves = &waves_ep2;

    bool done = false;
    while ( !done ) {
        pullStats(); // Pull stats from the previous episode (returnedWaves' performance) to host memory
        clearStats(); // Reset device memory stats
        restoreSettled(); // Reset state
        stimulate(*newWaves, true); // Initiate next stimulation episode

        // Calculate fitness & MAPE coordinates of the previous episode's waves, and compete with the elite
        mape_tournament(*returnedWaves);

        // Swap pointers: After this, returnedWaves points at the waves currently being stimulated,
        // while newWaves points at the already handled waves of the previous episode,
        // which can be safely overwritten with a new preparation.
        using std::swap;
        swap(newWaves, returnedWaves);

        /// TODO: Insert stopping conditions.

        // Prepare next episode's waves
        if ( initialising ) {
            // Generate at random
            for ( Stimulation &w : *newWaves )
                w = getRandomStim();
            nInitialWaves += numWavesPerEpisode;
            initialising = nInitialWaves < r.nInitialWaves;
        } else {
            // Pick offspring of existing pool of elites
            // If the MAPE population is expected to be sparse, it might be more efficient to keep a list of actually populated cells...
            // However, since this is running in parallel to a full stimulation episode, it is unlikely to a be performance-relevant section.
            for ( Stimulation &w : *newWaves ) {
                const Stimulation *parent, *crossoverParent;
                do {
                    parent = mapeArchive[RNG.uniform<size_t>(0, mapeArchive.size() - 1)].get();
                    crossoverParent = mapeArchive[RNG.uniform<size_t>(0, mapeArchive.size() - 1)].get();
                } while ( !parent || !crossoverParent || parent == crossoverParent );
                w = mutate(*parent, *crossoverParent);
            }
        }
    }

    // Pull and evaluate the last episode
    pullStats();
    mape_tournament(*returnedWaves);
}

int Wavegen::mape_tournament(const std::vector<Stimulation> &waves)
{
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

            // Reset stats, to be pushed on restoreSettled call in main loop
            wavestats[group] = {};
        }

        // Average
        fitness /= m.numGroups;
        for ( size_t dim = 0; dim < r.dim.size(); dim++ )
            coords[dim] = r.dim[dim]->bin(behaviour[dim] / m.numGroups);

        // Compare averages to elite & insert
        return mape_insert(waves[0], coords, fitness);

    } else {
        int inserted = 0;
        for ( int group = 0; group < m.numGroups; group++ ) {
            // Calculate fitness value for this group:
            double fitness = r.fitnessFunc(wavestats[group]);

            // Gather bin coordinate data for this group:
            std::vector<size_t> coords(r.dim.size(), 0);
            for ( size_t dim = 0; dim < r.dim.size(); dim++ )
                coords[dim] = r.dim[dim]->bin(waves[group], wavestats[group]);

            // Reset stats
            wavestats[group] = {};

            // Compare to elite & insert
            inserted += mape_insert(waves[group], coords, fitness);
        }
        return inserted;
    }
}

int Wavegen::mape_insert(const Stimulation &I, const std::vector<size_t> &coords, double fitness)
{
    if ( !mapeArchive[coords] || fitness > mapeFitness[coords] ) {
        mapeFitness[coords] = fitness;
        if ( mapeArchive[coords] )
            *(mapeArchive[coords]) = I;
        else
            mapeArchive[coords].reset(new Stimulation(I));
        return 1;
    }
    return 0;
}
