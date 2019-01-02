#include "wavegen.h"
#include "session.h"

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

    std::vector<size_t> stim_bins, cluster_bins;
    constexpr size_t paramIdx_bin = 0;
    size_t clusterIdx_bin = 0, numClusters_bin = 0;
    for ( size_t i = 1; i < dims.size(); i++ ) {
        switch ( dims[i].func ) {
        case MAPEDimension::Func::BestBubbleDuration:
        case MAPEDimension::Func::BestBubbleTime:
        case MAPEDimension::Func::EE_MeanCurrent:
            cluster_bins.push_back(i);
            break;
        case MAPEDimension::Func::VoltageDeviation:
        case MAPEDimension::Func::VoltageIntegral:
            stim_bins.push_back(i);
            break;
        case MAPEDimension::Func::EE_ClusterIndex:
            clusterIdx_bin = i;
            break;
        case MAPEDimension::Func::EE_NumClusters:
            numClusters_bin = i;
            break;
        default:
            break;
        }
    }

    std::vector<bool> clusterValid(ulib.maxClusters);
    for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
        std::shared_ptr<iStimulation> stim = std::make_shared<iStimulation>(stims[stimIdx]);

        // Find valid clusters
        size_t nClusters = 0, nValidClusters = 0;
        for ( size_t clusterIdx = 0; clusterIdx < ulib.maxClusters; clusterIdx++ ) {
            const iObservations &obs = ulib.clusterObs[stimIdx * ulib.maxClusters + clusterIdx];
            int len = 0;
            for ( size_t i = 0; i < iObservations::maxObs; i++ )
                len += obs.stop[i] - obs.start[i];
            if ( len >= minLength ) {
                ++nValidClusters;
                clusterValid[clusterIdx] = true;
            } else {
                clusterValid[clusterIdx] = false;
            }
            if ( len == 0 )
                break;
            ++nClusters;
        }

        // Populate stim-wide bins
        for ( size_t binIdx : stim_bins )
            bins[binIdx] = dims[binIdx].bin(*stim, 1, dt);
        if ( numClusters_bin )
            bins[numClusters_bin] = nValidClusters;

        // Construct a MAPElite for each non-zero parameter contribution of each valid cluster
        for ( size_t clusterIdx = 0; clusterIdx < nClusters; clusterIdx++ ) {
            if ( !clusterValid[clusterIdx] )
                continue;

            MAPElite el(bins, stim, 0, std::vector<scalar>(nParams), ulib.clusterObs[stimIdx * ulib.maxClusters + clusterIdx]);

            for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
                el.deviations[paramIdx] = ulib.clusters[stimIdx*ulib.maxClusters*nParams + clusterIdx*nParams + paramIdx];

            el.current = ulib.clusterCurrent[stimIdx * ulib.maxClusters + clusterIdx];
            if ( el.current > maxCurrent )
                maxCurrent = el.current;

            // Populate cluster-level bins
            for ( size_t binIdx : cluster_bins )
                el.bin[binIdx] = dims[binIdx].bin(el, dt);
            if ( clusterIdx_bin )
                el.bin[clusterIdx_bin] = clusterIdx;

            // One entry for each parameter
            for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ ) {
                if ( el.deviations[paramIdx] > 0 ) {
                    el.bin[paramIdx_bin] = paramIdx;
                    el.fitness = el.deviations[paramIdx];
                    candidates_by_param[paramIdx].push_front(el);
                    ++nCandidates;
                }
            }
        }
    }

    // Sort and consolidate the lists
    std::forward_list<MAPElite> candidates = sortCandidates(candidates_by_param, dims);
    insertCandidates(candidates);

    current.nCandidates.push_back(nCandidates);
    return maxCurrent;
}
