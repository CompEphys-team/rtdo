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

    // Note the dimensions that can't be computed once for an entire stim
    // NOTE: This expects that the first dimension is always EE_ParamIdx.
    constexpr int bin_for_paramIdx = 0;
    int bin_for_clusterIdx = -1, bin_for_clusterDuration = -1, bin_for_current = -1;
    for ( int i = 0; i < nBins; i++ ) {
        if ( dims[i].func == MAPEDimension::Func::EE_ClusterIndex )
            bin_for_clusterIdx = i;
        else if ( dims[i].func == MAPEDimension::Func::BestBubbleDuration )
            bin_for_clusterDuration = i;
        else if ( dims[i].func == MAPEDimension::Func::EE_MeanCurrent )
            bin_for_current = i;
    }

    for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
        std::shared_ptr<iStimulation> stim = std::make_shared<iStimulation>(stims[stimIdx]);
        stim->tObsBegin = 0;

        // Find number of valid clusters
        size_t nClusters = 0, nValidClusters = 0;
        for ( size_t clusterIdx = 0; clusterIdx < ulib.maxClusters; clusterIdx++ ) {
            const iObservations &obs = ulib.clusterObs[stimIdx * ulib.maxClusters + clusterIdx];
            int len = 0;
            for ( size_t i = 0; i < iObservations::maxObs; i++ )
                len += obs.stop[i] - obs.start[i];
            if ( len >= minLength )
                ++nValidClusters;
            if ( len == 0 )
                break;
            ++nClusters;
        }

        // Populate all bins (with some garbage for clusterIdx, paramIdx, clusterDuration)
        for ( int binIdx = 0; binIdx < nBins; binIdx++ )
            bins[binIdx] = dims[binIdx].bin(*stim, 0, 0, nValidClusters, 1, dt);

        // Construct a MAPElite for each non-zero parameter contribution of each valid cluster
        for ( size_t clusterIdx = 0; clusterIdx < nClusters; clusterIdx++ ) {
            const iObservations &obs = ulib.clusterObs[stimIdx * ulib.maxClusters + clusterIdx];

            // Check for valid length
            int len = 0;
            for ( size_t i = 0; i < iObservations::maxObs; i++ )
                len += obs.stop[i] - obs.start[i];
            if ( len >= minLength ) {
                scalar meanCurrent = ulib.clusterCurrent[stimIdx * ulib.maxClusters + clusterIdx];
                if ( meanCurrent > maxCurrent )
                    maxCurrent = meanCurrent;

                // Populate cluster-level bins
                if ( bin_for_clusterDuration > 0 ) {
                    stim->tObsEnd = len;
                    bins[bin_for_clusterDuration] = dims[bin_for_clusterDuration].bin(*stim, 1, dt);
                }
                if ( bin_for_clusterIdx > 0 )
                    bins[bin_for_clusterIdx] = dims[bin_for_clusterIdx].bin(*stim, 0, clusterIdx, 0, 1, dt);
                if ( bin_for_current > 0 )
                    bins[bin_for_current] = dims[bin_for_current].bin(meanCurrent, 1);

                // One entry for each parameter
                std::vector<scalar> contrib(nParams);
                for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
                    contrib[paramIdx] = ulib.clusters[stimIdx*ulib.maxClusters*nParams + clusterIdx*nParams + paramIdx];
                for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ ) {
                    if ( contrib[paramIdx] > 0 ) {
                        bins[bin_for_paramIdx] = dims[bin_for_paramIdx].bin(*stim, paramIdx, 0, 0, 1, dt);
                        candidates_by_param[paramIdx].emplace_front(MAPElite {bins, stim, contrib[paramIdx], contrib, obs});
                        candidates_by_param[paramIdx].front().current = meanCurrent;
                        ++nCandidates;
                    }
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
