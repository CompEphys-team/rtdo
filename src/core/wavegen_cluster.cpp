/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#include "wavegen.h"
#include "session.h"

scalar Wavegen::cluster_scoreAndInsert(const std::vector<iStimulation> &stims, const int nStims, const std::vector<MAPEDimension> &dims)
{
    const double dt = session.runData().dt;
    const int minLength = searchd.cluster.minLen / dt;
    const size_t nParams = ulib.adjustableParams.size();

    const int nBins = dims.size();
    std::vector<size_t> bins(nBins);
    std::vector<std::forward_list<MAPElite>> candidates_by_param(nParams);
    int nCandidates = 0;
    scalar maxCurrent = 0;

    std::vector<size_t> genotype_bins, cluster_bins;
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
            genotype_bins.push_back(i);
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
        for ( size_t binIdx : genotype_bins )
            bins[binIdx] = dims[binIdx].bin_genotype(*stim, 1, dt);
        if ( numClusters_bin )
            bins[numClusters_bin] = nValidClusters-1;

        // Construct a MAPElite for each non-zero parameter contribution of each valid cluster
        for ( size_t clusterIdx = 0; clusterIdx < nClusters; clusterIdx++ ) {
            if ( !clusterValid[clusterIdx] )
                continue;

            MAPElite el(bins, stim, 0, std::vector<scalar>(nParams), ulib.clusterObs[stimIdx * ulib.maxClusters + clusterIdx]);

            el.current = ulib.clusterCurrent[stimIdx * ulib.maxClusters + clusterIdx];
            if ( std::isnan(el.current) )
                continue;
            if ( el.current > maxCurrent )
                maxCurrent = el.current;

            for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
                el.deviations[paramIdx] = ulib.clusters[stimIdx*ulib.maxClusters*nParams + clusterIdx*nParams + paramIdx];

            // Populate cluster-level bins
            for ( size_t binIdx : cluster_bins )
                el.bin[binIdx] = dims[binIdx].bin_elite(el, dt);
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
