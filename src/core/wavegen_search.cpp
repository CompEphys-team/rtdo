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

scalar Wavegen::bubble_scoreAndInsert(const std::vector<iStimulation> &stims, const int nStims, const std::vector<MAPEDimension> &dims)
{
    const double dt = session.runData().dt;
    const size_t nParams = ulib.adjustableParams.size();

    const int nBins = dims.size();
    std::vector<size_t> bins(nBins, 0);
    std::vector<std::forward_list<MAPElite>> candidates_by_param(nParams);
    int nCandidates = 0;
    scalar maxCurrent = 0;

    std::vector<size_t> genotype_bins, bubble_bins;
    constexpr size_t paramIdx_bin = 0;
    for ( size_t i = 1; i < dims.size(); i++ ) {
        switch ( dims[i].func ) {
        case MAPEDimension::Func::BestBubbleDuration:
        case MAPEDimension::Func::BestBubbleTime:
        case MAPEDimension::Func::EE_MeanCurrent:
            bubble_bins.push_back(i);
            break;
        case MAPEDimension::Func::VoltageDeviation:
        case MAPEDimension::Func::VoltageIntegral:
            genotype_bins.push_back(i);
        default: // ignore clusterIdx/numClusters (left at 0)
            break;
        }
    }

    for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
        std::shared_ptr<iStimulation> stim = std::make_shared<iStimulation>(stims[stimIdx]);

        // Populate stim-wide bins
        for ( size_t binIdx : genotype_bins )
            bins[binIdx] = dims[binIdx].bin_genotype(*stim, 1, dt);

        // Construct a MAPElite for each bubble (one per target parameter)
        for ( size_t targetParamIdx = 0; targetParamIdx < nParams; targetParamIdx++ ) {
            const Bubble &bubble = ulib.bubbles[stimIdx * nParams + targetParamIdx];
            if ( !bubble.cycles )
                continue;

            candidates_by_param[targetParamIdx].emplace_front(MAPElite {bins, stim, bubble.value, std::vector<scalar>(nParams), iObservations {{},{}}});
            ++nCandidates;
            MAPElite &el = candidates_by_param[targetParamIdx].front();

            el.current = ulib.clusterCurrent[stimIdx * nParams + targetParamIdx];
            if ( std::isnan(el.current) )
                continue;
            if ( el.current > maxCurrent )
                maxCurrent = el.current;

            for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
                el.deviations[paramIdx] = ulib.clusters[stimIdx * nParams * nParams + targetParamIdx * nParams + paramIdx];

            el.obs.start[0] = bubble.startCycle;
            el.obs.stop[0] = bubble.startCycle + bubble.cycles;

            // Populate bubble-level bins
            for ( size_t binIdx : bubble_bins )
                el.bin[binIdx] = dims[binIdx].bin_elite(el, dt);
            el.bin[paramIdx_bin] = targetParamIdx;
        }
    }

    // Sort and consolidate the lists
    std::forward_list<MAPElite> candidates = sortCandidates(candidates_by_param, dims);
    insertCandidates(candidates);

    current.nCandidates.push_back(nCandidates);
    return maxCurrent;
}
