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

    std::vector<size_t> stim_bins, bubble_bins;
    constexpr size_t paramIdx_bin = 0;
    for ( size_t i = 0; i < dims.size(); i++ ) {
        switch ( dims[i].func ) {
        case MAPEDimension::Func::BestBubbleDuration:
        case MAPEDimension::Func::BestBubbleTime:
        case MAPEDimension::Func::EE_MeanCurrent:
            bubble_bins.push_back(i);
            break;
        case MAPEDimension::Func::VoltageDeviation:
        case MAPEDimension::Func::VoltageIntegral:
            stim_bins.push_back(i);
        default: // ignore paramIdx (dealt with individually through paramIdx_bin) and clusterIdx/numClusters (left at 0)
            break;
        }
    }

    for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
        std::shared_ptr<iStimulation> stim = std::make_shared<iStimulation>(stims[stimIdx]);

        // Populate stim-wide bins
        for ( size_t binIdx : stim_bins )
            bins[binIdx] = dims[binIdx].bin(*stim, 1, dt);

        // Construct a MAPElite for each bubble (one per target parameter)
        for ( size_t targetParamIdx = 0; targetParamIdx < nParams; targetParamIdx++ ) {
            const Bubble &bubble = ulib.bubbles[stimIdx * nParams + targetParamIdx];
            if ( !bubble.cycles )
                continue;

            candidates_by_param[targetParamIdx].emplace_front(MAPElite {bins, stim, bubble.value, std::vector<scalar>(nParams), iObservations {{},{}}});
            ++nCandidates;
            MAPElite &el = candidates_by_param[targetParamIdx].front();

            for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
                el.deviations[paramIdx] = ulib.clusters[stimIdx * nParams * nParams + targetParamIdx * nParams + paramIdx];

            el.current = ulib.clusterCurrent[stimIdx * nParams + targetParamIdx];
            if ( el.current > maxCurrent )
                maxCurrent = el.current;

            el.obs.start[0] = bubble.startCycle;
            el.obs.stop[0] = bubble.startCycle + bubble.cycles;

            // Populate bubble-level bins
            for ( size_t binIdx : bubble_bins )
                el.bin[binIdx] = dims[binIdx].bin(el, dt);
            el.bin[paramIdx_bin] = targetParamIdx;
        }
    }

    // Sort and consolidate the lists
    std::forward_list<MAPElite> candidates = sortCandidates(candidates_by_param, dims);
    insertCandidates(candidates);

    current.nCandidates.push_back(nCandidates);
    return maxCurrent;
}
