#include "wavegen.h"
#include "session.h"

void Wavegen::prepare_models()
{
    int nParams = ulib.adjustableParams.size();
    std::vector<scalar> values(nParams), baseValues;
    unsigned int optionBits = 0;
    size_t nModelsPerStim = searchd.nTrajectories * searchd.trajectoryLength;
    std::vector<int> detuneIndices = ulib.model.get_detune_indices(searchd.trajectoryLength, searchd.nTrajectories);
    auto detIdx = detuneIndices.begin();
    for ( size_t i = 0; i < ulib.NMODELS; i++ ) {
        if ( *detIdx < 0 ) { // Pick a starting point
            if ( i < nModelsPerStim ) { // Generate novel values for first stim
                if ( searchd.useBaseParameters && *detIdx == -2 ) {
                    for ( int j = 0; j < nParams; j++ )
                        values[j] = ulib.adjustableParams[j].initial;
                } else if ( session.runData().VC ) {
                    for ( int j = 0; j < nParams; j++ )
                        values[j] = session.RNG.uniform(ulib.adjustableParams[j].min, ulib.adjustableParams[j].max);
                } else {
                    ++optionBits;
                    for ( int j = 0; j < ulib.model.nNormalAdjustableParams; j++ )
                        values[j] = session.RNG.uniform(ulib.adjustableParams[j].min, ulib.adjustableParams[j].max);
                    for ( int j = 0, jj = ulib.model.nNormalAdjustableParams; j < ulib.model.nOptions; j++, jj++ )
                        values[jj] = ulib.adjustableParams[jj].initial * (((optionBits>>j) & 0x1) * -1);
                }
            } else // Copy from first stim
                for ( int j = 0; j < nParams; j++ )
                    values[j] = ulib.adjustableParams[j][i % nModelsPerStim];

            ++detIdx;
            baseValues = values;
        } else {
            // Add a sigma-sized step to one parameter at a time
            AdjustableParam &p = ulib.adjustableParams[*detIdx];
            if ( *detIdx >= ulib.model.nNormalAdjustableParams )
                values[*detIdx] *= -1;
            else
                values[*detIdx] += p.sigma;
            if ( ++detIdx == detuneIndices.end() )
                detIdx = detuneIndices.begin();
        }

        for ( size_t j = 0; j < values.size(); j++ )
            ulib.adjustableParams[j][i] = values[j];

        if ( !session.runData().VC )
            values = baseValues;
    }
}

void Wavegen::settle_models()
{
    const RunData &rd = session.runData();

    ulib.setSingularRund();
    ulib.setRundata(0, rd);
    ulib.integrator = rd.integrator;
    ulib.simCycles = rd.simCycles;

    ulib.setSingularTarget();
    ulib.targetOffset[0] = 0;

    ulib.setSingularStim();
    ulib.stim[0].baseV = session.stimulationData().baseV;

    if ( session.runData().VC ) {
        ulib.assignment = ulib.assignment_base | ASSIGNMENT_SETTLE_ONLY | ASSIGNMENT_MAINTAIN_STATE;
        ulib.push();
        ulib.run();
    } else {
        ulib.stim[0].clear();
        ulib.stim[0].duration = ulib.iSettleDuration[0];
        ulib.obs[0].start[0] = 0;
        ulib.obs[0].stop[0] = ulib.iSettleDuration[0];
        ulib.iSettleDuration[0] = 0;
        ulib.assignment = ulib.assignment_base | ASSIGNMENT_MAINTAIN_STATE | ASSIGNMENT_PATTERNCLAMP
                | ((unsigned int)(searchd.trajectoryLength - 1) << ASSIGNMENT_PC_PIN__SHIFT);
        ulib.push();
        ulib.run();
    }
}

void Wavegen::pushStimsAndObserve(const std::vector<iStimulation> &stims, int nModelsPerStim, int blankCycles)
{
    ulib.setSingularStim(false);
    size_t nTotalStims = stims.size()*nModelsPerStim;
    for ( size_t i = 0, end = std::min(ulib.NMODELS, nTotalStims); i < end; i++ ) {
        ulib.stim[i] = stims[i / nModelsPerStim];
    }
    if ( nTotalStims < ulib.NMODELS )
        for ( size_t i = nTotalStims; i < ulib.NMODELS; i++ )
            ulib.stim[i].duration = 0;
    ulib.push(ulib.stim);
    ulib.observe_no_steps(blankCycles);
}

std::vector<double> Wavegen::getDeltabar()
{
    int nModelsPerStim = searchd.nTrajectories * searchd.trajectoryLength;
    std::vector<iStimulation> stims(ulib.NMODELS / nModelsPerStim);
    size_t nParams = ulib.adjustableParams.size();
    std::vector<double> deltabar(nParams, 0);

    ulib.iSettleDuration[0] = 0;
    ulib.push(ulib.iSettleDuration);

    for ( size_t runIdx = 0; runIdx < searchd.nDeltabarRuns; runIdx++ ) {
        for ( iStimulation &stim : stims )
            stim = session.wavegen().getRandomStim(stimd, istimd);
        pushStimsAndObserve(stims, nModelsPerStim, searchd.cluster.blank / session.runData().dt);
        ulib.assignment = ulib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES;
        if ( session.runData().VC )
            ulib.assignment |= ASSIGNMENT_TIMESERIES_COMPARE_PREVTHREAD;
        else
            ulib.assignment |= ASSIGNMENT_TIMESERIES_COMPARE_NONE | ASSIGNMENT_PATTERNCLAMP | ASSIGNMENT_PC_REPORT_PIN |
                    ((unsigned int)(searchd.trajectoryLength - 1) << ASSIGNMENT_PC_PIN__SHIFT);
        ulib.run();

        std::vector<double> dbar = ulib.find_deltabar(searchd.trajectoryLength, searchd.nTrajectories);
        for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ )
            deltabar[paramIdx] += dbar[paramIdx];
    }

    std::cout << "Mean current delta from detuning (nA):" << '\n';
    for ( size_t paramIdx = 0; paramIdx < nParams; paramIdx++ ) {
        deltabar[paramIdx] /= searchd.nDeltabarRuns;
        std::cout << ulib.adjustableParams[paramIdx].name << ":\t" << deltabar[paramIdx] << std::endl;
    }
    return deltabar;
}

/// Radix sort by MAPElite::bin, starting from dimension @a firstDimIdx upwards.
/// Returns: An iterator to the final element of the sorted list.
std::forward_list<MAPElite>::iterator radixSort(std::forward_list<MAPElite> &list, const std::vector<MAPEDimension> &dims, int firstDimIdx = 0)
{
    auto tail = list.before_begin();
    for ( int dimIdx = dims.size()-1; dimIdx >= firstDimIdx; dimIdx-- ) {
        size_t nBuckets = dims[dimIdx].resolution;
        std::vector<std::forward_list<MAPElite>> buckets(nBuckets);
        std::vector<std::forward_list<MAPElite>::iterator> tails(nBuckets);
        for ( size_t bucketIdx = 0; bucketIdx < nBuckets; bucketIdx++ )
            tails[bucketIdx] = buckets[bucketIdx].before_begin();

        // Sort into buckets, maintaining existing order
        while ( !list.empty() ) {
            const size_t bucketIdx = list.begin()->bin[dimIdx];
            buckets[bucketIdx].splice_after(tails[bucketIdx], list, list.before_begin());
            ++tails[bucketIdx];
        }

        // Consolidate back into a contiguous list
        tail = list.before_begin();
        for ( size_t bucketIdx = 0; bucketIdx < nBuckets; bucketIdx++ ) {
            if ( !buckets[bucketIdx].empty() ) {
                list.splice_after(tail, buckets[bucketIdx]);
                tail = tails[bucketIdx];
            }
        }
    }

    // Guard against loop not being entered at all (1D archive?)
    if ( tail == list.before_begin() && !list.empty() ) {
        auto after_tail = tail;
        for ( ++after_tail; after_tail != list.end(); ++after_tail )
            ++tail;
    }
    return tail;
}

std::forward_list<MAPElite> Wavegen::sortCandidates(std::vector<std::forward_list<MAPElite>> &candidates_by_param, const std::vector<MAPEDimension> &dims)
{
    std::forward_list<MAPElite> ret;
    auto tail = ret.before_begin();
    for ( std::forward_list<MAPElite> l : candidates_by_param ) {
        auto next_tail = radixSort(l, dims, 1); // NOTE: Expects EE_ParamIdx as first dimension.
        ret.splice_after(tail, l);
        using std::swap;
        swap(tail, next_tail);
    }
    return ret;
}

void Wavegen::insertCandidates(std::forward_list<MAPElite> candidates)
{
    // Insert into archive
    int nInserted = 0, nReplaced = 0;
    auto archIter = current.elites.begin();
    for ( auto candIter = candidates.begin(); candIter != candidates.end(); candIter++ ) {
        while ( archIter != current.elites.end() && *archIter < *candIter ) // Advance to the first archive element with coords >= candidate
            ++archIter;
        if ( archIter == current.elites.end() || *candIter < *archIter ) { // No elite at candidate's coords, insert implicitly
            archIter = current.elites.insert(archIter, std::move(*candIter));
            ++nInserted;
        } else { // preexisting elite at the candidate's coords, compete
            nReplaced += archIter->compete(*candIter);
        }
    }

    current.nInsertions.push_back(nInserted);
    current.nReplacements.push_back(nReplaced);
    current.nElites.push_back(current.elites.size());
}
