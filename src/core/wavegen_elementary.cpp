#include "wavegen.h"
#include "session.h"

void Wavegen::prepare_EE_models()
{
    int nParams = ulib.adjustableParams.size(), nextParam = 0;
    bool initial = true;
    std::vector<scalar> values(nParams);
    size_t nModelsPerStim = searchd.nTrajectories * searchd.trajectoryLength;
    for ( size_t i = 0; i < ulib.NMODELS; i++ ) {
        if ( i % searchd.trajectoryLength == 0 ) { // Pick a starting point
            // For the benefit of rerandomiseParameters && useBaseParameters, provide each new stim with a trajectory from base model
            if ( i % nModelsPerStim == 0 ) {
                initial = true;
                nextParam = 0;
            }
            if ( searchd.useBaseParameters && initial ) // Reset to base model
                for ( int j = 0; j < nParams; j++ )
                    values[j] = ulib.adjustableParams[j].initial;
            else if ( searchd.rerandomiseParameters || i < nModelsPerStim ) // Randomise uniformly
                for ( int j = 0; j < nParams; j++ )
                    values[j] = session.RNG.uniform(ulib.adjustableParams[j].min, ulib.adjustableParams[j].max);
            else // Copy from first stim
                for ( int j = 0; j < nParams; j++ )
                    values[j] = ulib.adjustableParams[j][i % nModelsPerStim];
        } else {
            // Add a sigma-sized step to one parameter at a time
            values[nextParam] += ulib.adjustableParams[nextParam].sigma;
            nextParam = (nextParam+1) % nParams;
            if ( nextParam == 0 )
                initial = false;
        }

        for ( size_t j = 0; j < values.size(); j++ )
            ulib.adjustableParams[j][i] = values[j];
    }
}

void Wavegen::settle_EE_models()
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

    ulib.assignment = ulib.assignment_base | ASSIGNMENT_SETTLE_ONLY | ASSIGNMENT_MAINTAIN_STATE;
    ulib.push();
    ulib.run();
}

void Wavegen::pushStimsAndObserve(const std::vector<iStimulation> &stims, int nModelsPerStim, int blankCycles)
{
    ulib.setSingularStim(false);
    for ( size_t i = 0; i < ulib.NMODELS; i++ ) {
        ulib.stim[i] = stims[i / nModelsPerStim];
    }
    ulib.push(ulib.stim);
    ulib.observe_no_steps(blankCycles);
}

QVector<double> Wavegen::getDeltabar()
{
    int nModelsPerStim = searchd.nTrajectories * searchd.trajectoryLength;
    std::vector<iStimulation> stims(ulib.NMODELS / nModelsPerStim);
    for ( iStimulation &stim : stims )
        stim = session.wavegen().getRandomStim(stimd, istimd);

    ulib.iSettleDuration[0] = 0;
    ulib.push(ulib.iSettleDuration);

    pushStimsAndObserve(stims, nModelsPerStim, session.gaFitterSettings().cluster_blank_after_step / session.runData().dt);

    ulib.assignment = ulib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_PREVTHREAD;
    ulib.run();

    std::vector<double> dbar = ulib.find_deltabar(searchd.trajectoryLength, searchd.nTrajectories, stims[0].duration);
    return QVector<double>::fromStdVector(dbar);
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
