#include "errorprofiler.h"
#include "supportcode.h"
#include <cmath>
#include <cassert>
#include "util.h"

ErrorProfiler::ErrorProfiler(ExperimentLibrary &lib, DAQ *daq) :
    lib(lib),
    permutations(lib.adjustableParams.size()),
    simulator(lib.createSimulator()),
    daq(daq ? daq : simulator),
    aborted(false)
{
    connect(this, SIGNAL(didAbort()), this, SLOT(clearAbort()));
}

ErrorProfiler::~ErrorProfiler()
{
    lib.destroySimulator(simulator);
}

void ErrorProfiler::abort()
{
    aborted = true;
    emit didAbort();
}

void ErrorProfiler::clearAbort()
{
    aborted = false;
}

void ErrorProfiler::setPermutations(std::vector<ErrorProfiler::Permutation> p)
{
    assert( p.size() == lib.adjustableParams.size() );

    permutations = p;
    errors.clear();

    for ( Permutation &perm : permutations ) {
        if ( perm.fixed )
            perm.n = 1;
        if ( perm.n == 0 )
            perm.n = lib.expd.numCandidates;
        if ( perm.n == 1 )
            perm.min = perm.max = 0;
    }
}

size_t ErrorProfiler::getNumPermutations()
{
    size_t i = 1;
    for ( const Permutation &p : permutations )
        i *= p.n;
    return i;
}

size_t ErrorProfiler::getNumSimulations()
{
    size_t nCand = lib.expd.numCandidates;
    return (getNumPermutations() + nCand - 1) / nCand; // Get nearest multiple of nCand, rounded up
}

void ErrorProfiler::setStimulations(std::vector<Stimulation> stims)
{
    stimulations = stims;
}

double ErrorProfiler::getParameterValue(size_t param, size_t idx)
{
    const AdjustableParam &para = lib.adjustableParams[param];
    const Permutation &perm = permutations[param];
    if ( perm.fixed )
        return perm.value;
    else if ( perm.n == 1 )
        return para.initial;
    else {
        auto distribution = para.multiplicative ? logSpace : linSpace;
        if ( perm.min == 0 && perm.max == 0 )
            return distribution(para.min, para.max, perm.n, idx);
        else
            return distribution(perm.min, perm.max, perm.n, idx);
    }
}

size_t ErrorProfiler::getParameterIndex(size_t param, double value)
{
    const AdjustableParam &para = lib.adjustableParams[param];
    const Permutation &perm = permutations[param];
    if ( perm.fixed || perm.n == 1 )
        return 0;
    else {
        size_t ret;
        auto inverse = para.multiplicative ? logSpaceInverse : linSpaceInverse;
        if ( perm.min == 0 && perm.max == 0 )
            ret = inverse(para.min, para.max, perm.n, value);
        else
            ret = inverse(perm.min, perm.max, perm.n, value);

        return ret >= perm.n ? perm.n-1 : ret;
    }
}


std::vector<ErrorProfiler::Profile> ErrorProfiler::getProfiles(size_t targetParam, const std::vector<scalar> &profile)
{
    if ( profile.size() != getNumPermutations() )
        throw std::runtime_error("Profile does not match permutation settings.");

    // Find the stride used to populate errors
    size_t stride = 1;
    for ( size_t param = 0; param < targetParam; param++ )
        stride *= permutations[param].n;

    // Prepare the vector of non-target parameter indices
    std::vector<size_t> pIdx(permutations.size(), 0);

    // Prepare the vector and count
    size_t nProfiles = getNumPermutations() / permutations[targetParam].n;
    std::vector<ErrorProfiler::Profile> ret;
    ret.reserve(nProfiles);

    // Populate the vector of Profiles
    for ( size_t i = 0, offset = 0, cluster = 0; i < nProfiles; i++ ) {
        /* The logic here mirrors the way parameters are populated during profiling:
         * A given parameter changes every (stride) field in errors, wrapping around
         * as necessary. Thus, the profiles of a parameter with stride 1 are lined up
         * in series (no interleaving); the profiles of the last permuted parameter
         * are perfectly interleaved (no wrapping around, the last profile begins before
         * the first one ends), and intermediate profiles are interleaved in clusters,
         * wrapping around every (stride * n) fields.
         */
        ret.push_back(Profile(profile.cbegin() + offset, stride, permutations[targetParam].n, pIdx));

        if ( ++offset % stride == 0 )
            offset = ++cluster * stride * permutations[targetParam].n;

        for ( size_t j = 0; j < permutations.size(); j++ ) {
            if ( j == targetParam )
                continue;
            ++pIdx[j];
            if ( pIdx[j] < permutations[j].n )
                break;
            pIdx[j] = 0;
        }
    }

    return ret;
}

void ErrorProfiler::profile()
{
    if ( aborted )
        return;
    using std::swap;
    profiles = std::list<std::vector<scalar>>(stimulations.size());
    auto iter = profiles.begin();
    int i = 0;
    for ( Stimulation const& stim : stimulations ) {
        if ( aborted )
            break;
        if ( stim.duration > 0 ) {
            profile(stim);
            swap(*iter, errors);
        } // else, *iter is an empty vector, as befits an empty stimulation
        iter++;
        emit profileComplete(i++);
    }
    emit done();
}

void ErrorProfiler::profile(const Stimulation &stim)
{
    size_t numSimulations = getNumSimulations(), numPermutations = getNumPermutations();

    lib.reset();
    errors.resize(numPermutations);

    // Prepare all parameter values
    std::vector<double> values[permutations.size()];
    std::vector<size_t> pStride(permutations.size());
    std::vector<size_t> pIdx(permutations.size(), 0);
    for ( size_t param = 0, stride = 1; param < permutations.size(); param++ ) {
        values[param] = std::vector<double>(permutations[param].n);
        if ( permutations[param].fixed ) {
            values[param][0] = permutations[param].value;
        } else if ( permutations[param].n == 1 ) {
            values[param][0] = lib.adjustableParams[param].initial;
        } else {
            for ( size_t j = 0; j < permutations[param].n; j++ ) {
                values[param][j] = getParameterValue(param, j);
            }
        }
        pStride[param] = stride;
        stride *= permutations[param].n;
    }

    for ( size_t sim = 0, offset = 0; sim < numSimulations; sim++, offset += lib.expd.numCandidates ) {
        size_t batchSize = lib.expd.numCandidates;
        if ( sim == numSimulations-1 )
            batchSize = numPermutations - sim*lib.expd.numCandidates; // Last round does leftovers

        // Populate lib.adjustableParams from values
        for ( size_t param = 0; param < lib.adjustableParams.size(); param++ ) {
            for ( size_t iM = 0; iM < batchSize; ) {
                lib.adjustableParams[param][iM] = values[param][pIdx[param]];
                if ( ++iM % pStride[param] == 0 )
                    pIdx[param] = (pIdx[param]+1) % values[param].size();
            }
        }

        // Reset err
        for ( size_t iM = 0; iM < batchSize; iM++ )
            lib.err[iM] = 0;

        // Settle
        lib.push();
        settle(stim.baseV);

        // Stimulate
        stimulate(stim);
        lib.pullErr();

        // Store errors
        for ( size_t iM = 0; iM < batchSize; iM++ ) {
            errors[iM + offset] = lib.err[iM];
        }
    }
}


void ErrorProfiler::settle(scalar baseV)
{
    // Create holding stimulation
    Stimulation I {};
    I.duration = lib.expd.settleDuration;
    I.baseV = baseV;

    // Set up library
    lib.t = 0.;
    lib.iT = 0;
    lib.getErr = false;
    lib.VC = true;
    lib.Vmem = I.baseV;

    // Set up DAQ
    daq->reset();
    daq->run(I);

    // Run
    while ( lib.t < I.duration ) {
        daq->next();
        lib.step();
    }

    daq->reset();
}

void ErrorProfiler::stimulate(const Stimulation &stim)
{
    // Set up library
    lib.t = 0.;
    lib.iT = 0;
    lib.VC = true;

    // Set up DAQ
    daq->reset();
    daq->run(stim);

    // Stimulate both
    while ( lib.t < stim.duration ) {
        daq->next();
        lib.Imem = daq->current;
        lib.Vmem = getCommandVoltage(stim, lib.t);
        lib.getErr = (lib.t > stim.tObsBegin && lib.t < stim.tObsEnd);
        lib.step();
    }

    daq->reset();
}
