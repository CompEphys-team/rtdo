#include "errorprofiler.h"
#include "supportcode.h"
#include <cmath>
#include <cassert>
#include "util.h"
#include "session.h"

ErrorProfile::ErrorProfile(Session &session) :
    lib(session.project.experiment()),
    m_permutations(lib.adjustableParams.size())
{
}

void ErrorProfile::setPermutations(std::vector<ErrorProfile::Permutation> p)
{
    assert(errors.empty() /* No changes to settings during or after profiling */);
    assert(p.size() == m_permutations.size());
    for ( size_t i = 0; i < m_permutations.size(); i++ )
        setPermutation(i, p[i]);
}

void ErrorProfile::setPermutation(size_t param, ErrorProfile::Permutation perm)
{
    assert(errors.empty() /* No changes to settings during or after profiling */);
    assert(param < m_permutations.size());
    if ( perm.fixed )
        perm.n = 1;
    if ( perm.n == 0 )
        perm.n = lib.project.expNumCandidates();
    if ( perm.n == 1 )
        perm.min = perm.max = 0;
    m_permutations[param] = perm;
}

void ErrorProfile::setStimulations(std::vector<Stimulation> &&stim)
{
    assert(errors.empty() /* No changes to settings during or after profiling */);
    m_stimulations = std::move(stim);
}

size_t ErrorProfile::numPermutations() const
{
    size_t i = 1;
    for ( const Permutation &p : m_permutations )
        i *= p.n;
    return i;
}

size_t ErrorProfile::numSimulations() const
{
    size_t nCand = lib.project.expNumCandidates();
    return (numPermutations() + nCand - 1) / nCand; // Get nearest multiple of nCand, rounded up
}

double ErrorProfile::parameterValue(size_t param, size_t idx) const
{
    const AdjustableParam &para = lib.adjustableParams[param];
    const Permutation &perm = m_permutations[param];
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

std::vector<double> ErrorProfile::parameterValues(size_t param, std::vector<size_t> idx) const
{
    std::vector<double> ret(m_permutations.size());
    for ( size_t i = 0; i < ret.size(); i++ )
        ret[i] = parameterValue(param, idx[i]);
    return ret;
}

size_t ErrorProfile::parameterIndex(size_t param, double value) const
{
    const AdjustableParam &para = lib.adjustableParams[param];
    const Permutation &perm = m_permutations[param];
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

std::vector<std::vector<ErrorProfile::Profile>> ErrorProfile::profiles(size_t targetParam) const
{
    assert(!errors.empty());

    // Find the stride used to populate errors
    size_t stride = 1;
    for ( size_t param = 0; param < targetParam; param++ )
        stride *= m_permutations[param].n;

    // Prepare the vector of non-target parameter indices
    std::vector<size_t> pIdx(m_permutations.size(), 0);

    // Prepare the profile vectors and count
    size_t nProfiles = numPermutations() / m_permutations[targetParam].n;
    std::vector<std::vector<Profile>> ret(m_stimulations.size());
    for ( std::vector<Profile> &r : ret )
        r.reserve(nProfiles);

    // Populate the profile vectors
    for ( size_t i = 0, offset = 0, cluster = 0; i < nProfiles; i++ ) {
        /* The logic here mirrors the way parameters are populated during profiling:
         * A given parameter changes every (stride) field in errors, wrapping around
         * as necessary. Thus, the profiles of a parameter with stride 1 are lined up
         * in series (no interleaving); the profiles of the last permuted parameter
         * are perfectly interleaved (no wrapping around, the last profile begins before
         * the first one ends), and intermediate profiles are interleaved in clusters,
         * wrapping around every (stride * n) fields.
         */
        // Populate for each stimulation
        size_t k = 0;
        for ( std::vector<scalar> const& err : errors )
            ret[k++].push_back(Profile(err.cbegin() + offset, stride, m_permutations[targetParam].n, pIdx));

        // Increase offset and cluster (=simulation number)
        if ( ++offset % stride == 0 )
            offset = ++cluster * stride * m_permutations[targetParam].n;

        // Adjust non-target parameter indices
        for ( size_t j = 0; j < m_permutations.size(); j++ ) {
            if ( j == targetParam )
                continue;
            ++pIdx[j];
            if ( pIdx[j] < m_permutations[j].n )
                break;
            pIdx[j] = 0;
        }
    }

    return ret;
}



void ErrorProfile::generate(const Stimulation &stim, std::vector<scalar> &errors, DAQ *daq, scalar settleDuration)
{
    size_t nSimulations = numSimulations(), nPermutations = numPermutations();

    lib.reset();
    errors.resize(nPermutations);

    // Prepare all parameter values
    std::vector<double> values[m_permutations.size()];
    std::vector<size_t> pStride(m_permutations.size());
    std::vector<size_t> pIdx(m_permutations.size(), 0);
    for ( size_t param = 0, stride = 1; param < m_permutations.size(); param++ ) {
        values[param] = std::vector<double>(m_permutations[param].n);
        if ( m_permutations[param].fixed ) {
            values[param][0] = m_permutations[param].value;
        } else if ( m_permutations[param].n == 1 ) {
            values[param][0] = lib.adjustableParams[param].initial;
        } else {
            for ( size_t j = 0; j < m_permutations[param].n; j++ ) {
                values[param][j] = parameterValue(param, j);
            }
        }
        pStride[param] = stride;
        stride *= m_permutations[param].n;
    }

    for ( size_t sim = 0, offset = 0; sim < nSimulations; sim++, offset += lib.project.expNumCandidates() ) {
        size_t batchSize = lib.project.expNumCandidates();
        if ( sim == nSimulations-1 )
            batchSize = nPermutations - sim*lib.project.expNumCandidates(); // Last round does leftovers

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
        settle(stim.baseV, daq, settleDuration);

        // Stimulate
        stimulate(stim, daq);
        lib.pullErr();

        // Store errors
        for ( size_t iM = 0; iM < batchSize; iM++ ) {
            errors[iM + offset] = lib.err[iM];
        }
    }
}


void ErrorProfile::settle(scalar baseV, DAQ *daq, scalar settleDuration)
{
    // Create holding stimulation
    Stimulation I {};
    I.duration = settleDuration;
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

void ErrorProfile::stimulate(const Stimulation &stim, DAQ *daq)
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








ErrorProfiler::ErrorProfiler(Session &session, DAQ *daq) :
    SessionWorker(session),
    simulator(session.project.experiment().createSimulator()),
    daq(daq ? daq : simulator),
    aborted(false)
{
    connect(this, SIGNAL(didAbort()), this, SLOT(clearAbort()));
}

ErrorProfiler::~ErrorProfiler()
{
    session.project.experiment().destroySimulator(simulator);
}

void ErrorProfiler::load(const QString &action, const QString &args, QFile &results)
{
    // NYI
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

void ErrorProfiler::generate()
{
    ErrorProfile ep = m_queue.front();
    m_queue.pop_front();
    if ( aborted )
        return;
    ep.errors.resize(ep.stimulations().size());
    auto iter = ep.errors.begin();
    int i = 0;
    for ( Stimulation const& stim : ep.stimulations() ) {
        if ( aborted )
            break;
        if ( stim.duration > 0 ) {
            ep.generate(stim, *iter, daq, session.experimentData().settleDuration);
        } // else, *iter is an empty vector, as befits an empty stimulation
        iter++;
        emit progress(++i, ep.stimulations().size());
    }
    m_profiles.push_back(std::move(ep));
    emit done();

    // Saving NYI
    session.log(this, "generate", "Results NYI");
}
