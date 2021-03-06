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


#include "errorprofile.h"
#include "session.h"
#include <cassert>

ErrorProfile::ErrorProfile(Session &session, Result r) :
    Result(r),
    lib(session.project.universal()),
    m_permutations(lib.adjustableParams.size()),
    m_stats(lib.adjustableParams.size(), ProfileStats(session)),
    session(session)
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
        perm.n = lib.NMODELS;
    if ( perm.n == 1 )
        perm.min = perm.max = 0;
    m_permutations[param] = perm;
}

void ErrorProfile::setSource(WaveSource src)
{
    assert(errors.empty() /* No changes to settings during or after profiling */);
    m_stimulations = src.iStimulations();
    m_observations = src.observations();
    m_src = src;
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
    size_t nCand = lib.NMODELS;
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

void ErrorProfile::process_stats()
{
    for ( size_t i = 0; i < m_permutations.size(); i++ ) {
        if ( m_permutations[i].n > 1 ) {
            m_stats[i].process(this, i);
        }
    }
}



QString ErrorProfile::prettyName() const
{
    QString source = (m_src.session ? QString("%2 waves from %1").arg(m_src.prettyName()) : QString("%2 unsourced waves"))
            .arg(m_stimulations.size());
    QStringList dims;
    for ( size_t i = 0; i < m_permutations.size(); i++ ) {
        QString schema;
        if ( m_permutations[i].n == 1 ) {
            if ( m_permutations[i].fixed )
                schema = QString("%10=%1").arg(m_permutations[i].value);
            else
                continue;
        } else {
            schema = QString("%1 %10 ∈ [%2,%3]")
                    .arg(m_permutations[i].n)
                    .arg(m_permutations[i].min)
                    .arg(m_permutations[i].max);
        }
        dims << schema.arg(QString::fromStdString(lib.adjustableParams[i].name));
    }
    return QString("%1 {%2}").arg(source).arg(dims.join("; "));
}



void ErrorProfile::generate(const iStimulation &stim, const iObservations &obs, std::vector<scalar> &errors)
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

    for ( size_t sim = 0, offset = 0; sim < nSimulations; sim++, offset += lib.NMODELS ) {
        size_t batchSize = lib.NMODELS;
        if ( sim == nSimulations-1 )
            batchSize = nPermutations - sim*lib.NMODELS; // Last round does leftovers

        // Populate lib.adjustableParams from values
        for ( size_t param = 0; param < lib.adjustableParams.size(); param++ ) {
            for ( size_t iM = 0; iM < batchSize; ) {
                lib.adjustableParams[param][iM] = values[param][pIdx[param]];
                if ( ++iM % pStride[param] == 0 )
                    pIdx[param] = (pIdx[param]+1) % values[param].size();
            }
        }

        // Stimulate
        session.profiler().stimulate(stim, obs);
        lib.pullSummary();

        // Store errors
        for ( size_t iM = 0; iM < batchSize; iM++ ) {
            errors[iM + offset] = std::sqrt(lib.summary[iM]); // RMSE (profiler assigns SUMMARY_AVERAGE, so only sqrt needed)
        }
    }
}



QDataStream &operator<<(QDataStream &os, const ErrorProfile &ep)
{
    os << quint32(ep.m_permutations.size());
    for ( const ErrorProfile::Permutation &p : ep.m_permutations ) {
        os << quint32(p.n) << p.min << p.max << p.fixed << p.value;
    }

    os << quint32(ep.errors.size());
    for ( const std::vector<scalar> err : ep.errors ) {
        os << quint32(err.size());
        for ( const scalar &e : err ) {
            os << e;
        }
    }

    bool hasSrc = ep.m_src.session != nullptr;
    os << hasSrc;
    if ( hasSrc )
        os << ep.m_src;

    for ( size_t i = 0; i < ep.m_stats.size(); i++ ) {
        if ( ep.m_permutations[i].n > 1 ) {
            os << ep.m_stats[i];
        }
    }

    return os;
}

QDataStream &operator>>(QDataStream &is, ErrorProfile &ep)
{
    quint32 permutations_size, stimulations_size, errors_size, err_size, n;

    is >> permutations_size;
    ep.m_permutations.resize(permutations_size);
    for ( ErrorProfile::Permutation &p : ep.m_permutations ) {
        is >> n >> p.min >> p.max >> p.fixed >> p.value;
        p.n = n;
    }

    if ( ep.version < 103 ) {
        is >> stimulations_size;
        ep.m_stimulations.reserve(stimulations_size);
        ep.m_observations.resize(stimulations_size, {{}, {}});
        Stimulation s;
        double dt = ep.session.runData(ep.resultIndex).dt;
        for ( quint32 i = 0; i < stimulations_size; i++ ) {
            is >> s;
            ep.m_stimulations.emplace_back(s, dt);
            ep.m_observations[i].start[0] = s.tObsBegin/dt;
            ep.m_observations[i].stop[0] = s.tObsEnd/dt;
        }
    }

    is >> errors_size;
    ep.errors.resize(errors_size);
    for ( std::vector<scalar> &err : ep.errors ) {
        is >> err_size;
        err.resize(err_size);
        for ( scalar &e : err ) {
            is >> e;
        }
    }

    if ( ep.version >= 101 ) {
        bool hasSrc;
        is >> hasSrc;
        if ( hasSrc ) {
            ep.m_src.session =& ep.session;
            is >> ep.m_src;
        }
    }

    if ( ep.version >= 102 ) {
        for ( size_t i = 0; i < ep.m_permutations.size(); i++ ) {
            if ( ep.m_permutations[i].n > 1 ) {
                is >> ep.m_stats[i];
            }
        }
    } else {
        ep.process_stats();
    }

    if ( ep.version >= 103 ) {
        ep.m_stimulations = ep.m_src.iStimulations();
        ep.m_observations = ep.m_src.observations();
    }

    return is;
}
