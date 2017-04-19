#include "errorprofiler.h"
#include "supportcode.h"
#include <cmath>
#include <cassert>
#include "util.h"
#include "session.h"

ErrorProfile::ErrorProfile(Session &session) :
    lib(session.project.experiment()),
    m_permutations(lib.adjustableParams.size()),
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
        perm.n = lib.project.expNumCandidates();
    if ( perm.n == 1 )
        perm.min = perm.max = 0;
    m_permutations[param] = perm;
}

void ErrorProfile::setSource(WaveSource src)
{
    assert(errors.empty() /* No changes to settings during or after profiling */);
    m_stimulations.clear();
    m_src = src;
    if ( src.type == WaveSource::Selection ) {
        const WavegenSelection &sel = *src.selection();
        m_stimulations.reserve(sel.size());
        std::vector<size_t> idx(sel.ranges.size());
        for ( size_t i = 0; i < sel.size(); i++ ) {
            for ( int j = sel.ranges.size() - 1; j >= 0; j-- ) {
                if ( ++idx[j] % sel.width(j) == 0 )
                    idx[j] = 0;
                else
                    break;
            }
            bool ok;
            auto it = sel.data_relative(idx, &ok);
            if ( ok )
                m_stimulations.push_back(it->wave);
        }
    } else {
        m_stimulations.reserve(src.archive().elites.size());
        for ( MAPElite const& e : src.archive().elites )
            m_stimulations.push_back(e.wave);
    }
}

void ErrorProfile::setStimulations(std::vector<Stimulation> &&stim)
{
    assert(errors.empty() /* No changes to settings during or after profiling */);
    m_stimulations = std::move(stim);
    m_src = WaveSource();
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



QDataStream &operator<<(QDataStream &os, const ErrorProfile &ep)
{
    os << quint32(ep.m_permutations.size());
    for ( const ErrorProfile::Permutation &p : ep.m_permutations ) {
        os << quint32(p.n) << p.min << p.max << p.fixed << p.value;
    }

    os << quint32(ep.m_stimulations.size());
    for ( const Stimulation &s : ep.m_stimulations ) {
        os << s;
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

    is >> stimulations_size;
    ep.m_stimulations.resize(stimulations_size);
    for ( Stimulation &s : ep.m_stimulations ) {
        is >> s;
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

    return is;
}





const QString ErrorProfiler::action = QString("generate");
const quint32 ErrorProfiler::magic = 0x2be4e5cb;
const quint32 ErrorProfiler::version = 101;

ErrorProfiler::ErrorProfiler(Session &session, DAQ *daq) :
    SessionWorker(session),
    simulator(session.project.experiment().createSimulator()),
    daq(daq ? daq : simulator),
    aborted(false)
{
    connect(this, SIGNAL(doAbort()), this, SLOT(clearAbort()));
}

ErrorProfiler::~ErrorProfiler()
{
    session.project.experiment().destroySimulator(simulator);
}

void ErrorProfiler::load(const QString &act, const QString &, QFile &results)
{
    if ( act != action )
        throw std::runtime_error(std::string("Unknown action: ") + act.toStdString());
    QDataStream is;
    quint32 ver = openLoadStream(results, is, magic);
    if ( ver < 100 || ver > version )
        throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

    m_profiles.push_back(ErrorProfile(session));
    m_profiles.back().version = ver;
    is >> m_profiles.back();
}

void ErrorProfiler::abort()
{
    aborted = true;
    emit doAbort();
}

bool ErrorProfiler::queueProfile(ErrorProfile &&p)
{
    try {
        p.errors.resize(p.stimulations().size());
        for ( auto &err : p.errors )
            err.resize(p.numPermutations());
    } catch (std::bad_alloc) {
        return false;
    }
    m_queue.push_back(p);
    return true;
}

void ErrorProfiler::clearAbort()
{
    aborted = false;
    emit didAbort();
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
            return;
        if ( stim.duration > 0 ) {
            ep.generate(stim, *iter, daq, session.experimentData().settleDuration);
        } // else, *iter is an empty vector, as befits an empty stimulation
        iter++;
        emit progress(++i, ep.stimulations().size());
    }
    m_profiles.push_back(std::move(ep));
    emit done();

    // Save
    QFile file(session.log(this, action, "Descriptive metadata goes here"));
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) )
        os << m_profiles.back();
}
