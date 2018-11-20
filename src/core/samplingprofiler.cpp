#include "samplingprofiler.h"
#include "session.h"
#include "clustering.h"

const QString SamplingProfiler::action = QString("generate");
const quint32 SamplingProfiler::magic = 0x674b5c54;
const quint32 SamplingProfiler::version = 100;

QDataStream &operator<<(QDataStream &os, const SamplingProfiler::Profile &);
QDataStream &operator>>(QDataStream &is, SamplingProfiler::Profile &);

SamplingProfiler::SamplingProfiler(Session &session) :
    SessionWorker(session),
    lib(session.project.universal())
{
    qRegisterMetaType<Profile>();
}

void SamplingProfiler::load(const QString &act, const QString &, QFile &results, Result r)
{
    if ( act == action ) {
        QDataStream is;
        quint32 ver = openLoadStream(results, is, magic);
        if ( ver < 100 || ver > version )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        m_profiles.push_back(Profile(r));
        m_profiles.back().src.session = &session;
        is >> m_profiles.back();
    } else {
        throw std::runtime_error(std::string("Unknown action: ") + act.toStdString());
    }
}

void SamplingProfiler::generate(SamplingProfiler::Profile prof)
{
    session.queue(actorName(), action, prof.src.prettyName(), new Profile(std::move(prof)));
}

int SamplingProfiler::generate_observations(const Profile &prof, const std::vector<iStimulation> &stims, std::vector<iObservations> &observations)
{
    std::vector<double> deltabar = prof.src.archive()->deltabar.toStdVector();
    const RunData &rd = session.runData();
    int blankCycles = session.gaFitterSettings().cluster_blank_after_step/rd.dt;
    int secLen = session.gaFitterSettings().cluster_fragment_dur/rd.dt;
    int minLen = session.gaFitterSettings().cluster_min_dur/rd.dt;
    double dotp_threshold = session.gaFitterSettings().cluster_threshold;
    auto elit = prof.src.elites().begin();
    int maxObsDuration = 0;

    // Generate diff traces one GPU load at a time
    size_t nTotalStims = stims.size(), nMaxDiffStims = maxDetunedDiffTraceStims(lib);
    for ( size_t stimOffset = 0; stimOffset < nTotalStims; stimOffset += nMaxDiffStims ) {
        size_t nStims = std::min(nTotalStims-stimOffset, nMaxDiffStims);
        std::vector<iStimulation> stims_chunk;
        stims_chunk.insert(stims_chunk.end(), stims.begin()+stimOffset, stims.begin()+stimOffset+nStims);
        auto pDelta = getDetunedDiffTraces(stims_chunk, lib, rd);

        // Cluster on CPU
        for ( size_t i = 0; i < nStims; i++ ) {
            auto clusters = constructClusters(stims_chunk[i], pDelta[i], lib.NMODELS, blankCycles, deltabar, secLen, dotp_threshold, minLen);
            const iObservations &obs = elit->obs;
            ++elit;

            // Find and choose the cluster with the greatest observational overlap with the GPU's version
            int bestOverlap = -1, bestOverlapIdx = 0;
            for ( size_t j = 0; j < clusters.size(); j++ ) {
                size_t k = 0;
                int overlap = 0;
                for ( const Section &sec : clusters[j] ) {
                    while ( k < iObservations::maxObs && sec.start > obs.stop[k] )
                        ++k;
                    if ( k == iObservations::maxObs )
                        break;
                    if ( sec.end < obs.start[k] )
                        continue;
                    overlap += std::min(obs.stop[k], sec.end) - std::max(obs.start[k], sec.start);
                }
                if ( overlap > bestOverlap ) {
                    bestOverlap = overlap;
                    bestOverlapIdx = j;
                }
            }
            const std::vector<Section> &cluster = clusters[bestOverlapIdx];

            // Turn best-fit cluster into iObservations
            size_t nObs = cluster.size();
            int obsDuration = 0;
            if ( nObs > iObservations::maxObs ) {
                std::vector<int> start(nObs), stop(nObs);
                for ( size_t j = 0; j < nObs; j++ ) {
                    start[j] = cluster[j].start;
                    stop[j] = cluster[j].end;
                }
                while ( nObs > iObservations::maxObs )
                    reduceObsCount(start.data(), stop.data(), nObs--, minLen);

                for ( size_t j = 0; j < nObs; j++ ) {
                    observations[stimOffset + i].start[j] = start[j];
                    observations[stimOffset + i].stop[j] = stop[j];
                    obsDuration += stop[j] - start[j];
                }
            } else {
                for ( size_t j = 0; j < nObs; j++ ) {
                    observations[stimOffset + i].start[j] = cluster[j].start;
                    observations[stimOffset + i].stop[j] = cluster[j].end;
                    obsDuration += cluster[j].end - cluster[j].start;
                }
            }
            maxObsDuration = std::max(maxObsDuration, obsDuration);
        }
    }
    return maxObsDuration;
}

bool SamplingProfiler::execute(QString action, QString, Result *res, QFile &file)
{
    clearAbort();
    if ( action != this->action )
        return false;

    const RunData &rd = session.runData();
    Profile &prof = *static_cast<Profile*>(res);
    std::vector<iStimulation> stims = prof.src.iStimulations(rd.dt);
    std::vector<iObservations> obs(stims.size(), {{}, {}});
    int maxObsDuration = 0;

    if ( prof.src.archive() && prof.src.archive()->param == -1 ) {
        maxObsDuration = generate_observations(prof, stims, obs);
    } else {
        for ( size_t i = 0; i < stims.size(); i++ ) {
            obs[i].start[0] = stims[i].tObsBegin;
            obs[i].stop[0] = stims[i].tObsEnd;
            maxObsDuration = std::max(maxObsDuration, stims[i].tObsEnd-stims[i].tObsBegin);
        }
    }

    // Populate
    for ( size_t param = 0; param < lib.adjustableParams.size(); param++ ) {
        AdjustableParam &p = lib.adjustableParams[param];
        std::function<scalar(void)> uniform = [=](){ return session.RNG.uniform<scalar>(prof.value1[param], prof.value2[param]); };
        std::function<scalar(void)> gaussian = [=](){ return session.RNG.variate<scalar, std::normal_distribution>(prof.value1[param], prof.value2[param]); };
        auto gen = prof.uniform[param] ? uniform : gaussian;
        if ( param == prof.target ) {
            for ( size_t pair = 0; pair < lib.NMODELS/2; pair++ ) {
                double value = gen();
                p[2*pair] = value;
                p[2*pair+1] = value + prof.sigma;
            }
        } else {
            for ( size_t pair = 0; pair < lib.NMODELS/2; pair++ ) {
                double value = gen();
                p[2*pair] = value;
                p[2*pair+1] = value;
            }
        }
    }

    // Set up library
    lib.setSingularRund();
    lib.simCycles = rd.simCycles;
    lib.integrator = rd.integrator;
    lib.setRundata(0, rd);

    lib.setSingularStim();
    lib.stim[0].baseV = NAN;
    lib.obs[0] = {{}, {}};

    lib.resizeOutput(maxObsDuration);

    lib.push();

    unsigned int assignment = lib.assignment_base
            | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPACT | ASSIGNMENT_TIMESERIES_COMPARE_NONE;

    size_t total = stims.size();
    prof.gradient.resize(total);
    prof.accuracy.resize(total);

    // Run
    for ( size_t i = 0; i < total; i++ ) {
        if ( isAborted() ) {
            delete res;
            return false;
        }

        lib.stim[0] = stims[i];
        lib.obs[0] = obs[i];
        lib.push(lib.stim);
        lib.push(lib.obs);

        // Settle, if necessary - lib maintains settled state
        if ( i == 0 || stims[i-1].baseV != stims[i].baseV ) {
            lib.iSettleDuration[0] = rd.settleDuration / rd.dt;
            lib.push(lib.iSettleDuration);

            lib.assignment = assignment | ASSIGNMENT_SETTLE_ONLY;
            lib.run();

            lib.iSettleDuration[0] = 0;
            lib.push(lib.iSettleDuration);
        }

        // Stimulate
        lib.assignment = assignment;
        lib.run();

        // Profile
        int nSamples = stims[i].tObsEnd - stims[i].tObsBegin;
        lib.profile(nSamples, prof.samplingInterval, prof.target, prof.accuracy[i], prof.gradient[i]);
        prof.gradient[i] /= prof.sigma;
        emit progress(i, total);
    }

    m_profiles.push_back(std::move(prof));
    emit done();

    // Save
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) )
        os << m_profiles.back();

    delete res;
    return true;
}

QDataStream &operator<<(QDataStream &os, const SamplingProfiler::Profile &p)
{
    os << p.src << quint32(p.target) << p.sigma << quint32(p.samplingInterval);
    os << p.uniform << p.value1 << p.value2;
    os << p.gradient << p.accuracy;
    return os;
}

QDataStream &operator>>(QDataStream &is, SamplingProfiler::Profile &p)
{
    quint32 target, samplingInterval;
    is >> p.src >> target >> p.sigma >> samplingInterval;
    p.target = target;
    p.samplingInterval = samplingInterval;
    is >> p.uniform >> p.value1 >> p.value2;
    is >> p.gradient >> p.accuracy;
    return is;
}

SamplingProfiler::Profile::Profile(WaveSource src, size_t target, Result r) :
    Result(r),
    src(src),
    target(target),
    sigma(src.session->project.model().adjustableParams[target].sigma),
    samplingInterval(1),
    uniform(src.session->project.model().adjustableParams.size()),
    value1(uniform.size()),
    value2(uniform.size()),
    gradient(src.stimulations().size()),
    accuracy(gradient.size())
{}
