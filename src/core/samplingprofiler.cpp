#include "samplingprofiler.h"
#include "session.h"

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

bool SamplingProfiler::execute(QString action, QString, Result *res, QFile &file)
{
    clearAbort();
    if ( action != this->action )
        return false;

    const RunData &rd = session.runData();
    Profile &prof = *static_cast<Profile*>(res);
    std::vector<iStimulation> stims = prof.src.iStimulations(rd.dt);
    std::vector<iObservations> obs = prof.src.observations(rd.dt);

    int maxObsDuration = 0;
    for ( const iObservations &o : obs ) {
        int obsDuration = 0;
        for ( size_t i = 0; i < iObservations::maxObs; i++ )
            obsDuration += o.stop[i] - o.start[i];
        maxObsDuration = std::max(maxObsDuration, obsDuration);
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

    prof.gradient.resize(stims.size());
    prof.accuracy.resize(stims.size());

    // Run
    for ( size_t i = 0; i < stims.size(); i++ ) {
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
        emit progress(i, stims.size());
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
