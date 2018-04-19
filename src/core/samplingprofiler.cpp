#include "samplingprofiler.h"
#include "session.h"

const QString SamplingProfiler::action = QString("generate");
const quint32 SamplingProfiler::magic = 0x674b5c54;
const quint32 SamplingProfiler::version = 100;

QDataStream &operator<<(QDataStream &os, const SamplingProfiler::Profile &);
QDataStream &operator>>(QDataStream &is, SamplingProfiler::Profile &);

SamplingProfiler::SamplingProfiler(Session &session) :
    SessionWorker(session),
    lib(session.project.profiler())
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

    Profile &prof = *static_cast<Profile*>(res);

    // Populate
    for ( size_t param = 0; param < lib.adjustableParams.size(); param++ ) {
        AdjustableParam &p = lib.adjustableParams[param];
        std::function<scalar(void)> uniform = [=](){ return session.RNG.uniform<scalar>(prof.value1[param], prof.value2[param]); };
        std::function<scalar(void)> gaussian = [=](){ return session.RNG.variate<scalar, std::normal_distribution>(prof.value1[param], prof.value2[param]); };
        auto gen = prof.uniform[param] ? uniform : gaussian;
        if ( param == prof.target ) {
            for ( size_t pair = 0; pair < lib.project.profNumPairs(); pair++ ) {
                double value = gen();
                p[2*pair] = value;
                p[2*pair+1] = value + prof.sigma;
            }
        } else {
            for ( size_t pair = 0; pair < lib.project.profNumPairs(); pair++ ) {
                double value = gen();
                p[2*pair] = value;
                p[2*pair+1] = value;
            }
        }
    }
    lib.push();

    lib.samplingInterval = prof.samplingInterval;

    iStimulation hold;
    hold.clear();
    hold.duration = lrint(session.runData().settleDuration / session.wavegenData().dt);
    hold.baseV = NAN;

    std::vector<iStimulation> stims = prof.src.iStimulations(session.wavegenData().dt);
    size_t total = stims.size();
    prof.gradient.resize(total);
    prof.accuracy.resize(total);

    // Run
    for ( size_t i = 0; i < total; i++ ) {
        if ( isAborted() ) {
            delete res;
            return false;
        }
        // Settle, if necessary - lib maintains settled state
        if ( hold.baseV != stims[i].baseV ) {
            hold.baseV = stims[i].baseV;
            lib.settle(hold);
        }

        // Stimulate
        lib.profile(stims[i], prof.target, prof.accuracy[i], prof.gradient[i]);
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

SamplingProfiler::Profile::Profile(WaveSource src, Result r) :
    Result(r),
    src(src),
    target(src.archive() ? src.archive()->param : 0),
    sigma(src.archive() ? src.session->project.model().adjustableParams[target].adjustedSigma : 1e-5),
    uniform(src.session->project.model().adjustableParams.size()),
    value1(uniform.size()),
    value2(uniform.size()),
    gradient(src.stimulations().size()),
    accuracy(gradient.size())
{}
