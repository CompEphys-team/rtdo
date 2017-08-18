#include "samplingprofiler.h"
#include "session.h"

const QString SamplingProfiler::action = QString("generate");
const quint32 SamplingProfiler::magic = 0x674b5c54;
const quint32 SamplingProfiler::version = 100;

QDataStream &operator<<(QDataStream &os, const SamplingProfiler::Profile &);
QDataStream &operator>>(QDataStream &is, SamplingProfiler::Profile &);

SamplingProfiler::SamplingProfiler(Session &session) :
    SessionWorker(session),
    lib(session.project.profiler()),
    aborted(false)
{
    qRegisterMetaType<Profile>();
    connect(this, SIGNAL(doAbort()), this, SLOT(clearAbort()));
}

void SamplingProfiler::load(const QString &act, const QString &, QFile &results)
{
    if ( act == action ) {
        QDataStream is;
        quint32 ver = openLoadStream(results, is, magic);
        if ( ver < 100 || ver > version )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        m_profiles.push_back(Profile());
        m_profiles.back().src.session = &session;
        is >> m_profiles.back();
    } else {
        throw std::runtime_error(std::string("Unknown action: ") + act.toStdString());
    }
}

void SamplingProfiler::abort()
{
    aborted = true;
    emit doAbort();
}

void SamplingProfiler::clearAbort()
{
    aborted = false;
    emit didAbort();
}

void SamplingProfiler::generate(SamplingProfiler::Profile prof)
{
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

    Stimulation hold;
    hold.clear();
    hold.duration = session.runData().settleDuration;
    hold.baseV = NAN;

    std::vector<Stimulation> stims = prof.src.stimulations();
    size_t total = stims.size();
    prof.gradient.resize(total);
    prof.accuracy.resize(total);

    // Run
    for ( size_t i = 0; i < total && !aborted; i++ ) {
        // Settle, if necessary - lib maintains settled state
        if ( hold.baseV != stims[i].baseV ) {
            hold.baseV = stims[i].baseV;
            lib.settle(hold);
        }

        // Stimulate
        lib.profile(stims[i], prof.target, prof.accuracy[i], prof.gradient[i]);
        prof.gradient[i] /= prof.sigma;
        std::cout << prof.gradient[i] << "\t" << prof.accuracy[i] << std::endl;
        emit progress(i, total);
    }

    m_profiles.push_back(prof);
    emit done();

    // Save
    QFile file(session.log(this, action, "Descriptive metadata goes here..."));
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) )
        os << m_profiles.back();
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

SamplingProfiler::Profile::Profile(WaveSource src) :
    src(src),
    target(src.archive() ? src.archive()->param : 0),
    sigma(src.archive() ? src.session->project.model().adjustableParams[target].adjustedSigma : 1e-5),
    samplingInterval(src.session->runData().simCycles),
    uniform(src.session->project.model().adjustableParams.size()),
    value1(uniform.size()),
    value2(uniform.size()),
    gradient(src.stimulations().size()),
    accuracy(gradient.size())
{}
