#include "samplingprofiler.h"
#include "session.h"

const QString SamplingProfiler::action = QString("generate");
const quint32 SamplingProfiler::magic = 0x678b5c54;
const quint32 SamplingProfiler::version = 100;

QDataStream &operator<<(QDataStream &os, const SamplingProfiler::Profile &);
QDataStream &operator>>(QDataStream &is, SamplingProfiler::Profile &);

SamplingProfiler::SamplingProfiler(Session &session) :
    SessionWorker(session),
    lib(session.project.universal())
{
    qRegisterMetaType<Profile>();
}

Result *SamplingProfiler::load(const QString &act, const QString &, QFile &results, Result r)
{
    if ( act == action ) {
        QDataStream is;
        quint32 ver = openLoadStream(results, is, magic);
        if ( ver < 100 || ver > version )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        Profile *prof;
        if ( r.dryrun )
            prof = new Profile(r);
        else {
            m_profiles.push_back(Profile(r));
            prof =& m_profiles.back();
        }
        prof->src.session = &session;
        is >> *prof;
        return prof;
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
    if ( res->dryrun ) {
        QDataStream os;
        if ( openSaveStream(file, os, magic, version) )
            os << prof;
        delete res;
        return true;
    }

    const RunData &rd = session.runData();
    std::vector<MAPElite> elites = prof.src.elites();

    int maxObsDuration = 0;
    for ( const MAPElite &el : elites ) {
        int obsDuration = el.obs.duration();
        maxObsDuration = std::max(maxObsDuration, obsDuration);
    }

    // Populate
    for ( size_t param = 0; param < lib.adjustableParams.size(); param++ ) {
        AdjustableParam &p = lib.adjustableParams[param];
        if ( param >= lib.model.nNormalAdjustableParams ) // choice params
            for ( size_t i = 0; i < lib.NMODELS; i++ )
                p[i] = session.RNG.pick({-0.5, 0.5});
        else if ( prof.uniform[param] )
            for ( size_t i = 0; i < lib.NMODELS; i++ )
                p[i] = session.RNG.uniform<scalar>(prof.value1[param], prof.value2[param]);
        else
            for ( size_t i = 0; i < lib.NMODELS; i++ )
                p[i] = session.RNG.variate<scalar, std::normal_distribution>(prof.value1[param], prof.value2[param]);
    }

    // Set up library
    lib.setSingularRund();
    lib.simCycles = rd.simCycles;
    lib.integrator = rd.integrator;
    lib.noiseExp = rd.noiseExp();
    lib.noiseAmplitude = rd.noiseAmplitude();
    lib.setRundata(0, rd);

    lib.setSingularStim();
    lib.stim[0].baseV = NAN;
    lib.obs[0] = {{}, {}};

    lib.resizeOutput(maxObsDuration);

    lib.push();

    unsigned int assignment = lib.assignment_base
            | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPACT | ASSIGNMENT_TIMESERIES_COMPARE_NONE;

    if ( rd.noisy ) {
        assignment |= (rd.noisyChannels ? ASSIGNMENT_NOISY_CHANNELS : ASSIGNMENT_NOISY_OBSERVATION);
        lib.generate_random_samples(maxObsDuration * 2*rd.simCycles * lib.NMODELS,
                                    0, rd.noisyChannels ? 1 : rd.noiseAmplitude(),
                                    session.RNG.uniform(0ull, std::numeric_limits<unsigned long long>::max()));
    }

    prof.rho_weighted.resize(elites.size());
    prof.rho_unweighted.resize(elites.size());
    prof.rho_target_only.resize(elites.size());
    prof.grad_weighted.resize(elites.size());
    prof.grad_unweighted.resize(elites.size());
    prof.grad_target_only.resize(elites.size());

    std::vector<double> invariants;

    // Run
    for ( size_t i = 0; i < elites.size(); i++ ) {
        if ( isAborted() ) {
            delete res;
            return false;
        }

        lib.stim[0] = *elites[i].wave;
        lib.obs[0] = elites[i].obs;
        lib.push(lib.stim);
        lib.push(lib.obs);

        // Settle, if necessary - lib maintains settled state
        if ( i == 0 || elites[i-1].wave->baseV != elites[i].wave->baseV ) {
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
        lib.profile(elites[i].obs.duration(), prof.target, elites[i].deviations,
                    prof.rho_weighted[i], prof.rho_unweighted[i], prof.rho_target_only[i],
                    prof.grad_weighted[i], prof.grad_unweighted[i], prof.grad_target_only[i],
                    invariants);
        emit progress(i, elites.size());
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
    os << p.src << quint32(p.target);
    os << p.uniform << p.value1 << p.value2;
    os << p.rho_weighted << p.rho_unweighted << p.rho_target_only;
    os << p.grad_weighted << p.grad_unweighted << p.grad_target_only;
    return os;
}

QDataStream &operator>>(QDataStream &is, SamplingProfiler::Profile &p)
{
    quint32 target;
    is >> p.src >> target;
    p.target = target;
    is >> p.uniform >> p.value1 >> p.value2;
    is >> p.rho_weighted >> p.rho_unweighted >> p.rho_target_only;
    is >> p.grad_weighted >> p.grad_unweighted >> p.grad_target_only;
    return is;
}

SamplingProfiler::Profile::Profile(WaveSource src, size_t target, Result r) :
    Result(r),
    src(src),
    target(target),
    uniform(src.session->project.model().adjustableParams.size()),
    value1(uniform.size()),
    value2(uniform.size())
{}
