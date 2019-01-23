#include "errorprofiler.h"
#include "supportcode.h"
#include "util.h"
#include "session.h"

const QString ErrorProfiler::action = QString("generate");
const quint32 ErrorProfiler::magic = 0x2be4e5cb;
const quint32 ErrorProfiler::version = 103;

ErrorProfiler::ErrorProfiler(Session &session) :
    SessionWorker(session),
    lib(session.project.universal()),
    daq(lib.createSimulator(0, session, session.getSettings(), false))
{
}

ErrorProfiler::~ErrorProfiler()
{
    lib.destroySimulator(daq);
}

Result *ErrorProfiler::load(const QString &act, const QString &, QFile &results, Result r)
{
    if ( act != action )
        throw std::runtime_error(std::string("Unknown action: ") + act.toStdString());
    QDataStream is;
    quint32 ver = openLoadStream(results, is, magic);
    if ( ver < 100 || ver > version )
        throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

    ErrorProfile *prof;
    if ( r.dryrun )
        prof = new ErrorProfile(session, r);
    else {
        m_profiles.push_back(ErrorProfile(session, r));
        prof =& m_profiles.back();
    }
    prof->version = ver;
    is >> (*prof);

    return prof;
}

void ErrorProfiler::queueProfile(ErrorProfile &&p)
{
    session.queue(actorName(), action, p.source().prettyName(), new ErrorProfile(std::move(p)));

}

bool ErrorProfiler::execute(QString action, QString, Result *result, QFile &file)
{
    clearAbort();
    if ( action != this->action )
        return false;

    ErrorProfile ep = *static_cast<ErrorProfile*>(result);
    try {
        ep.errors.resize(ep.stimulations().size());
        for ( auto &err : ep.errors )
            err.resize(ep.numPermutations());
    } catch (std::bad_alloc) {
        delete result;
        return false;
    }

    if ( result->dryrun ) {
        QDataStream os;
        if ( openSaveStream(file, os, magic, version) )
            os << ep;
        delete result;
        return true;
    }

    auto iter = ep.errors.begin();
    for ( size_t i = 0; i < ep.stimulations().size(); i++ ) {
        if ( isAborted() ) {
            emit didAbort();
            return false;
        }
        if ( ep.stimulations()[i].duration > 0 ) {
            ep.generate(ep.stimulations()[i], ep.observations()[i], *iter);
        } // else, *iter is an empty vector, as befits an empty stimulation
        iter++;
        emit progress(i+1, ep.stimulations().size());
    }

    ep.process_stats();

    m_profiles.push_back(std::move(ep));
    emit done();

    // Save
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) )
        os << m_profiles.back();

    delete result;
    return true;
}


void ErrorProfiler::stimulate(const iStimulation &stim, const iObservations &obs)
{
    const RunData &rd = session.runData();

    // Set up lib
    lib.setSingularRund();
    lib.simCycles = rd.simCycles;
    lib.integrator = rd.integrator;
    lib.setRundata(0, rd);

    lib.setSingularStim();
    lib.assignment = lib.assignment_base
            | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_TARGET
            | ASSIGNMENT_SUMMARY_SQUARED | ASSIGNMENT_SUMMARY_AVERAGE;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_PATTERNCLAMP | ASSIGNMENT_PC_REPORT_PIN;
    lib.stim[0] = stim;
    lib.obs[0] = obs;

    lib.setSingularTarget();
    lib.resizeTarget(1, stim.duration);
    lib.targetOffset[0] = 0;
    lib.push();

    // Set up DAQ
    daq->reset();
    daq->run(Stimulation(stim, rd.dt), rd.settleDuration);

    // Settle DAQ
    if ( rd.VC ) {
        for ( int iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ )
            daq->next();
    } else if ( rd.settleDuration > 0 ) {
        for ( int iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ ) {
            daq->next();
            lib.target[iT] = daq->voltage;
        }
        lib.assignment |= ASSIGNMENT_SETTLE_ONLY;
        lib.pushTarget();
        lib.run();
        lib.assignment &= ~ASSIGNMENT_SETTLE_ONLY;
        lib.iSettleDuration[0] = 0;
        lib.push(lib.iSettleDuration);
    }

    // Run DAQ
    for ( int iT = 0; iT < stim.duration; iT++ ) {
        daq->next();
        lib.target[iT] = rd.VC ? daq->current : daq->voltage;
    }

    // Run lib against target
    lib.pushTarget();
    lib.run();
}
