#include "errorprofiler.h"
#include "supportcode.h"
#include "util.h"
#include "session.h"


const QString ErrorProfiler::action = QString("generate");
const quint32 ErrorProfiler::magic = 0x2be4e5cb;
const quint32 ErrorProfiler::version = 103;

ErrorProfiler::ErrorProfiler(Session &session) :
    SessionWorker(session),
    lib(session.project.experiment()),
    daq(lib.createSimulator(0, session, false)),
    aborted(false)
{
    connect(this, SIGNAL(doAbort()), this, SLOT(clearAbort()));
}

ErrorProfiler::~ErrorProfiler()
{
    session.project.experiment().destroySimulator(daq);
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

    auto iter = ep.errors.begin();
    int i = 0;
    for ( Stimulation const& stim : ep.stimulations() ) {
        if ( aborted )
            return;
        if ( stim.duration > 0 ) {
            ep.generate(stim, *iter);
        } // else, *iter is an empty vector, as befits an empty stimulation
        iter++;
        emit progress(++i, ep.stimulations().size());
    }

    ep.process_stats();

    m_profiles.push_back(std::move(ep));
    emit done();

    // Save
    QFile file(session.log(this, action, "Descriptive metadata goes here"));
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) )
        os << m_profiles.back();
}


void ErrorProfiler::settle(scalar baseV, scalar settleDuration)
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
    daq->VC = true;
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
    daq->VC = true;
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
