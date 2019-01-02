#include "wavegen.h"
#include "session.h"

Wavegen::Wavegen(Session &session) :
    SessionWorker(session),
    searchd(session.wavegenData()),
    stimd(session.stimulationData()),
    lib(session.project.wavegen()),
    ulib(session.project.universal())
{
}

Result *Wavegen::load(const QString &action, const QString &args, QFile &results, Result r)
{
    if ( action == cluster_action )
        return cluster_load(results, args, r);
    else if ( action == bubble_action )
        return bubble_load(results, args, r);
    else
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
}

bool Wavegen::execute(QString action, QString, Result *res, QFile &file)
{
    clearAbort();
    istimd = iStimData(stimd, session.runData().dt);
    if ( action == cluster_action )
        return cluster_exec(file, res);
    else if ( action == bubble_action )
        return bubble_exec(file, res);
    else
        return false;
}
