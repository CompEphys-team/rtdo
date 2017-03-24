#ifndef SESSION_H
#define SESSION_H

#include <QObject>
#include <QThread>
#include <QDir>
#include "sessionlog.h"
#include "project.h"
#include "wavegen.h"
#include "experiment.h"
#include "errorprofiler.h"
#include "wavegenselector.h"

Q_DECLARE_METATYPE(RunData)
Q_DECLARE_METATYPE(WavegenData)
Q_DECLARE_METATYPE(StimulationData)
Q_DECLARE_METATYPE(ExperimentData)

class Session : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Session constructs a new session or loads an existing one.
     * @param p is the project the session will belong to
     * @param sessiondir is (a) empty, in which case a completely new session will be created in a timestamped directory
     * within the project directory; (b) an existing directory containing a session.log, which will be loaded; or
     * (c) an existing directory without a session.log, in which case a new session will be created in the directory named.
     */
    Session(Project &p, const QString &sessiondir = QString());

    Wavegen &wavegen();
    ErrorProfiler &profiler();
    WavegenSelector &wavegenselector();

    void quit();

    /**
     * @brief log enters an action into the session log. This should be called by the subordinate objects
     * (wavegen, profiler etc) upon action completion.
     * @param actor is a pointer to the object that completed the action (usually <caller>.this)
     * @param action is a single-word string describing the action, e.g. "search". action must not be
     * empty and will be read back to the actor on session load.
     * @param args is an arbitrary string providing closer specification to the action, e.g. arguments
     * the action received on its execution. Must not contain any newline characters.
     * @return a string indicating the default result file path, e.g. "/path/to/session/0000.Wavegen.search"
     */
    QString log(const SessionWorker *actor, const QString &action, const QString &args = QString());

    inline SessionLog *getLog() { return &m_log; }

    const RunData &runData() const { return rund; }
    const WavegenData &wavegenData() const { return searchd; }
    const StimulationData &stimulationData() const { return stimd; }
    const ExperimentData &experimentData() const { return expd; }

public slots:
    /// Set runtime data
    void setRunData(RunData d);
    void setWavegenData(WavegenData d);
    void setStimulationData(StimulationData d);
    void setExperimentData(ExperimentData d);

public:
    Project &project;

protected:
    QThread thread;

    RunData rund;
    WavegenData searchd;
    StimulationData stimd;
    ExperimentData expd;

    bool dirtyRund, dirtySearchd, dirtyStimd, dirtyExpd;
    std::vector<std::unique_ptr<AP>> runAP, searchAP, stimAP, expAP;

    std::unique_ptr<Wavegen> m_wavegen;
    std::unique_ptr<ErrorProfiler> m_profiler;
    std::unique_ptr<WavegenSelector> m_wavegenselector;

    QDir dir;

    SessionLog m_log;

    void addAPs();
    void load();
    void readConfig(const QString &filename);

    static QString results(int idx, const QString &actor, const QString &action);

signals:
    void redirectRunData(RunData d, QPrivateSignal);
    void redirectWavegenData(WavegenData d, QPrivateSignal);
    void redirectStimulationData(StimulationData d, QPrivateSignal);
    void redirectExperimentData(ExperimentData d, QPrivateSignal);
};

#endif // SESSION_H
