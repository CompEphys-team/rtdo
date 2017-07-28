#ifndef SESSION_H
#define SESSION_H

#include <QObject>
#include <QThread>
#include <QDir>
#include <QMutex>
#include "sessionlog.h"
#include "project.h"
#include "wavegen.h"
#include "errorprofiler.h"
#include "wavesetcreator.h"
#include "gafitter.h"
#include "samplingprofiler.h"

Q_DECLARE_METATYPE(RunData)
Q_DECLARE_METATYPE(WavegenData)
Q_DECLARE_METATYPE(StimulationData)
Q_DECLARE_METATYPE(GAFitterSettings)
Q_DECLARE_METATYPE(DAQData)

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
    WavesetCreator &wavesets();
    GAFitter &gaFitter();
    SamplingProfiler &samplingProfiler();

    /**
     * Note, DAQ instances are not SessionWorkers.
     * Caution: The DAQ object is deleted whenever a DAQData change becomes effective, and must therefore be reacquired
     * for every new task.
     */
    DAQ *daq();

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
    const GAFitterSettings &gaFitterSettings() const { return gafs; }
    const DAQData &daqData() const { return daqd; }

    inline QString name() const { return dir.dirName(); }

public slots:
    /// Set runtime data
    void setRunData(RunData d);
    void setWavegenData(WavegenData d);
    void setStimulationData(StimulationData d);
    void setGAFitterSettings(GAFitterSettings d);
    void setDAQData(DAQData d);

public:
    Project &project;

protected:
    QThread thread;

    RunData rund;
    WavegenData searchd;
    StimulationData stimd;
    GAFitterSettings gafs;
    DAQData daqd;

    bool dirtyRund, dirtySearchd, dirtyStimd, dirtyGafs, dirtyDaqd;
    std::vector<std::unique_ptr<AP>> runAP, searchAP, stimAP, gafAP, daqAP;

    std::unique_ptr<Wavegen> m_wavegen;
    std::unique_ptr<ErrorProfiler> m_profiler;
    std::unique_ptr<WavesetCreator> m_wavesets;
    std::unique_ptr<GAFitter> m_gafitter;
    std::unique_ptr<SamplingProfiler> m_sprofiler;

    class DAQDeleter
    {
    public:
        DAQDeleter(Session& s) : s(s) {}
        void operator()(DAQ* daq);
        Session &s;
    };
    std::unique_ptr<DAQ, DAQDeleter> m_daq;

    QDir dir;

    SessionLog m_log;

    QMutex log_mutex;

    void addAPs();
    void load();
    void readConfig(const QString &filename);

    static QString results(int idx, const QString &actor, const QString &action);

signals:
    void actionLogged(QString actorName, QString action, QString args, int idx);

    void sanitiseWavegenData(WavegenData *d);

    void redirectRunData(RunData d, QPrivateSignal);
    void redirectWavegenData(WavegenData d, QPrivateSignal);
    void redirectStimulationData(StimulationData d, QPrivateSignal);
    void redirectGAFitterSettings(GAFitterSettings d, QPrivateSignal);
    void redirectDAQData(DAQData d, QPrivateSignal);

    void runDataChanged();
    void wavegenDataChanged();
    void stimulationDataChanged();
    void GAFitterSettingsChanged();
    void DAQDataChanged();
};

#endif // SESSION_H
