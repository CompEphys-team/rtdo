#ifndef SESSION_H
#define SESSION_H

#include <QObject>
#include <QThread>
#include "project.h"
#include "wavegen.h"
#include "experiment.h"
#include "errorprofiler.h"

Q_DECLARE_METATYPE(RunData)
Q_DECLARE_METATYPE(WavegenData)
Q_DECLARE_METATYPE(StimulationData)
Q_DECLARE_METATYPE(ExperimentData)

class Session : public QObject
{
    Q_OBJECT

public:
    Session(Project &p);

    Wavegen &wavegen();
    ErrorProfiler &profiler();

    void quit();

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

    std::unique_ptr<Wavegen> m_wavegen;
    std::unique_ptr<ErrorProfiler> m_profiler;

signals:
    void redirectRunData(RunData d, QPrivateSignal);
    void redirectWavegenData(WavegenData d, QPrivateSignal);
    void redirectStimulationData(StimulationData d, QPrivateSignal);
    void redirectExperimentData(ExperimentData d, QPrivateSignal);
};

#endif // SESSION_H
