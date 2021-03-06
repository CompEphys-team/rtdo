/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#ifndef SESSION_H
#define SESSION_H

#include <QObject>
#include <QThread>
#include <QDir>
#include <QMutex>
#include <atomic>
#include "sessionlog.h"
#include "project.h"
#include "wavegen.h"
#include "errorprofiler.h"
#include "wavesetcreator.h"
#include "gafitter.h"
#include "samplingprofiler.h"
#include "randutils.hpp"

class Dispatcher : public QObject
{
    Q_OBJECT
public:
    Dispatcher(Session &s) : s(s), running(true), busy(false) {}
    Session &s;
    std::atomic<bool> running, busy;
    bool dryrun = false;
    QDir dir;
    SessionLog::Entry nextEntry;
    QMutex mutex;
public slots:
    void dispatch();
signals:
    void actionComplete(bool success);
    void requestNextEntry();
};

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
    ~Session();

    Wavegen &wavegen();
    ErrorProfiler &profiler();
    WavesetCreator &wavesets();
    GAFitter &gaFitter();
    SamplingProfiler &samplingProfiler();

    void quit();
    void pause();
    void resume();
    void abort();

    void desiccate(const QString &file, const QString &directory);
    void exec_desiccated(const QString &file, bool nogui = true);

    /**
     * @brief queue adds an action to the queue, starting asynchronous execution immediately (unless paused).
     * Typically, this function is called by worker objects, although it can equally be called directly from the GUI.
     * @param actor is the unique actor name (queue/log entry, file name suffix).
     * @param action is the name of the action (queue/log entry, file name body); actors are responsible for making sense of this.
     * @param args is an arbitrary string (no newlines) that may be used for both visual identification or functional purposes
     * @param res is a pointer to a Result object containing e.g. bulkier arguments to the action. SessionLog takes ownership of this
     * pointer, handing it back in a call to Worker::execute() or deleting it when the queue entry is removed.
     */
    void queue(QString actor, QString action, QString args, Result *res);

    inline SessionLog *getLog() { return &m_log; }

    // q___ return the latest queued settings for use in the GUI
    const Settings &qSettings() const { return q_settings; }
    const RunData &qRunData() const { return q_settings.rund; }
    const WavegenData &qWavegenData() const { return q_settings.searchd; }
    const StimulationData &qStimulationData() const { return q_settings.stimd; }
    const GAFitterSettings &qGaFitterSettings() const { return q_settings.gafs; }
    const DAQData &qDaqData() const { return q_settings.daqd; }

    // Settings/___Data return the settings at execution time
    const Settings &getSettings() const { return m_settings; }
    const RunData &runData() const { return m_settings.rund; }
    const WavegenData &wavegenData() const { return m_settings.searchd; }
    const StimulationData &stimulationData() const { return m_settings.stimd; }
    const GAFitterSettings &gaFitterSettings() const { return m_settings.gafs; }
    const DAQData &daqData() const { return m_settings.daqd; }

    // Settings/___Data(resultIndex) return the settings at historic time points
    Settings getSettings(int resultIndex) const;
    RunData runData(int resultIndex) const;
    WavegenData wavegenData(int resultIndex) const;
    StimulationData stimulationData(int resultIndex) const;
    GAFitterSettings gaFitterSettings(int resultIndex) const;
    DAQData daqData(int resultIndex) const;

    inline QString directory() const { return dir.absolutePath(); }
    inline QString name() const { return dir.dirName(); }

    void crossloadConfig(const QString &crossSessionDir);
    void loadConfig(const QString &configFile);

    inline void appropriate(QObject *worker) { worker->moveToThread(&thread); }

    randutils::mt19937_rng RNG;

    CannedDAQ::ChannelAssociation cdaq_assoc;

    QString resultFileName(int idx) const;
    QString resultFileName(int idx, const QString &actor, const QString &action) const;
    inline QString resultFilePath(int idx, const QString &actor, const QString &action) const { return dir.filePath(resultFileName(idx, actor, action)); }
    inline QString resultFilePath(int idx) const { return dir.filePath(resultFileName(idx)); }

public slots:
    /// Set runtime data
    void setRunData(RunData d);
    void setWavegenData(WavegenData d);
    void setStimulationData(StimulationData d);
    void setGAFitterSettings(GAFitterSettings d);
    void setDAQData(DAQData d);

    inline bool busy() { return dispatcher.busy.load(); }

public:
    Project &project;

protected:
    QThread thread;

    Dispatcher dispatcher;
    friend class Dispatcher;

    Settings m_settings, q_settings; // m_settings: execution-time; q_settings: latest queued settings for GUI
    std::vector<std::pair<int, Settings>> hist_settings;
    bool initial = true;

    std::vector<std::unique_ptr<AP>> runAP, searchAP, stimAP, gafAP, daqAP, cdaqAP;

    std::unique_ptr<Wavegen> m_wavegen;
    std::unique_ptr<ErrorProfiler> m_profiler;
    std::unique_ptr<WavesetCreator> m_wavesets;
    std::unique_ptr<GAFitter> m_gafitter;
    std::unique_ptr<SamplingProfiler> m_sprofiler;

    QDir dir;

    SessionLog m_log;

    void addAPs();
    std::vector<SessionLog::Entry> load(bool dryrun = false, SessionLog *log = nullptr, int rowOffset = 0);
    bool readConfig(const QString &filename, bool incremental = false);

    void sanitiseSettings(Settings &s);

protected slots:
    void getNextEntry();
    void onActionComplete(bool success);
    void updateSettings();

signals:
    void actionLogged(QString actorName, QString action, QString args, int idx);
    void doDispatch();
    void dispatchComplete();

    void runDataChanged();
    void wavegenDataChanged();
    void stimulationDataChanged();
    void GAFitterSettingsChanged();
    void DAQDataChanged();
};

#endif // SESSION_H
