#include "session.h"
#include <QDateTime>
#include <QMutexLocker>

Session::Session(Project &p, const QString &sessiondir) :
    RNG(),
    project(p),
    dirtyRund(true),
    dirtySearchd(true),
    dirtyStimd(true),
    dirtyGafs(true),
    dirtyDaqd(true)
{
    static bool registered = false;
    if ( !registered ) {
        qRegisterMetaType<RunData>();
        qRegisterMetaType<WavegenData>();
        qRegisterMetaType<StimulationData>();
        qRegisterMetaType<GAFitterSettings>();
        qRegisterMetaType<DAQData>();

        qRegisterMetaType<WaveSource>();

        registered = true;
    }

    addAPs();

    // Connect redirection signals to ensure set* is always called on this->thread
    connect(this, SIGNAL(redirectRunData(RunData)), this, SLOT(setRunData(RunData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectWavegenData(WavegenData)), this, SLOT(setWavegenData(WavegenData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectStimulationData(StimulationData)), this, SLOT(setStimulationData(StimulationData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectGAFitterSettings(GAFitterSettings)), this, SLOT(setGAFitterSettings(GAFitterSettings)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectDAQData(DAQData)), this, SLOT(setDAQData(DAQData)), Qt::BlockingQueuedConnection);

    if ( sessiondir.isEmpty() ) {
        dir = QDir(project.dir());
        QString newsessiondir = QDateTime::currentDateTime().toString("yyyy.MM.dd-hh.mm.ss");
        dir.mkpath(QString("sessions/%1").arg(newsessiondir));
        dir.cd("sessions");
        dir.cd(newsessiondir);
    } else {
        if ( QDir::isRelativePath(sessiondir) ) {
            dir = QDir(project.dir());
            dir.cd(sessiondir);
        } else {
            dir = QDir(sessiondir);
        }
    }

    moveToThread(&thread);
    thread.start();

    daqd = project.daqData(); // Load project defaults

    m_log.setLogFile(dir.filePath("session.log"));
    load(); // Load state from m_log

    project.wavegen().setRunData(rund);
    project.experiment().setRunData(rund);
    project.profiler().setRunData(rund);
}

void Session::crossloadConfig(const QString &crossSessionDir)
{
    SessionLog crosslog;
    QDir crossdir;
    if ( QDir::isRelativePath(crossSessionDir) ) {
        crossdir = QDir(project.dir());
        crossdir.cd(crossSessionDir);
    } else {
        crossdir = QDir(crossSessionDir);
    }
    crosslog.setLogFile(crossdir.filePath("session.log"));

    for ( int row = 0; row < crosslog.rowCount(); row++ ) {
        SessionLog::Entry entry = crosslog.entry(row);
        QString filename = results(row, entry.actor, entry.action);
        QFile file(crossdir.filePath(filename));
        try {
            if ( entry.actor == "Config" )
                readConfig(file.fileName(), true);
        } catch (std::runtime_error err) {
            std::cerr << "An action could not be loaded (" << crosslog.data(crosslog.index(row, 0), Qt::UserRole).toString()
                      << ", " << filename << ") : "
                      << err.what() << std::endl;
        }
    }
}

void Session::addAPs()
{
    addAP(runAP, "S.Run.accessResistance", this, &Session::rund, &RunData::accessResistance);
    addAP(runAP, "S.Run.clampGain", this, &Session::rund, &RunData::clampGain);
    addAP(runAP, "S.Run.simCycles", this, &Session::rund, &RunData::simCycles);
    addAP(runAP, "S.Run.settleDuration", this, &Session::rund, &RunData::settleDuration);

    addAP(searchAP, "S.Wavegen.numSigmaAdjustWaveforms", this, &Session::searchd, &WavegenData::numSigmaAdjustWaveforms);
    addAP(searchAP, "S.Wavegen.nInitialWaves", this, &Session::searchd, &WavegenData::nInitialWaves);
    addAP(searchAP, "S.Wavegen.nGroupsPerWave", this, &Session::searchd, &WavegenData::nGroupsPerWave);
    addAP(searchAP, "S.Wavegen.useBaseParameters", this, &Session::searchd, &WavegenData::useBaseParameters);
    addAP(searchAP, "S.Wavegen.nWavesPerEpoch", this, &Session::searchd, &WavegenData::nWavesPerEpoch);
    addAP(searchAP, "S.Wavegen.rerandomiseParameters", this, &Session::searchd, &WavegenData::rerandomiseParameters);
    addAP(searchAP, "S.Wavegen.precisionIncreaseEpochs[#]", this, &Session::searchd, &WavegenData::precisionIncreaseEpochs);
    addAP(searchAP, "S.Wavegen.maxIterations", this, &Session::searchd, &WavegenData::maxIterations);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].func", this, &Session::searchd, &WavegenData::mapeDimensions, &MAPEDimension::func);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].min", this, &Session::searchd, &WavegenData::mapeDimensions, &MAPEDimension::min);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].max", this, &Session::searchd, &WavegenData::mapeDimensions, &MAPEDimension::max);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].resolution", this, &Session::searchd, &WavegenData::mapeDimensions, &MAPEDimension::resolution);

    addAP(stimAP, "S.Stimulation.baseV", this, &Session::stimd, &StimulationData::baseV);
    addAP(stimAP, "S.Stimulation.duration", this, &Session::stimd, &StimulationData::duration);
    addAP(stimAP, "S.Stimulation.minSteps", this, &Session::stimd, &StimulationData::minSteps);
    addAP(stimAP, "S.Stimulation.maxSteps", this, &Session::stimd, &StimulationData::maxSteps);
    addAP(stimAP, "S.Stimulation.minVoltage", this, &Session::stimd, &StimulationData::minVoltage);
    addAP(stimAP, "S.Stimulation.maxVoltage", this, &Session::stimd, &StimulationData::maxVoltage);
    addAP(stimAP, "S.Stimulation.minStepLength", this, &Session::stimd, &StimulationData::minStepLength);
    addAP(stimAP, "S.Stimulation.muta.lCrossover", this, &Session::stimd, &StimulationData::muta, &MutationData::lCrossover);
    addAP(stimAP, "S.Stimulation.muta.lLevel", this, &Session::stimd, &StimulationData::muta, &MutationData::lLevel);
    addAP(stimAP, "S.Stimulation.muta.lNumber", this, &Session::stimd, &StimulationData::muta, &MutationData::lNumber);
    addAP(stimAP, "S.Stimulation.muta.lSwap", this, &Session::stimd, &StimulationData::muta, &MutationData::lSwap);
    addAP(stimAP, "S.Stimulation.muta.lTime", this, &Session::stimd, &StimulationData::muta, &MutationData::lTime);
    addAP(stimAP, "S.Stimulation.muta.lType", this, &Session::stimd, &StimulationData::muta, &MutationData::lType);
    addAP(stimAP, "S.Stimulation.muta.n", this, &Session::stimd, &StimulationData::muta, &MutationData::n);
    addAP(stimAP, "S.Stimulation.muta.sdLevel", this, &Session::stimd, &StimulationData::muta, &MutationData::sdLevel);
    addAP(stimAP, "S.Stimulation.muta.sdTime", this, &Session::stimd, &StimulationData::muta, &MutationData::sdTime);
    addAP(stimAP, "S.Stimulation.muta.std", this, &Session::stimd, &StimulationData::muta, &MutationData::std);

    addAP(gafAP, "S.GAFitter.maxEpochs", this, &Session::gafs, &GAFitterSettings::maxEpochs);
    addAP(gafAP, "S.GAFitter.randomOrder", this, &Session::gafs, &GAFitterSettings::randomOrder);
    addAP(gafAP, "S.GAFitter.orderBiasDecay", this, &Session::gafs, &GAFitterSettings::orderBiasDecay);
    addAP(gafAP, "S.GAFitter.orderBiasStartEpoch", this, &Session::gafs, &GAFitterSettings::orderBiasStartEpoch);
    addAP(gafAP, "S.GAFitter.nElite", this, &Session::gafs, &GAFitterSettings::nElite);
    addAP(gafAP, "S.GAFitter.nReinit", this, &Session::gafs, &GAFitterSettings::nReinit);
    addAP(gafAP, "S.GAFitter.crossover", this, &Session::gafs, &GAFitterSettings::crossover);
    addAP(gafAP, "S.GAFitter.decaySigma", this, &Session::gafs, &GAFitterSettings::decaySigma);
    addAP(gafAP, "S.GAFitter.sigmaInitial", this, &Session::gafs, &GAFitterSettings::sigmaInitial);
    addAP(gafAP, "S.GAFitter.sigmaHalflife", this, &Session::gafs, &GAFitterSettings::sigmaHalflife);

    Project::addDaqAPs(daqAP, &daqd);

    // Defaults
    searchd.mapeDimensions = {
        {MAPEDimension::Func::BestBubbleDuration,   0, 0, 128},
        {MAPEDimension::Func::BestBubbleTime,       0, 0,  32},
        {MAPEDimension::Func::VoltageDeviation,     0, 0,  32}
    };
    for ( MAPEDimension &m : searchd.mapeDimensions )
        m.setDefaultMinMax(stimd);
    searchd.precisionIncreaseEpochs = { 100 };
}

Wavegen &Session::wavegen()
{
    if ( !m_wavegen ) {
        m_wavegen.reset(new Wavegen(*this));
    }
    return *m_wavegen;
}

ErrorProfiler &Session::profiler()
{
    if ( !m_profiler ) {
        m_profiler.reset(new ErrorProfiler(*this));
    }
    return *m_profiler;
}

WavesetCreator &Session::wavesets()
{
    if ( !m_wavesets ) {
        m_wavesets.reset(new WavesetCreator(*this));
    }
    return *m_wavesets;
}

GAFitter &Session::gaFitter()
{
    if ( !m_gafitter ) {
        m_gafitter.reset(new GAFitter(*this));
    }
    return *m_gafitter;
}

SamplingProfiler &Session::samplingProfiler()
{
    if ( !m_sprofiler) {
        m_sprofiler.reset(new SamplingProfiler(*this));
    }
    return *m_sprofiler;
}

void Session::quit()
{
    if ( m_wavegen )
        m_wavegen->abort();
    if ( m_profiler )
        m_profiler->abort();
    if ( m_wavesets)
        m_wavesets->abort();
    if ( m_sprofiler )
        m_sprofiler->abort();
    thread.quit();
    thread.wait();
}

QString Session::log(const SessionWorker *actor, const QString &action, const QString &args)
{
    int idx;
    QMutexLocker lock(&log_mutex);

    if ( dirtyRund || dirtySearchd || dirtyStimd || dirtyGafs || dirtyDaqd ) {
        QString actorName = "Config";
        QString cfg = "cfg";
        idx = m_log.put(actorName, cfg, "");
        std::ofstream os(dir.filePath(results(idx, actorName, cfg)).toStdString());
        if ( dirtyRund ) {
            for ( auto const& p : runAP )
                p->write(os);
            hist_rund.push_back(std::make_pair(idx, rund));
        }
        if ( dirtySearchd ) {
            for ( auto const& p : searchAP )
                p->write(os);
            hist_searchd.push_back(std::make_pair(idx, searchd));
        }
        if ( dirtyStimd ) {
            for ( auto const& p : stimAP )
                p->write(os);
            hist_stimd.push_back(std::make_pair(idx, stimd));
        }
        if ( dirtyGafs ) {
            for ( auto const& p : gafAP )
                p->write(os);
            hist_gafs.push_back(std::make_pair(idx, gafs));
        }
        if ( dirtyDaqd ) {
            for ( auto const& p : daqAP )
                p->write(os);
            hist_daqd.push_back(std::make_pair(idx, daqd));
        }
        dirtyRund = dirtySearchd = dirtyStimd = dirtyGafs = dirtyDaqd = false;
    }

    idx = m_log.put(actor->actorName(), action, args);
    emit actionLogged(actor->actorName(), action, args, idx);
    return dir.filePath(results(idx, actor->actorName(), action));
}

template <typename data>
data getHistoricData(int index, std::vector<std::pair<int, data>> history)
{
    auto it = history.begin() + 1;
    while ( it != history.end() && it->first < index )
        ++it;
    return (it-1)->second;
}
RunData Session::runData(int i) const { return i < 0 ? rund : getHistoricData(i, hist_rund); }
WavegenData Session::wavegenData(int i) const { return i < 0 ? searchd : getHistoricData(i, hist_searchd); }
StimulationData Session::stimulationData(int i) const { return i < 0 ? stimd : getHistoricData(i, hist_stimd); }
GAFitterSettings Session::gaFitterSettings(int i) const { return i < 0 ? gafs : getHistoricData(i, hist_gafs); }
DAQData Session::daqData(int i) const { return i < 0 ? daqd : getHistoricData(i, hist_daqd); }

void Session::appropriate(QObject *worker)
{
    worker->moveToThread(&thread);
}

void Session::load()
{
    for ( int row = 0; row < m_log.rowCount(); row++ ) {
        SessionLog::Entry entry = m_log.entry(row);
        QString filename = results(row, entry.actor, entry.action);
        QFile file(dir.filePath(filename));
        try {
            if ( entry.actor == wavegen().actorName() )
                wavegen().load(entry.action, entry.args, file);
            else if ( entry.actor == profiler().actorName() )
                profiler().load(entry.action, entry.args, file);
            else if ( entry.actor == "Config" ) {
                readConfig(file.fileName());
            } else if ( entry.actor == wavesets().actorName() ) {
                wavesets().load(entry.action, entry.args, file);
            } else if ( entry.actor == gaFitter().actorName() ) {
                gaFitter().load(entry.action, entry.args, file);
            } else if ( entry.actor == samplingProfiler().actorName() ) {
                samplingProfiler().load(entry.action, entry.args, file);
            } else {
                throw std::runtime_error(std::string("Unknown actor: ") + entry.actor.toStdString());
            }
        } catch (std::runtime_error err) {
            std::cerr << "An action could not be loaded (" << m_log.data(m_log.index(row, 0), Qt::UserRole).toString()
                      << ", " << filename << ") : "
                      << err.what() << std::endl;
        }
    }
}

void Session::readConfig(const QString &filename, bool raiseDirtyFlags)
{
    std::ifstream is(filename.toStdString());
    QString name;
    AP *it;
    bool hasRun(false), hasSearch(false), hasStim(false), hasGafs(false), hasDaq(false);
    is >> name;
    while ( is.good() ) {
        if ( (it = AP::find(name, &runAP)) ) {
            if ( !hasRun ) {
                rund = RunData();
                hasRun = true;
            }
        } else if ( (it = AP::find(name, &searchAP)) ) {
            if ( !hasSearch ) {
                searchd = WavegenData();
                hasSearch = true;
            }
        } else if ( (it = AP::find(name, &stimAP)) ) {
            if ( !hasStim ) {
                stimd = StimulationData();
                hasStim = true;
            }
        } else if ( (it = AP::find(name, &gafAP)) ) {
            if ( !hasGafs ) {
                gafs = GAFitterSettings();
                hasGafs = true;
            }
        } else if ( (it = AP::find(name, &daqAP)) ) {
            if ( !hasDaq ) {
                daqd = DAQData();
                hasDaq = true;
            }
        }
        if ( it )
            it->readNow(name, is);
        is >> name;
    }
    if ( hasRun ) {
        dirtyRund = raiseDirtyFlags;
        project.wavegen().setRunData(rund);
        project.experiment().setRunData(rund);
        project.profiler().setRunData(rund);
        emit runDataChanged();
    }
    if ( hasSearch ) {
        dirtySearchd = raiseDirtyFlags;
        emit wavegenDataChanged();
    }
    if ( hasStim ) {
        dirtyStimd = raiseDirtyFlags;
        emit stimulationDataChanged();
    }
    if ( hasGafs ) {
        dirtyGafs = raiseDirtyFlags;
        emit GAFitterSettingsChanged();
    }
    if ( hasDaq ) {
        dirtyDaqd = raiseDirtyFlags;
        emit DAQDataChanged();
    }
}

QString Session::results(int idx, const QString &actor, const QString &action)
{
    return QString("%1.%2.%3")
            .arg(idx, 4, 10, QChar('0')) // 4 digits, pad with zeroes
            .arg(actor, action);
}

void Session::setRunData(RunData d)
{
    if ( QThread::currentThread() == &thread ) {
        project.wavegen().setRunData(d);
        project.experiment().setRunData(d);
        project.profiler().setRunData(d);
        rund = d;
        dirtyRund = true;
        emit runDataChanged();
    } else {
        redirectRunData(d, QPrivateSignal());
    }
}

void Session::setWavegenData(WavegenData d)
{
    if ( QThread::currentThread() == &thread ) {
        sanitiseWavegenData(&d);
        searchd = d;
        dirtySearchd = true;
        emit wavegenDataChanged();
    } else {
        redirectWavegenData(d, QPrivateSignal());
    }
}

void Session::setStimulationData(StimulationData d)
{
    if ( QThread::currentThread() == &thread ) {
        stimd = d;
        dirtyStimd = true;
        emit stimulationDataChanged();
    } else {
        redirectStimulationData(d, QPrivateSignal());
    }
}

void Session::setGAFitterSettings(GAFitterSettings d)
{
    if ( QThread::currentThread() == &thread ) {
        gafs = d;
        dirtyGafs = true;
        emit GAFitterSettingsChanged();
    } else {
        redirectGAFitterSettings(d, QPrivateSignal());
    }
}

void Session::setDAQData(DAQData d)
{
    if ( QThread::currentThread() == &thread ) {
        daqd = d;
        dirtyDaqd = true;
        emit DAQDataChanged();
    } else {
        redirectDAQData(d, QPrivateSignal());
    }
}
