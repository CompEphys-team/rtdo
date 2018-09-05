#include "session.h"
#include <QDateTime>
#include <QMutexLocker>

Session::Session(Project &p, const QString &sessiondir) :
    RNG(),
    project(p),
    dispatcher(*this)
{
    addAPs();

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

    dispatcher.moveToThread(&thread);
    thread.start();
    connect(this, SIGNAL(doDispatch()), &dispatcher, SLOT(dispatch()));
    connect(&dispatcher, SIGNAL(actionComplete(bool)), this, SLOT(onActionComplete(bool)), Qt::BlockingQueuedConnection);
    connect(&dispatcher, SIGNAL(requestNextEntry()), this, SLOT(getNextEntry()), Qt::BlockingQueuedConnection);
    connect(&m_log, SIGNAL(queueAltered()), this, SLOT(updateSettings()));

    q_settings.daqd = project.daqData(); // Load project defaults

    m_log.setLogFile(dir.filePath("session.log"));
    load(); // Load state from m_log

    project.wavegen().setRunData(m_settings.rund);
    project.experiment().setRunData(m_settings.rund);
    project.profiler().setRunData(m_settings.rund);
}

Session::~Session()
{
    quit();
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
                readConfig(file.fileName());
        } catch (std::runtime_error err) {
            std::cerr << "An action could not be loaded (" << crosslog.data(crosslog.index(row, 0), Qt::UserRole).toString()
                      << ", " << filename << ") : "
                      << err.what() << std::endl;
        }
    }
    m_log.queue("Config", "cfg", "", new Settings(q_settings));
}

void Session::addAPs()
{
    addAP(runAP, "S.Run.accessResistance", &q_settings, &Settings::rund, &RunData::accessResistance);
    addAP(runAP, "S.Run.clampGain", &q_settings, &Settings::rund, &RunData::clampGain);
    addAP(runAP, "S.Run.simCycles", &q_settings, &Settings::rund, &RunData::simCycles);
    addAP(runAP, "S.Run.settleDuration", &q_settings, &Settings::rund, &RunData::settleDuration);
    addAP(runAP, "S.Run.integrator", &q_settings, &Settings::rund, &RunData::integrator);
    addAP(runAP, "S.Run.dt", &q_settings, &Settings::rund, &RunData::dt);
    addAP(runAP, "S.Run.Imax", &q_settings, &Settings::rund, &RunData::Imax);

    addAP(searchAP, "S.Wavegen.numSigmaAdjustWaveforms", &q_settings, &Settings::searchd, &WavegenData::numSigmaAdjustWaveforms);
    addAP(searchAP, "S.Wavegen.nInitialWaves", &q_settings, &Settings::searchd, &WavegenData::nInitialWaves);
    addAP(searchAP, "S.Wavegen.nGroupsPerWave", &q_settings, &Settings::searchd, &WavegenData::nGroupsPerWave);
    addAP(searchAP, "S.Wavegen.useBaseParameters", &q_settings, &Settings::searchd, &WavegenData::useBaseParameters);
    addAP(searchAP, "S.Wavegen.nWavesPerEpoch", &q_settings, &Settings::searchd, &WavegenData::nWavesPerEpoch);
    addAP(searchAP, "S.Wavegen.rerandomiseParameters", &q_settings, &Settings::searchd, &WavegenData::rerandomiseParameters);
    addAP(searchAP, "S.Wavegen.precisionIncreaseEpochs[#]", &q_settings, &Settings::searchd, &WavegenData::precisionIncreaseEpochs);
    addAP(searchAP, "S.Wavegen.maxIterations", &q_settings, &Settings::searchd, &WavegenData::maxIterations);
    addAP(searchAP, "S.Wavegen.noise_sd", &q_settings, &Settings::searchd, &WavegenData::noise_sd);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].func", &q_settings, &Settings::searchd, &WavegenData::mapeDimensions, &MAPEDimension::func);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].min", &q_settings, &Settings::searchd, &WavegenData::mapeDimensions, &MAPEDimension::min);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].max", &q_settings, &Settings::searchd, &WavegenData::mapeDimensions, &MAPEDimension::max);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].resolution", &q_settings, &Settings::searchd, &WavegenData::mapeDimensions, &MAPEDimension::resolution);

    addAP(stimAP, "S.Stimulation.baseV", &q_settings, &Settings::stimd, &StimulationData::baseV);
    addAP(stimAP, "S.Stimulation.duration", &q_settings, &Settings::stimd, &StimulationData::duration);
    addAP(stimAP, "S.Stimulation.minSteps", &q_settings, &Settings::stimd, &StimulationData::minSteps);
    addAP(stimAP, "S.Stimulation.maxSteps", &q_settings, &Settings::stimd, &StimulationData::maxSteps);
    addAP(stimAP, "S.Stimulation.minVoltage", &q_settings, &Settings::stimd, &StimulationData::minVoltage);
    addAP(stimAP, "S.Stimulation.maxVoltage", &q_settings, &Settings::stimd, &StimulationData::maxVoltage);
    addAP(stimAP, "S.Stimulation.minStepLength", &q_settings, &Settings::stimd, &StimulationData::minStepLength);
    addAP(stimAP, "S.Stimulation.muta.lCrossover", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::lCrossover);
    addAP(stimAP, "S.Stimulation.muta.lLevel", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::lLevel);
    addAP(stimAP, "S.Stimulation.muta.lNumber", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::lNumber);
    addAP(stimAP, "S.Stimulation.muta.lSwap", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::lSwap);
    addAP(stimAP, "S.Stimulation.muta.lTime", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::lTime);
    addAP(stimAP, "S.Stimulation.muta.lType", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::lType);
    addAP(stimAP, "S.Stimulation.muta.n", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::n);
    addAP(stimAP, "S.Stimulation.muta.sdLevel", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::sdLevel);
    addAP(stimAP, "S.Stimulation.muta.sdTime", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::sdTime);
    addAP(stimAP, "S.Stimulation.muta.std", &q_settings, &Settings::stimd, &StimulationData::muta, &MutationData::std);

    addAP(gafAP, "S.GAFitter.maxEpochs", &q_settings, &Settings::gafs, &GAFitterSettings::maxEpochs);
    addAP(gafAP, "S.GAFitter.randomOrder", &q_settings, &Settings::gafs, &GAFitterSettings::randomOrder);
    addAP(gafAP, "S.GAFitter.orderBiasDecay", &q_settings, &Settings::gafs, &GAFitterSettings::orderBiasDecay);
    addAP(gafAP, "S.GAFitter.orderBiasStartEpoch", &q_settings, &Settings::gafs, &GAFitterSettings::orderBiasStartEpoch);
    addAP(gafAP, "S.GAFitter.nElite", &q_settings, &Settings::gafs, &GAFitterSettings::nElite);
    addAP(gafAP, "S.GAFitter.nReinit", &q_settings, &Settings::gafs, &GAFitterSettings::nReinit);
    addAP(gafAP, "S.GAFitter.crossover", &q_settings, &Settings::gafs, &GAFitterSettings::crossover);
    addAP(gafAP, "S.GAFitter.decaySigma", &q_settings, &Settings::gafs, &GAFitterSettings::decaySigma);
    addAP(gafAP, "S.GAFitter.sigmaInitial", &q_settings, &Settings::gafs, &GAFitterSettings::sigmaInitial);
    addAP(gafAP, "S.GAFitter.sigmaHalflife", &q_settings, &Settings::gafs, &GAFitterSettings::sigmaHalflife);
    addAP(gafAP, "S.GAFitter.constraints[#]", &q_settings, &Settings::gafs, &GAFitterSettings::constraints);
    addAP(gafAP, "S.GAFitter.min[#]", &q_settings, &Settings::gafs, &GAFitterSettings::min);
    addAP(gafAP, "S.GAFitter.max[#]", &q_settings, &Settings::gafs, &GAFitterSettings::max);
    addAP(gafAP, "S.GAFitter.fixedValue[#]", &q_settings, &Settings::gafs, &GAFitterSettings::fixedValue);
    addAP(gafAP, "S.GAFitter.useLikelihood", &q_settings, &Settings::gafs, &GAFitterSettings::useLikelihood);
    addAP(gafAP, "S.GAFitter.cluster_blank_after_step", &q_settings, &Settings::gafs, &GAFitterSettings::cluster_blank_after_step);
    addAP(gafAP, "S.GAFitter.cluster_min_dur", &q_settings, &Settings::gafs, &GAFitterSettings::cluster_min_dur);
    addAP(gafAP, "S.GAFitter.cluster_fragment_dur", &q_settings, &Settings::gafs, &GAFitterSettings::cluster_fragment_dur);
    addAP(gafAP, "S.GAFitter.cluster_threshold", &q_settings, &Settings::gafs, &GAFitterSettings::cluster_threshold);
    addAP(gafAP, "S.GAFitter.useDE", &q_settings, &Settings::gafs, &GAFitterSettings::useDE);
    addAP(gafAP, "S.GAFitter.useClustering", &q_settings, &Settings::gafs, &GAFitterSettings::useClustering);
    addAP(gafAP, "S.GAFitter.mutationSelectivity", &q_settings, &Settings::gafs, &GAFitterSettings::mutationSelectivity);

    Project::addDaqAPs(daqAP, &q_settings.daqd);

    // Defaults
    q_settings.searchd.mapeDimensions = {
        {MAPEDimension::Func::BestBubbleDuration,   0, 0, 128},
        {MAPEDimension::Func::BestBubbleTime,       0, 0,  32},
        {MAPEDimension::Func::VoltageDeviation,     0, 0,  32}
    };
    for ( MAPEDimension &m : q_settings.searchd.mapeDimensions )
        m.setDefaultMinMax(q_settings.stimd);
    q_settings.searchd.precisionIncreaseEpochs = { 100 };

    sanitiseSettings(q_settings);
}

void Session::sanitiseSettings(Settings &s)
{
    if ( s.searchd.nGroupsPerWave > (size_t) project.wavegen().numGroups )
        s.searchd.nGroupsPerWave = project.wavegen().numGroups;
    while ( project.wavegen().numGroups % s.searchd.nGroupsPerWave ||
            !(s.searchd.nGroupsPerWave%32==0 || s.searchd.nGroupsPerWave==16 || s.searchd.nGroupsPerWave==8
              || s.searchd.nGroupsPerWave==4 || s.searchd.nGroupsPerWave==2 || s.searchd.nGroupsPerWave==1) )
        --s.searchd.nGroupsPerWave;
    int nWavesPerKernel = project.wavegen().numGroups / s.searchd.nGroupsPerWave;
    s.searchd.nWavesPerEpoch = ((s.searchd.nWavesPerEpoch + nWavesPerKernel - 1) / nWavesPerKernel) * nWavesPerKernel;

    size_t n = project.model().adjustableParams.size();
    if ( s.gafs.constraints.empty() ) {
        s.gafs.constraints = std::vector<int>(n, 0);
        s.gafs.min.resize(n);
        s.gafs.max.resize(n);
        s.gafs.fixedValue.resize(n);
        for ( size_t i = 0; i < n; i++ ) {
            s.gafs.min[i] = project.model().adjustableParams[i].min;
            s.gafs.max[i] = project.model().adjustableParams[i].max;
            s.gafs.fixedValue[i] = project.model().adjustableParams[i].initial;
        }
    } else if ( s.gafs.constraints.size() != n ) {
        s.gafs.constraints.resize(n);
        s.gafs.min.resize(n);
        s.gafs.max.resize(n);
        s.gafs.fixedValue.resize(n);
    }
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

void Session::pause()
{
    dispatcher.running = false;
}

void Session::resume()
{
    dispatcher.running = true;
    emit doDispatch();
}

void Session::abort()
{
    dispatcher.running = false;
    if ( m_wavegen )
        m_wavegen->abort();
    if ( m_profiler )
        m_profiler->abort();
    if ( m_wavesets)
        m_wavesets->abort();
    if ( m_sprofiler )
        m_sprofiler->abort();
    if ( m_gafitter )
        m_gafitter->abort();
}

void Session::queue(QString actor, QString action, QString args, Result *res, bool wake)
{
    m_log.queue(actor, action, args, res);
    if ( wake )
        emit doDispatch();
}

Settings Session::getSettings(int i) const
{
    auto it = hist_settings.begin() + 1;
    while ( it != hist_settings.end() && it->first < i )
        ++it;
    return (it-1)->second;
}
RunData Session::runData(int i) const { return getSettings(i).rund; }
WavegenData Session::wavegenData(int i) const { return getSettings(i).searchd; }
StimulationData Session::stimulationData(int i) const { return getSettings(i).stimd; }
GAFitterSettings Session::gaFitterSettings(int i) const { return getSettings(i).gafs; }
DAQData Session::daqData(int i) const { return getSettings(i).daqd; }

void Session::load()
{
    Result result;
    for ( int row = 0; row < m_log.rowCount(); row++ ) {
        initial = false;
        SessionLog::Entry entry = m_log.entry(row);
        QString filename = results(row, entry.actor, entry.action);
        QFile file(dir.filePath(filename));
        result.resultIndex = row;
        try {
            if ( entry.actor == wavegen().actorName() )
                wavegen().load(entry.action, entry.args, file, result);
            else if ( entry.actor == profiler().actorName() )
                profiler().load(entry.action, entry.args, file, result);
            else if ( entry.actor == "Config" ) {
                readConfig(file.fileName());
                hist_settings.push_back(std::make_pair(row, q_settings));
                m_settings = q_settings;
            } else if ( entry.actor == wavesets().actorName() ) {
                wavesets().load(entry.action, entry.args, file, result);
            } else if ( entry.actor == gaFitter().actorName() ) {
                gaFitter().load(entry.action, entry.args, file, result);
            } else if ( entry.actor == samplingProfiler().actorName() ) {
                samplingProfiler().load(entry.action, entry.args, file, result);
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

void Session::readConfig(const QString &filename)
{
    std::ifstream is(filename.toStdString());
    QString name;
    AP *it;
    bool hasRun(false), hasSearch(false), hasStim(false), hasGafs(false), hasDaq(false);
    is >> name;
    while ( is.good() ) {
        if ( (it = AP::find(name, &runAP)) ) {
            if ( !hasRun ) {
                q_settings.rund = RunData();
                hasRun = true;
            }
        } else if ( (it = AP::find(name, &searchAP)) ) {
            if ( !hasSearch ) {
                q_settings.searchd = WavegenData();
                hasSearch = true;
            }
        } else if ( (it = AP::find(name, &stimAP)) ) {
            if ( !hasStim ) {
                q_settings.stimd = StimulationData();
                hasStim = true;
            }
        } else if ( (it = AP::find(name, &gafAP)) ) {
            if ( !hasGafs ) {
                q_settings.gafs = GAFitterSettings();
                hasGafs = true;
            }
        } else if ( (it = AP::find(name, &daqAP)) ) {
            if ( !hasDaq ) {
                q_settings.daqd = DAQData();
                hasDaq = true;
            }
        }
        if ( it )
            it->readNow(name, is);
        is >> name;
    }
    if ( hasRun )
        emit runDataChanged();
    if ( hasSearch )
        emit wavegenDataChanged();
    if ( hasStim )
        emit stimulationDataChanged();
    if ( hasGafs )
        emit GAFitterSettingsChanged();
    if ( hasDaq )
        emit DAQDataChanged();
    sanitiseSettings(q_settings);
}

QString Session::results(int idx, const QString &actor, const QString &action)
{
    return QString("%1.%2.%3")
            .arg(idx, 4, 10, QChar('0')) // 4 digits, pad with zeroes
            .arg(actor, action);
}

void Session::setRunData(RunData d)
{
    q_settings.rund = d;
    sanitiseSettings(q_settings);
    m_log.queue("Config", "cfg", "", new Settings(q_settings));
    emit runDataChanged();
}

void Session::setWavegenData(WavegenData d)
{
    q_settings.searchd = d;
    sanitiseSettings(q_settings);
    m_log.queue("Config", "cfg", "", new Settings(q_settings));
    emit wavegenDataChanged();
}

void Session::setStimulationData(StimulationData d)
{
    q_settings.stimd = d;
    sanitiseSettings(q_settings);
    m_log.queue("Config", "cfg", "", new Settings(q_settings));
    emit stimulationDataChanged();
}

void Session::setGAFitterSettings(GAFitterSettings d)
{
    q_settings.gafs = d;
    sanitiseSettings(q_settings);
    m_log.queue("Config", "cfg", "", new Settings(q_settings));
    emit GAFitterSettingsChanged();
}

void Session::setDAQData(DAQData d)
{
    q_settings.daqd = d;
    sanitiseSettings(q_settings);
    m_log.queue("Config", "cfg", "", new Settings(q_settings));
    emit DAQDataChanged();
}



void Dispatcher::dispatch()
{
    busy = true;
    while ( running.load() ) {
        emit requestNextEntry();
        if ( nextEntry.res == nullptr ) // Empty queue: Keep running==true, but wait for next doDispatch signal
            break;

        bool success = false;
        QFile file(s.dir.filePath(s.results(nextEntry.res->resultIndex, nextEntry.actor, nextEntry.action)));

        if ( nextEntry.actor == s.wavegen().actorName() ) {
            success = s.wavegen().execute(nextEntry.action, nextEntry.args, nextEntry.res, file);
        } else if ( nextEntry.actor == s.profiler().actorName() ) {
            success = s.profiler().execute(nextEntry.action, nextEntry.args, nextEntry.res, file);
        } else if ( nextEntry.actor == s.wavesets().actorName() ) {
            success = s.wavesets().execute(nextEntry.action, nextEntry.args, nextEntry.res, file);
        } else if ( nextEntry.actor == s.gaFitter().actorName() ) {
            success = s.gaFitter().execute(nextEntry.action, nextEntry.args, nextEntry.res, file);
        } else if ( nextEntry.actor == s.samplingProfiler().actorName() ) {
            success = s.samplingProfiler().execute(nextEntry.action, nextEntry.args, nextEntry.res, file);
        } else if ( nextEntry.actor == "Config" ) {
            QMutexLocker locker(&mutex);
            std::ofstream os(QFileInfo(file).filePath().toStdString());
            s.m_settings = *static_cast<Settings*>(nextEntry.res);
            delete nextEntry.res;

            // Briefly swap m_ and q_ settings to write out via (q_settings-based) APs
            using std::swap;
            swap(s.m_settings, s.q_settings);
            for ( auto const& p : s.runAP )
                p->write(os);
            for ( auto const& p : s.searchAP )
                p->write(os);
            for ( auto const& p : s.stimAP )
                p->write(os);
            for ( auto const& p : s.gafAP )
                p->write(os);
            for ( auto const& p : s.daqAP )
                p->write(os);
            swap(s.m_settings, s.q_settings);

            s.project.wavegen().setRunData(s.m_settings.rund);
            s.project.experiment().setRunData(s.m_settings.rund);
            s.project.profiler().setRunData(s.m_settings.rund);
            s.hist_settings.push_back(std::make_pair(s.m_settings.resultIndex, s.m_settings));
            success = true;
        }
        emit actionComplete(success);
    }
    busy = false;
}

void Session::getNextEntry()
{
    if ( m_log.queueSize() == 0 ) {
        dispatcher.nextEntry = SessionLog::Entry();
        return;
    }
    if ( initial ) {
        dispatcher.nextEntry = SessionLog::Entry(QDateTime::currentDateTime(), "Config", "cfg", "", new Settings(q_settings));
        initial = false;
    } else {
        dispatcher.nextEntry = m_log.dequeue(true);
    }
    dispatcher.nextEntry.res->resultIndex = m_log.nextIndex();

    // Consolidate consecutive settings entries into one (the latest):
    if ( dispatcher.nextEntry.actor == "Config" ) {
        Settings *set = static_cast<Settings*>(dispatcher.nextEntry.res);
        while (m_log.queueSize() > 0) {
            if ( m_log.entry(set->resultIndex + 1).actor == "Config" ) {
                // consolidate:
                Settings *replacement = static_cast<Settings*>(m_log.dequeue(false).res);
                replacement->resultIndex = set->resultIndex;
                using std::swap;
                swap(set, replacement); // pointer swap
                delete replacement; // delete no longer used older settings object
            } else {
                break;
            }
        }
        dispatcher.nextEntry.res = set;
    }
}

void Session::onActionComplete(bool success)
{
    emit actionLogged(dispatcher.nextEntry.actor, dispatcher.nextEntry.action, dispatcher.nextEntry.args, m_log.nextIndex());
    m_log.clearActive(success);
}

void Session::updateSettings()
{
    // Reload settings from history and existing queue:
    QMutexLocker locker(&dispatcher.mutex);
    q_settings = hist_settings.back().second;

    int fullSize = m_log.rowCount(), qSize = m_log.queueSize();
    for ( int i = fullSize - qSize; i < fullSize; i++ ) {
        SessionLog::Entry e = m_log.entry(i);
        if ( e.actor == "Config" )
            q_settings = Settings(*static_cast<Settings*>(e.res));
    }
}
