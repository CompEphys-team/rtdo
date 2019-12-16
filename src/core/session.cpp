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
    dispatcher.dir = dir;

    dispatcher.moveToThread(&thread);
    thread.start();
    connect(this, SIGNAL(doDispatch()), &dispatcher, SLOT(dispatch()));
    connect(&dispatcher, SIGNAL(actionComplete(bool)), this, SLOT(onActionComplete(bool)), Qt::BlockingQueuedConnection);
    connect(&dispatcher, SIGNAL(requestNextEntry()), this, SLOT(getNextEntry()), Qt::BlockingQueuedConnection);
    connect(&m_log, SIGNAL(queueAltered()), this, SLOT(updateSettings()));

    q_settings.daqd = project.daqData(); // Load project defaults

    m_log.setLogFile(dir.filePath("session.log"));
    load(); // Load state from m_log
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
        QString filename = resultFileName(row, entry.actor, entry.action);
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

void Session::loadConfig(const QString &configFile)
{
    if ( readConfig(configFile, true) )
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
    addAP(runAP, "S.Run.VC", &q_settings, &Settings::rund, &RunData::VC);
    addAP(runAP, "S.Run.Imax", &q_settings, &Settings::rund, &RunData::Imax);
    addAP(runAP, "S.Run.noisy", &q_settings, &Settings::rund, &RunData::noisy);
    addAP(runAP, "S.Run.noisyChannels", &q_settings, &Settings::rund, &RunData::noisyChannels);
    addAP(runAP, "S.Run.noiseStd", &q_settings, &Settings::rund, &RunData::noiseStd);
    addAP(runAP, "S.Run.noiseTau", &q_settings, &Settings::rund, &RunData::noiseTau);

    addAP(searchAP, "S.Wavegen.nInitialWaves", &q_settings, &Settings::searchd, &WavegenData::nInitialWaves);
    addAP(searchAP, "S.Wavegen.useBaseParameters", &q_settings, &Settings::searchd, &WavegenData::useBaseParameters);
    addAP(searchAP, "S.Wavegen.precisionIncreaseEpochs[#]", &q_settings, &Settings::searchd, &WavegenData::precisionIncreaseEpochs);
    addAP(searchAP, "S.Wavegen.maxIterations", &q_settings, &Settings::searchd, &WavegenData::maxIterations);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].func", &q_settings, &Settings::searchd, &WavegenData::mapeDimensions, &MAPEDimension::func);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].min", &q_settings, &Settings::searchd, &WavegenData::mapeDimensions, &MAPEDimension::min);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].max", &q_settings, &Settings::searchd, &WavegenData::mapeDimensions, &MAPEDimension::max);
    addAP(searchAP, "S.Wavegen.mapeDimensions[#].resolution", &q_settings, &Settings::searchd, &WavegenData::mapeDimensions, &MAPEDimension::resolution);
    addAP(searchAP, "S.Wavegen.nTrajectories", &q_settings, &Settings::searchd, &WavegenData::nTrajectories);
    addAP(searchAP, "S.Wavegen.trajectoryLength", &q_settings, &Settings::searchd, &WavegenData::trajectoryLength);
    addAP(searchAP, "S.Wavegen.nDeltabarRuns", &q_settings, &Settings::searchd, &WavegenData::nDeltabarRuns);
    addAP(searchAP, "S.Wavegen.adjustToMaxCurrent", &q_settings, &Settings::searchd, &WavegenData::adjustToMaxCurrent);
    addAP(searchAP, "S.Wavegen.cluster.blank", &q_settings, &Settings::searchd, &WavegenData::cluster, &ClusterData::blank);
    addAP(searchAP, "S.Wavegen.cluster.minLen", &q_settings, &Settings::searchd, &WavegenData::cluster, &ClusterData::minLen);
    addAP(searchAP, "S.Wavegen.cluster.secLen", &q_settings, &Settings::searchd, &WavegenData::cluster, &ClusterData::secLen);
    addAP(searchAP, "S.Wavegen.cluster.dotp_threshold", &q_settings, &Settings::searchd, &WavegenData::cluster, &ClusterData::dotp_threshold);

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
    addAP(stimAP, "S.Stimulation.endWithRamp", &q_settings, &Settings::stimd, &StimulationData::endWithRamp);

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
    addAP(gafAP, "S.GAFitter.sigma[#]", &q_settings, &Settings::gafs, &GAFitterSettings::sigma);
    addAP(gafAP, "S.GAFitter.useLikelihood", &q_settings, &Settings::gafs, &GAFitterSettings::useLikelihood);
    addAP(gafAP, "S.GAFitter.useDE", &q_settings, &Settings::gafs, &GAFitterSettings::useDE);
    addAP(gafAP, "S.GAFitter.useClustering", &q_settings, &Settings::gafs, &GAFitterSettings::useClustering);
    addAP(gafAP, "S.GAFitter.mutationSelectivity", &q_settings, &Settings::gafs, &GAFitterSettings::mutationSelectivity);
    addAP(gafAP, "S.GAFitter.obsSource", &q_settings, &Settings::gafs, &GAFitterSettings::obsSource);
    addAP(gafAP, "S.GAFitter.chunkDuration", &q_settings, &Settings::gafs, &GAFitterSettings::chunkDuration);
    addAP(gafAP, "S.GAFitter.cl_nStims", &q_settings, &Settings::gafs, &GAFitterSettings::cl_nStims);
    addAP(gafAP, "S.GAFitter.cl_nSelect", &q_settings, &Settings::gafs, &GAFitterSettings::cl_nSelect);
    addAP(gafAP, "S.GAFitter.cl_validation_interval", &q_settings, &Settings::gafs, &GAFitterSettings::cl_validation_interval);
    addAP(gafAP, "S.GAFitter.DE_decay", &q_settings, &Settings::gafs, &GAFitterSettings::DE_decay);
    addAP(gafAP, "S.GAFitter.num_populations", &q_settings, &Settings::gafs, &GAFitterSettings::num_populations);
    addAP(gafAP, "S.GAFitter.cl.err_weight_trace", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::err_weight_trace);
    addAP(gafAP, "S.GAFitter.cl.Kfilter", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::Kfilter);
    addAP(gafAP, "S.GAFitter.cl.Kfilter2", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::Kfilter2);
    addAP(gafAP, "S.GAFitter.cl.err_weight_sdf", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::err_weight_sdf);
    addAP(gafAP, "S.GAFitter.cl.spike_threshold", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::spike_threshold);
    addAP(gafAP, "S.GAFitter.cl.sdf_tau", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::sdf_tau);
    addAP(gafAP, "S.GAFitter.cl.err_weight_dmap", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::err_weight_dmap);
    addAP(gafAP, "S.GAFitter.cl.tDelay", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::tDelay);
    addAP(gafAP, "S.GAFitter.cl.dmap_low", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::dmap_low);
    addAP(gafAP, "S.GAFitter.cl.dmap_step", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::dmap_step);
    addAP(gafAP, "S.GAFitter.cl.dmap_sigma", &q_settings, &Settings::gafs, &GAFitterSettings::cl, &ClosedLoopData::dmap_sigma);

    addAP(cdaqAP, "rec.Iidx", &cdaq_assoc, &CannedDAQ::ChannelAssociation::Iidx);
    addAP(cdaqAP, "rec.Vidx", &cdaq_assoc, &CannedDAQ::ChannelAssociation::Vidx);
    addAP(cdaqAP, "rec.V2idx", &cdaq_assoc, &CannedDAQ::ChannelAssociation::V2idx);
    addAP(cdaqAP, "rec.Iscale", &cdaq_assoc, &CannedDAQ::ChannelAssociation::Iscale);
    addAP(cdaqAP, "rec.Vscale", &cdaq_assoc, &CannedDAQ::ChannelAssociation::Vscale);
    addAP(cdaqAP, "rec.V2scale", &cdaq_assoc, &CannedDAQ::ChannelAssociation::V2scale);

    Project::addDaqAPs(daqAP, &q_settings.daqd);

    // Defaults
    size_t nParams = project.model().adjustableParams.size();
    q_settings.searchd.mapeDimensions = {
        {MAPEDimension::Func::EE_ParamIndex,        0, scalar(nParams),             nParams},
        {MAPEDimension::Func::BestBubbleDuration,   0, q_settings.stimd.duration,   64},
        {MAPEDimension::Func::EE_MeanCurrent,       0, 10000,                       64},
    };
    q_settings.searchd.precisionIncreaseEpochs = { 100 };
    q_settings.daqd.simd.paramValues.resize(nParams);
    q_settings.gafs.constraints.resize(nParams, 0);
    q_settings.gafs.min.resize(nParams);
    q_settings.gafs.max.resize(nParams);
    q_settings.gafs.fixedValue.resize(nParams);
    q_settings.gafs.sigma.resize(nParams);
    for ( size_t i = 0; i < nParams; i++ ) {
        const AdjustableParam &p = project.model().adjustableParams[i];
        q_settings.daqd.simd.paramValues[i] = p.initial;
        q_settings.gafs.fixedValue[i] = p.initial;
        q_settings.gafs.min[i] = p.min;
        q_settings.gafs.max[i] = p.max;
        q_settings.gafs.sigma[i] = p.sigma;
    }

    sanitiseSettings(q_settings);
}

void Session::sanitiseSettings(Settings &s)
{
    if ( s.searchd.trajectoryLength >= 32 )
        s.searchd.trajectoryLength = 32;
    else if ( s.searchd.trajectoryLength >= 16 )
        s.searchd.trajectoryLength = 16;
    else if ( s.searchd.trajectoryLength >= 8 )
        s.searchd.trajectoryLength = 8;
    else if ( s.searchd.trajectoryLength >= 4 )
        s.searchd.trajectoryLength = 4;
    else
        s.searchd.trajectoryLength = 2;

    int nTrajTotal = project.universal().NMODELS / s.searchd.trajectoryLength; // Guaranteed integer, NMODELS is k*64
    while ( nTrajTotal % s.searchd.nTrajectories )
        --s.searchd.nTrajectories;

    size_t n = project.model().adjustableParams.size();
    if ( s.searchd.mapeDimensions.front().func != MAPEDimension::Func::EE_ParamIndex )
        s.searchd.mapeDimensions.insert(s.searchd.mapeDimensions.begin(), MAPEDimension {MAPEDimension::Func::EE_ParamIndex, 0, scalar(n), n});
    for ( size_t i = 1; i < s.searchd.mapeDimensions.size(); i++ ) {
        if ( s.searchd.mapeDimensions[i].func == MAPEDimension::Func::EE_ParamIndex ) {
            s.searchd.mapeDimensions.erase(s.searchd.mapeDimensions.begin() + i);
            --i;
        }
    }

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

void Session::desiccate(const QString &file, const QString &directory)
{
    if ( dispatcher.busy.load() )
        return;
    m_log.setDesiccateFile(file, project.projectfile(), dir.absolutePath());
    dispatcher.dir = QDir(directory);
    dispatcher.dryrun = true;
    dispatcher.running = true;
    emit doDispatch();
}

void Session::exec_desiccated(const QString &file, bool nogui)
{
    SessionLog plannedLog;
    plannedLog.setLogFile(file);
    QDir dir_orig = dir;
    dir.setPath(file);
    dir.cdUp();
    std::vector<SessionLog::Entry> plannedActions = load(true, &plannedLog, m_log.rowCount());
    dir = dir_orig;
    for ( SessionLog::Entry &e : plannedActions ) {
        e.res->dryrun = false;
        m_log.queue(e);
    }

    if ( nogui ) {
        int nTotal = plannedActions.size();
        connect(this, &Session::actionLogged, this, [=](QString, QString, QString, int idx){
            static int nDone = 0;
            std::cout << "Action " << ++nDone << '/' << nTotal << " complete: " << m_log.data(m_log.index(idx, 0), Qt::UserRole).toString() << std::endl;
        });

        dispatcher.running = true;
        emit doDispatch();
    }
}

void Session::queue(QString actor, QString action, QString args, Result *res)
{
    m_log.queue(actor, action, args, res);
}

Settings Session::getSettings(int i) const
{
    // Check for queued settings
    for ( int firstQueued = m_log.rowCount() - m_log.queueSize(); i >= firstQueued; i-- )
        if ( m_log.entry(i).actor == "Config" )
            return *static_cast<Settings*>(m_log.entry(i).res);

    // Fall back to historic settings
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

std::vector<SessionLog::Entry> Session::load(bool dryrun, SessionLog *log, int rowOffset)
{
    if ( !log )
        log =& m_log;
    Result result;
    std::vector<SessionLog::Entry> ret;
    for ( int row = 0; row < log->rowCount(); row++ ) {
        initial = false;
        SessionLog::Entry entry = log->entry(row);
        QString filename = resultFileName(row + rowOffset, entry.actor, entry.action);
        QFile file(dir.filePath(filename));
        result.resultIndex = row + rowOffset;
        result.dryrun = dryrun;
        try {
            if ( entry.actor == wavegen().actorName() )
                entry.res = wavegen().load(entry.action, entry.args, file, result);
            else if ( entry.actor == profiler().actorName() )
                entry.res = profiler().load(entry.action, entry.args, file, result);
            else if ( entry.actor == "Config" ) {
                readConfig(file.fileName());
                if ( dryrun ) {
                    entry.res = new Settings(q_settings);
                } else {
                    hist_settings.push_back(std::make_pair(row + rowOffset, q_settings));
                    m_settings = q_settings;
                    entry.res =& hist_settings.back().second;
                }
            } else if ( entry.actor == wavesets().actorName() ) {
                entry.res = wavesets().load(entry.action, entry.args, file, result);
            } else if ( entry.actor == gaFitter().actorName() ) {
                entry.res = gaFitter().load(entry.action, entry.args, file, result);
            } else if ( entry.actor == samplingProfiler().actorName() ) {
                entry.res = samplingProfiler().load(entry.action, entry.args, file, result);
            } else {
                throw std::runtime_error(std::string("Unknown actor: ") + entry.actor.toStdString());
            }
        } catch (std::runtime_error err) {
            std::cerr << "An action could not be loaded (" << log->data(log->index(row, 0), Qt::UserRole).toString()
                      << ", " << filename << ") : "
                      << err.what() << std::endl;
        }
        if ( !dryrun )
            entry.res = nullptr;
        ret.push_back(entry);
    }
    return ret;
}

bool Session::readConfig(const QString &filename, bool incremental)
{
    std::ifstream is(filename.toStdString());
    QString name;
    AP *it;
    bool hasRun(false), hasSearch(false), hasStim(false), hasGafs(false), hasDaq(false), hasCDaq(false);
    bool chgRun(false), chgSearch(false), chgStim(false), chgGafs(false), chgDaq(false), chgCDaq(false);
    is >> name;
    while ( is.good() ) {
        if ( (it = AP::find(name, &runAP)) ) {
            if ( !incremental && !hasRun ) {
                q_settings.rund = RunData();
                hasRun = true;
            }
            chgRun |= it->readNow(name, is);
        } else if ( (it = AP::find(name, &searchAP)) ) {
            if ( !hasSearch ) {
                q_settings.searchd = WavegenData();
                hasSearch = true;
            }
            chgSearch |= it->readNow(name, is);
        } else if ( (it = AP::find(name, &stimAP)) ) {
            if ( !incremental && !hasStim ) {
                q_settings.stimd = StimulationData();
                hasStim = true;
            }
            chgStim |= it->readNow(name, is);
        } else if ( (it = AP::find(name, &gafAP)) ) {
            if ( !incremental && !hasGafs ) {
                q_settings.gafs = GAFitterSettings();
                hasGafs = true;
            }
            chgGafs |= it->readNow(name, is);
        } else if ( (it = AP::find(name, &daqAP)) ) {
            if ( !incremental && !hasDaq ) {
                q_settings.daqd = DAQData();
                hasDaq = true;
            }
            chgDaq |= it->readNow(name, is);
        } else if ( (it = AP::find(name, &cdaqAP)) ) {
            if ( !incremental && !hasCDaq ) {
                cdaq_assoc = CannedDAQ::ChannelAssociation();
                hasCDaq = true;
            }
            chgCDaq |= it->readNow(name, is);
        }
        is >> name;
    }
    if ( chgRun )
        emit runDataChanged();
    if ( chgSearch )
        emit wavegenDataChanged();
    if ( chgStim )
        emit stimulationDataChanged();
    if ( chgGafs )
        emit GAFitterSettingsChanged();
    if ( chgDaq )
        emit DAQDataChanged();
    sanitiseSettings(q_settings);
    return chgRun || chgSearch || chgStim || chgGafs || chgDaq; // chgCDaq omitted intentionally, as CDaq does not enter *.cfg output
}

QString Session::resultFileName(int idx) const
{
    return resultFileName(
                idx,
                m_log.data(m_log.index(idx, 1), Qt::DisplayRole).toString(),
                m_log.data(m_log.index(idx, 2), Qt::DisplayRole).toString());
}

QString Session::resultFileName(int idx, const QString &actor, const QString &action) const
{
    return QString("%1.%2.%3")
            .arg(idx, 4, 10, QChar('0'))
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
        QFile file(dir.filePath(s.resultFileName(nextEntry.res->resultIndex, nextEntry.actor, nextEntry.action)));
        nextEntry.res->dryrun = dryrun;

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

            s.hist_settings.push_back(std::make_pair(s.m_settings.resultIndex, s.m_settings));
            success = true;
        }
        emit actionComplete(success);
    }
    dryrun = false;
    dir = s.dir;
    s.m_log.clearDesiccateFile();
    busy = false;

    emit s.dispatchComplete();
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
