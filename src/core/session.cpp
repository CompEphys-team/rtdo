#include "session.h"
#include <QDateTime>

Session::Session(Project &p) :
    project(p),
    dirtyRund(true),
    dirtySearchd(true),
    dirtyStimd(true),
    dirtyExpd(true)
{
    qRegisterMetaType<RunData>();
    qRegisterMetaType<WavegenData>();
    qRegisterMetaType<StimulationData>();
    qRegisterMetaType<ExperimentData>();

    addAPs();

    // Connect redirection signals to ensure set* is always called on this->thread
    connect(this, SIGNAL(redirectRunData(RunData)), this, SLOT(setRunData(RunData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectWavegenData(WavegenData)), this, SLOT(setWavegenData(WavegenData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectStimulationData(StimulationData)), this, SLOT(setStimulationData(StimulationData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectExperimentData(ExperimentData)), this, SLOT(setExperimentData(ExperimentData)), Qt::BlockingQueuedConnection);

    dir = QDir(project.dir());
    QString sessiondir = QDateTime::currentDateTime().toString("yyyy.MM.dd-hh.mm.ss");
    dir.mkdir(sessiondir);
    dir.cd(sessiondir);

    m_log.setLogFile(dir.filePath("session.log"));

    project.wavegen().setRunData(rund);
    project.experiment().setRunData(rund);

    moveToThread(&thread);
    thread.start();
}

void Session::addAPs()
{
    addAP(runAP, "Run.accessResistance", this, &Session::rund, &RunData::accessResistance);
    addAP(runAP, "Run.clampGain", this, &Session::rund, &RunData::clampGain);
    addAP(runAP, "Run.simCycles", this, &Session::rund, &RunData::simCycles);

    addAP(searchAP, "Wavegen.settleTime", this, &Session::searchd, &WavegenData::settleTime);
    addAP(searchAP, "Wavegen.numSigmaAdjustWaveforms", this, &Session::searchd, &WavegenData::numSigmaAdjustWaveforms);
    addAP(searchAP, "Wavegen.nInitialWaves", this, &Session::searchd, &WavegenData::nInitialWaves);
    addAP(searchAP, "Wavegen.precisionIncreaseEpochs[#]", this, &Session::searchd, &WavegenData::precisionIncreaseEpochs);
    addAP(searchAP, "Wavegen.maxIterations", this, &Session::searchd, &WavegenData::maxIterations);
    addAP(searchAP, "Wavegen.mapeDimensions[#].func", this, &Session::searchd, &WavegenData::mapeDimensions, &MAPEDimension::func);
    addAP(searchAP, "Wavegen.mapeDimensions[#].min", this, &Session::searchd, &WavegenData::mapeDimensions, &MAPEDimension::min);
    addAP(searchAP, "Wavegen.mapeDimensions[#].max", this, &Session::searchd, &WavegenData::mapeDimensions, &MAPEDimension::max);
    addAP(searchAP, "Wavegen.mapeDimensions[#].resolution", this, &Session::searchd, &WavegenData::mapeDimensions, &MAPEDimension::resolution);

    addAP(stimAP, "Stimulation.baseV", this, &Session::stimd, &StimulationData::baseV);
    addAP(stimAP, "Stimulation.duration", this, &Session::stimd, &StimulationData::duration);
    addAP(stimAP, "Stimulation.minSteps", this, &Session::stimd, &StimulationData::minSteps);
    addAP(stimAP, "Stimulation.maxSteps", this, &Session::stimd, &StimulationData::maxSteps);
    addAP(stimAP, "Stimulation.minVoltage", this, &Session::stimd, &StimulationData::minVoltage);
    addAP(stimAP, "Stimulation.maxVoltage", this, &Session::stimd, &StimulationData::maxVoltage);
    addAP(stimAP, "Stimulation.minStepLength", this, &Session::stimd, &StimulationData::minStepLength);
    addAP(stimAP, "Stimulation.muta.lCrossover", this, &Session::stimd, &StimulationData::muta, &MutationData::lCrossover);
    addAP(stimAP, "Stimulation.muta.lLevel", this, &Session::stimd, &StimulationData::muta, &MutationData::lLevel);
    addAP(stimAP, "Stimulation.muta.lNumber", this, &Session::stimd, &StimulationData::muta, &MutationData::lNumber);
    addAP(stimAP, "Stimulation.muta.lSwap", this, &Session::stimd, &StimulationData::muta, &MutationData::lSwap);
    addAP(stimAP, "Stimulation.muta.lTime", this, &Session::stimd, &StimulationData::muta, &MutationData::lTime);
    addAP(stimAP, "Stimulation.muta.lType", this, &Session::stimd, &StimulationData::muta, &MutationData::lType);
    addAP(stimAP, "Stimulation.muta.n", this, &Session::stimd, &StimulationData::muta, &MutationData::n);
    addAP(stimAP, "Stimulation.muta.sdLevel", this, &Session::stimd, &StimulationData::muta, &MutationData::sdLevel);
    addAP(stimAP, "Stimulation.muta.sdTime", this, &Session::stimd, &StimulationData::muta, &MutationData::sdTime);
    addAP(stimAP, "Stimulation.muta.std", this, &Session::stimd, &StimulationData::muta, &MutationData::std);

    addAP(expAP, "Experiment.settleDuration", this, &Session::expd, &ExperimentData::settleDuration);
}

Wavegen &Session::wavegen()
{
    if ( !m_wavegen ) {
        Wavegen *w = new Wavegen(*this);
        w->moveToThread(&thread);
        m_wavegen.reset(w);
    }
    return *m_wavegen;
}

ErrorProfiler &Session::profiler()
{
    if ( !m_profiler ) {
        ErrorProfiler *p = new ErrorProfiler(*this);
        p->moveToThread(&thread);
        m_profiler.reset(p);
    }
    return *m_profiler;
}

void Session::quit()
{
    m_wavegen->abort();
    m_profiler->abort();
    thread.quit();
    thread.wait();
}

QString Session::log(const void *actor, const QString &action, const QString &args)
{
    QString actorName;
    int idx;

    if ( dirtyRund || dirtySearchd || dirtyStimd || dirtyExpd ) {
        actorName = "Config";
        idx = m_log.put(actorName, "set", "");
        QString conffile = QString("%1.%2").arg(idx, 4, 10, QChar('0')).arg(actorName);
        std::ofstream os(dir.filePath(conffile).toStdString());
        if ( dirtyRund )
            for ( auto const& p : runAP )
                p->write(os);
        if ( dirtySearchd)
            for ( auto const& p : searchAP )
                p->write(os);
        if ( dirtyStimd)
            for ( auto const& p : stimAP )
                p->write(os);
        if ( dirtyExpd)
            for ( auto const& p : expAP )
                p->write(os);
    }

    if ( actor == m_wavegen.get() )
        actorName = "Wavegen";
    else if ( actor == m_profiler.get() )
        actorName = "Profiler";
    else
        actorName = "unknown";

    idx = m_log.put(actorName, action, args);
    QString filename = QString("%1.%2.%3").arg(idx, 4, 10, QChar('0')) // 4 digits, pad with zeroes
                                          .arg(actorName, action);
    return dir.filePath(filename);
}

void Session::setRunData(RunData d)
{
    if ( QThread::currentThread() == &thread ) {
        project.wavegen().setRunData(d);
        project.experiment().setRunData(d);
        rund = d;
        dirtyRund = true;
    } else {
        redirectRunData(d, QPrivateSignal());
    }
}

void Session::setWavegenData(WavegenData d)
{
    if ( QThread::currentThread() == &thread ) {
        searchd = d;
        dirtySearchd = true;
    } else {
        redirectWavegenData(d, QPrivateSignal());
    }
}

void Session::setStimulationData(StimulationData d)
{
    if ( QThread::currentThread() == &thread ) {
        stimd = d;
        dirtyStimd = true;
    } else {
        redirectStimulationData(d, QPrivateSignal());
    }
}

void Session::setExperimentData(ExperimentData d)
{
    if ( QThread::currentThread() == &thread ) {
        expd = d;
        dirtyExpd = true;
    } else {
        redirectExperimentData(d, QPrivateSignal());
    }
}
