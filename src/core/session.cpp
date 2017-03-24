#include "session.h"
#include <QDateTime>

Session::Session(Project &p, const QString &sessiondir) :
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

    m_log.setLogFile(dir.filePath("session.log"));
    load(); // Load state from m_log

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

WavegenSelector &Session::wavegenselector()
{
    if ( !m_wavegenselector ) {
        WavegenSelector *s = new WavegenSelector(*this);
        s->moveToThread(&thread);
        m_wavegenselector.reset(s);
    }
    return *m_wavegenselector;
}

void Session::quit()
{
    if ( m_wavegen )
        m_wavegen->abort();
    if ( m_profiler )
        m_profiler->abort();
    if ( m_wavegenselector )
        m_wavegenselector->abort();
    thread.quit();
    thread.wait();
}

QString Session::log(const SessionWorker *actor, const QString &action, const QString &args)
{
    int idx;

    if ( dirtyRund || dirtySearchd || dirtyStimd || dirtyExpd ) {
        QString actorName = "Config";
        QString cfg = "cfg";
        idx = m_log.put(actorName, cfg, "");
        std::ofstream os(dir.filePath(results(idx, actorName, cfg)).toStdString());
        if ( dirtyRund )
            for ( auto const& p : runAP )
                p->write(os);
        if ( dirtySearchd )
            for ( auto const& p : searchAP )
                p->write(os);
        if ( dirtyStimd )
            for ( auto const& p : stimAP )
                p->write(os);
        if ( dirtyExpd )
            for ( auto const& p : expAP )
                p->write(os);
        dirtyRund = dirtySearchd = dirtyStimd = dirtyExpd = false;
    }

    idx = m_log.put(actor->actorName(), action, args);
    return dir.filePath(results(idx, actor->actorName(), action));
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
                readConfig(filename);
            } else if ( entry.actor == wavegenselector().actorName() ) {
                wavegenselector().load(entry.action, entry.args, file);
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
    bool hasRun(false), hasSearch(false), hasStim(false), hasExp(false);
    is >> name;
    while ( is.good() ) {
        // Make short-circuiting do some work - find the (first) matching AP and set the corresponding boolean to true:
        if ( ((it = AP::find(name, &runAP)) && (hasRun = true))
             || ((it = AP::find(name, &searchAP)) && (hasSearch = true))
             || ((it = AP::find(name, &stimAP)) && (hasStim = true))
             || ((it = AP::find(name, &expAP)) && (hasExp = true))
           )
            it->readNow(name, is);
        is >> name;
    }
    if ( hasRun ) {
        dirtyRund = false;
        project.wavegen().setRunData(rund);
        project.experiment().setRunData(rund);
    }
    if ( hasSearch )
        dirtySearchd = false;
    if ( hasStim )
        dirtyStimd = false;
    if ( hasExp )
        dirtyExpd = false;
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
