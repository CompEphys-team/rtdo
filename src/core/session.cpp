#include "session.h"

Session::Session(Project &p) :
    project(p)
{
    project.wavegen().setRunData(rund);
    project.experiment().setRunData(rund);

    moveToThread(&thread);
    thread.start();

    qRegisterMetaType<RunData>();
    qRegisterMetaType<WavegenData>();
    qRegisterMetaType<StimulationData>();
    qRegisterMetaType<ExperimentData>();

    // Connect redirection signals to ensure set* is always called on this->thread
    connect(this, SIGNAL(redirectRunData(RunData)), this, SLOT(setRunData(RunData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectWavegenData(WavegenData)), this, SLOT(setWavegenData(WavegenData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectStimulationData(StimulationData)), this, SLOT(setStimulationData(StimulationData)), Qt::BlockingQueuedConnection);
    connect(this, SIGNAL(redirectExperimentData(ExperimentData)), this, SLOT(setExperimentData(ExperimentData)), Qt::BlockingQueuedConnection);
}

Wavegen &Session::wavegen()
{
    if ( !m_wavegen ) {
        Wavegen *w = new Wavegen(project.wavegen(), stimd, searchd);
        w->moveToThread(&thread);
        m_wavegen.reset(w);
    }
    return *m_wavegen;
}

ErrorProfiler &Session::profiler()
{
    if ( !m_profiler ) {
        ErrorProfiler *p = new ErrorProfiler(project.experiment(), expd);
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

void Session::setRunData(RunData d)
{
    if ( QThread::currentThread() == &thread ) {
        project.wavegen().setRunData(d);
        project.experiment().setRunData(d);
        rund = d;
    } else {
        redirectRunData(d, QPrivateSignal());
    }
}

void Session::setWavegenData(WavegenData d)
{
    if ( QThread::currentThread() == &thread )
        searchd = d;
    else
        redirectWavegenData(d, QPrivateSignal());
}

void Session::setStimulationData(StimulationData d)
{
    if ( QThread::currentThread() == &thread )
        stimd = d;
    else
        redirectStimulationData(d, QPrivateSignal());
}

void Session::setExperimentData(ExperimentData d)
{
    if ( QThread::currentThread() == &thread )
        expd = d;
    else
        redirectExperimentData(d, QPrivateSignal());
}
