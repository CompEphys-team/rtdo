#include "project.h"
#include <QDir>
#include <fstream>

Project::Project() :
    loadExisting(false)
{
    addAPs();
}

Project::Project(const QString &projectfile) :
    p_projectfile(projectfile),
    loadExisting(true)
{
    addAPs();

    // Load config from file
    std::ifstream proj(p_projectfile.toStdString());
    QString name;
    AP *it;
    proj >> name;
    while ( proj.good() ) {
        if ( (it = AP::find(name, &ap)) )
            it->readNow(name, proj);
        proj >> name;
    }

    // Load model and libraries from existing files
    setModel(dir() + "/model.xml");
    wglib.reset(new WavegenLibrary(*this, false));
    explib.reset(new ExperimentLibrary(*this, false));
    proflib.reset(new ProfilerLibrary(*this, false));

    frozen = true;
}

void Project::addAPs()
{
    addAP(ap, "dt", this, &Project::m_dt);
    addAP(ap, "method", this, &Project::m_method);
    addAP(ap, "Wavegen.numGroups", this, &Project::wg_numGroups);
    addAP(ap, "Experiment.numCandidates", this, &Project::exp_numCandidates);
    addAP(ap, "Profiler.numPairs", this, &Project::prof_numPairs);
    addDaqAPs(ap, &daqd);
}

void Project::addDaqAPs(std::vector<std::unique_ptr<AP> > &arg, DAQData *p)
{
    addAP(arg, "DAQ.simulate", p, &DAQData::simulate);
    addAP(arg, "DAQ.devNo", p, &DAQData::devNo);

    QString labels[] = {"DAQ.V", "DAQ.I", "DAQ.Vcmd"};
    ChnData DAQData::*chans[] = {&DAQData::voltageChn, &DAQData::currentChn, &DAQData::stimChn};
    for ( int i = 0; i < 3; i++ ) {
        addAP(arg, labels[i] + ".active", p, chans[i], &ChnData::active);
        addAP(arg, labels[i] + ".idx", p, chans[i], &ChnData::idx);
        addAP(arg, labels[i] + ".range", p, chans[i], &ChnData::range);
        addAP(arg, labels[i] + ".aref", p, chans[i], &ChnData::aref);
        addAP(arg, labels[i] + ".gain", p, chans[i], &ChnData::gain);
        addAP(arg, labels[i] + ".offset", p, chans[i], &ChnData::offset);
    }

    addAP(arg, "DAQ.cache.active", p, &DAQData::cache, &CacheData::active);
    addAP(arg, "DAQ.cache.numTraces", p, &DAQData::cache, &CacheData::numTraces);
    addAP(arg, "DAQ.cache.useMedian", p, &DAQData::cache, &CacheData::useMedian);
    addAP(arg, "DAQ.cache.averageWhileCollecting", p, &DAQData::cache, &CacheData::averageWhileCollecting);

    addAP(arg, "DAQ.filter.active", p, &DAQData::filter, &FilterData::active);
    addAP(arg, "DAQ.filter.samplesPerDt", p, &DAQData::filter, &FilterData::samplesPerDt);
    addAP(arg, "DAQ.filter.method", p, &DAQData::filter, &FilterData::method);
    addAP(arg, "DAQ.filter.width", p, &DAQData::filter, &FilterData::width);
}

void Project::setModel(const QString &modelfile)
{
    if ( frozen )
        return;
    p_modelfile = modelfile;
    m_model.reset(new MetaModel(*this));
}

void Project::setLocation(const QString &projectfile)
{
    if ( frozen )
        return;
    p_projectfile = projectfile;
}

QString Project::dir() const {
    if ( p_projectfile.isEmpty() )
        return "";
    else
        return QFileInfo(p_projectfile).absoluteDir().absolutePath();
}

bool Project::compile()
{
    if ( frozen || !m_model || p_modelfile.isEmpty() || p_projectfile.isEmpty() )
        return false;
    QString dest = dir() + "/model.xml";
    QFile destFile(dest);
    if ( destFile.exists() )
        destFile.remove();
    QFile::copy(p_modelfile, dest);
    wglib.reset(new WavegenLibrary(*this, true));
    explib.reset(new ExperimentLibrary(*this, true));
    proflib.reset(new ProfilerLibrary(*this, true));
    std::ofstream proj(p_projectfile.toStdString());
    for ( auto const& p : ap )
        p->write(proj);
    frozen = true;
    return true;
}

void Project::setDaqData(DAQData p)
{
    daqd = p;
    if ( frozen ) {
        std::ofstream proj(p_projectfile.toStdString());
        for ( auto const& p : ap )
            p->write(proj);
    }
}


