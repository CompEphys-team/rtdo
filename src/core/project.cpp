#include "project.h"
#include <QDir>
#include <fstream>

Project::Project() :
    loadExisting(false)
{
    addAPs();
}

void Project::addAPs()
{
    addAP(ap, "dt", this, &Project::m_dt);
    addAP(ap, "method", this, &Project::m_method);
    addAP(ap, "Wavegen.permute", this, &Project::wg_permute);
    addAP(ap, "Wavegen.numWavesPerEpoch", this, &Project::wg_numWavesPerEpoch);
    addAP(ap, "Experiment.numCandidates", this, &Project::exp_numCandidates);
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
    QFile::copy(p_modelfile, dir() + "/model.xml");
    wglib.reset(new WavegenLibrary(*this));
    explib.reset(new ExperimentLibrary(*this));
    std::ofstream proj(p_projectfile.toStdString());
    for ( auto const& p : ap )
        p->write(proj);
    frozen = true;
    return true;
}


