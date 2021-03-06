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


#include "project.h"
#include <QDir>
#include <fstream>

Project::Project() :
    loadExisting(false)
{
    addAPs();
}

Project::Project(const QString &projectfile, bool light) :
    p_projectfile(projectfile),
    loadExisting(true)
{
    addAPs();
    loadSettings(projectfile);
    loadExtraModels();

    // Ensure correct UniLib sizing
    size_t n = exp_numCandidates;
    setExpNumCandidates(n);
    if ( n != exp_numCandidates ) {
        compile();
        return;
    }

    // Load library from existing file
    unilib = new UniversalLibrary(*this, false, light);

    frozen = true;
}

Project::~Project()
{
    delete unilib;
}

void Project::loadSettings(const QString &projectfile)
{
    if ( frozen )
        return;

    // Load config from file
    std::ifstream proj(projectfile.toStdString());
    QString name;
    AP *it;
    proj >> name;
    while ( proj.good() ) {
        if ( (it = AP::find(name, &ap)) )
            it->readNow(name, proj);
        proj >> name;
    }

    // Set paths
    setLocation(projectfile);
    setModel(dir() + "/model.xml");
}

void Project::addAPs()
{
    addAP(ap, "Experiment.numCandidates", this, &Project::exp_numCandidates);
    addDaqAPs(ap, &daqd);
    addAP(ap, "sim.extraModels[#]", this, &Project::m_extraModelFiles);
}

void Project::addDaqAPs(std::vector<std::unique_ptr<AP> > &arg, DAQData *p)
{
    addAP(arg, "DAQ.simulate", p, &DAQData::simulate);
    addAP(arg, "DAQ.devNo", p, &DAQData::devNo);
    addAP(arg, "DAQ.throttle", p, &DAQData::throttle);

    QString labels[] = {"DAQ.V", "DAQ.V2", "DAQ.I", "DAQ.Vcmd", "DAQ.Icmd"};
    ChnData DAQData::*chans[] = {&DAQData::voltageChn, &DAQData::V2Chan, &DAQData::currentChn, &DAQData::vclampChan, &DAQData::cclampChan};
    for ( int i = 0; i < 5; i++ ) {
        addAP(arg, labels[i] + ".active", p, chans[i], &ChnData::active);
        addAP(arg, labels[i] + ".idx", p, chans[i], &ChnData::idx);
        addAP(arg, labels[i] + ".range", p, chans[i], &ChnData::range);
        addAP(arg, labels[i] + ".aref", p, chans[i], &ChnData::aref);
        addAP(arg, labels[i] + ".gain", p, chans[i], &ChnData::gain);
        addAP(arg, labels[i] + ".offset", p, chans[i], &ChnData::offset);
    }

    addAP(arg, "DAQ.simd.noise", p, &DAQData::simd, &SimulatorData::noise);
    addAP(arg, "DAQ.simd.noiseStd", p, &DAQData::simd, &SimulatorData::noiseStd);
    addAP(arg, "DAQ.simd.noiseTau", p, &DAQData::simd, &SimulatorData::noiseTau);
    addAP(arg, "DAQ.simd.paramSet", p, &DAQData::simd, &SimulatorData::paramSet);
    addAP(arg, "DAQ.simd.paramValues[#]", p, &DAQData::simd, &SimulatorData::paramValues);
    addAP(arg, "DAQ.simd.outputResolution", p, &DAQData::simd, &SimulatorData::outputResolution);

    addAP(arg, "DAQ.cache.active", p, &DAQData::cache, &CacheData::active);
    addAP(arg, "DAQ.cache.numTraces", p, &DAQData::cache, &CacheData::numTraces);
    addAP(arg, "DAQ.cache.useMedian", p, &DAQData::cache, &CacheData::useMedian);
    addAP(arg, "DAQ.cache.timeout", p, &DAQData::cache, &CacheData::timeout);

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

void Project::setExtraModels(std::vector<QString> modelfiles)
{
    if ( frozen )
        return;
    m_extraModelFiles = std::move(modelfiles);
    loadExtraModels();
}

void Project::loadExtraModels()
{
    m_extraModels.clear();
    m_extraModels.reserve(m_extraModelFiles.size());
    for ( const QString &m : m_extraModelFiles )
        m_extraModels.emplace_back(*this, m.toStdString());
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

std::string Project::simulatorCode() const
{
    if ( !m_model )
        return "";
    std::stringstream ss;
    using std::endl;
    ss << m_model->daqCode(1);
    ss << endl;
    for ( size_t i = 0; i < m_extraModels.size(); i++ )
        ss << m_extraModels.at(i).daqCode(i+2) << endl;
    ss << "inline DAQ *createSim(int simNo, Session &session, const Settings &settings, bool useRealism) {" << endl;
    ss << "    switch ( simNo ) {" << endl;
    ss << "    default:" << endl;
    for ( size_t i = 0; i < m_extraModels.size()+1; i++ )
        ss << "    case " << (i+1) << " : return new Simulator_" << (i+1) << "(session, settings, useRealism);" << endl;
    ss << "    }" << endl;
    ss << "}" << endl;
    ss << "inline void destroySim(DAQ *sim) { delete sim; }" << endl << endl;
    return ss.str();
}

bool Project::compile()
{
    if ( frozen || !m_model || p_modelfile.isEmpty() || p_projectfile.isEmpty() || !QDir().mkpath(dir()) )
        return false;
    QString dest = dir() + "/model.xml";
    if ( dest != p_modelfile ) {
        QFile destFile(dest);
        if ( destFile.exists() )
            destFile.remove();
        QFile::copy(p_modelfile, dest);
    }
    for ( size_t i = 0; i < m_extraModelFiles.size(); i++ ) {
        QFileInfo info(m_extraModelFiles[i]);
        dest = dir() + "/" + info.baseName() + "." + QString::number(i) + ".xml";
        if ( dest != info.absoluteFilePath() ) {
            QFile destFile(dest);
            if ( destFile.exists() )
                destFile.remove();
            QFile::copy(m_extraModelFiles[i], dest);
            m_extraModelFiles[i] = dest;
        }
    }
    loadExtraModels();
    if ( unilib )
        delete unilib;
    unilib = new UniversalLibrary(*this, true);
    std::ofstream proj(p_projectfile.toStdString());
    for ( auto const& p : ap )
        p->write(proj);
    std::ofstream pnames(QString(dir() + "/paramnames").toStdString(), std::ios_base::out | std::ios_base::trunc);
    for ( const AdjustableParam &p : unilib->adjustableParams )
        pnames << p.name << '\t' << p.sigma << '\n';
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
