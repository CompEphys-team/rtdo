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


#ifndef PROJECT_H
#define PROJECT_H

#include <QString>
#include <memory>
#include "wavegen.h"
#include "universallibrary.h"
#include "AP.h"

class Project
{
public:
    /// Create a new project
    Project();

    /// Load project from saved config. Use @a light=true to load a non-functional project for crossloading purposes only
    Project(QString const& projectfile, bool light = false);

    void loadSettings(QString const& projectfile); //!< Load settings from an existing file (but not the libraries, and not freezing)

    /// Set compile-time parameters - before compilation only
    void setModel(QString const& modelfile); ///!< Loads the model immediately. At compile time, the model file is copied to the project directory.
    void setExtraModels(std::vector<QString> modelfiles); ///!< Load additional models for use as simulated targets
    void setLocation(QString const& projectfile); //!< Sets the project file. The project directory is set to the project file's path.
    inline void setExpNumCandidates(size_t num) { if ( !frozen ) exp_numCandidates = 64*int((num+63)/64); /* ensure numPairs == k*64 */ }

    /// Get compile-time parameters
    inline QString modelfile() const { return p_modelfile; }
    inline const std::vector<QString> &extraModelFiles() const { return m_extraModelFiles; }
    inline size_t expNumCandidates() const { return exp_numCandidates; }

    QString dir() const; //!< @brief dir() returns the absolute path to the project directory
    inline QString projectfile() const { return p_projectfile; }

    std::string simulatorCode() const; //!< Returns simulator code for all models for use in support code sections

    /// Check if compilation has already happened
    inline bool isFrozen() const { return frozen; }

    /// Compile and load libraries
    bool compile();

    /// Get the project objects
    const MetaModel &model(int simNo = 0) const { return simNo==0 ? *m_model : m_extraModels.at(simNo-1); }
    MetaModel &model(int simNo = 0) { return simNo==0 ? *m_model : m_extraModels.at(simNo-1); }
    const std::vector<MetaModel> &extraModels() const { return m_extraModels; }

    UniversalLibrary &universal() const { return *unilib; }

    /// Get/Set project default DAQ settings
    const DAQData &daqData() const { return daqd; }
    void setDaqData(DAQData);

    /// Populate an AP vector with DAQData APs (convenience/DRY function for Session)
    static void addDaqAPs(std::vector<std::unique_ptr<AP>> &ap, DAQData *p);

protected:
    void addAPs();

    void loadExtraModels();

    QString p_modelfile;
    QString p_projectfile;

    size_t exp_numCandidates = 10000;

    DAQData daqd;

    bool frozen = false;

    bool loadExisting;

    std::unique_ptr<MetaModel> m_model;
    std::unique_ptr<UniversalLibrary> unilib;

    std::vector<QString> m_extraModelFiles;
    std::vector<MetaModel> m_extraModels;

    std::vector<std::unique_ptr<AP>> ap;
};

#endif // PROJECT_H
