#ifndef PROJECT_H
#define PROJECT_H

#include <QString>
#include <memory>
#include "wavegen.h"
#include "experimentlibrary.h"
#include "profilerlibrary.h"
#include "AP.h"

class Project
{
public:
    /// Create a new project
    Project();

    /// Load project from saved config
    Project(QString const& projectfile);

    void loadSettings(QString const& projectfile); //!< Load settings from an existing file (but not the libraries, and not freezing)

    /// Set compile-time parameters - before compilation only
    void setModel(QString const& modelfile); ///!< Loads the model immediately. At compile time, the model file is copied to the project directory.
    void setExtraModels(std::vector<QString> modelfiles); ///!< Load additional models for use as simulated targets
    void setLocation(QString const& projectfile); //!< Sets the project file. The project directory is set to the project file's path.
    inline void setDt(double dt) { if ( !frozen ) m_dt = dt; }
    inline void setMethod(IntegrationMethod method) { if ( !frozen ) m_method = method; }
    inline void setWgNumGroups(size_t num) { if ( !frozen ) wg_numGroups = num; }
    inline void setExpNumCandidates(size_t num) { if ( !frozen ) exp_numCandidates = num; }
    inline void setProfNumPairs(size_t num) { if ( !frozen ) prof_numPairs = 32*int((num+31)/32); /* ensure numPairs == k*32 */ }

    /// Get compile-time parameters
    inline QString modelfile() const { return p_modelfile; }
    inline const std::vector<QString> &extraModelFiles() const { return m_extraModelFiles; }
    inline double dt() const { return m_dt; }
    inline IntegrationMethod method() const { return m_method; }
    inline size_t wgNumGroups() const { return wg_numGroups; }
    inline size_t expNumCandidates() const { return exp_numCandidates; }
    inline size_t profNumPairs() const { return prof_numPairs; }

    QString dir() const; //!< @brief dir() returns the absolute path to the project directory

    std::string simulatorCode() const; //!< Returns simulator code for all models for use in support code sections

    /// Check if compilation has already happened
    inline bool isFrozen() const { return frozen; }

    /// Compile and load libraries
    bool compile();

    /// Get the project objects
    const MetaModel &model(int simNo = 0) const { return simNo==0 ? *m_model : m_extraModels.at(simNo-1); }
    const std::vector<MetaModel> &extraModels() const { return m_extraModels; }
    WavegenLibrary &wavegen() const { return *wglib; }
    ExperimentLibrary &experiment() const { return *explib; }
    ProfilerLibrary &profiler() const { return *proflib; }

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

    double m_dt = 0.25;
    IntegrationMethod m_method = IntegrationMethod::RungeKutta4;

    size_t wg_numGroups = 8192; //!< Number of model groups (base + each parameter detuned) per lib step

    size_t exp_numCandidates = 10000;

    size_t prof_numPairs = 8192; //!< Number of tuned/detuned model pairs to profile against each other

    DAQData daqd;

    bool frozen = false;

    bool loadExisting;

    std::unique_ptr<MetaModel> m_model;
    std::unique_ptr<WavegenLibrary> wglib;
    std::unique_ptr<ExperimentLibrary> explib;
    std::unique_ptr<ProfilerLibrary> proflib;

    std::vector<QString> m_extraModelFiles;
    std::vector<MetaModel> m_extraModels;

    std::vector<std::unique_ptr<AP>> ap;
};

#endif // PROJECT_H
