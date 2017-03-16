#ifndef PROJECT_H
#define PROJECT_H

#include "types.h"
#include <QString>
#include <memory>
#include "wavegen.h"
#include "experiment.h"

class Project
{
public:
    /// Create a new project
    Project();

    /// Set compile-time parameters - before compilation only
    void setModel(QString const& modelfile); ///!< Loads the model immediately. At compile time, the model file is copied to the project directory.
    void setLocation(QString const& projectfile); //!< Sets the project file. The project directory is set to the project file's path.
    inline void setDt(double dt) { if ( !frozen ) m_dt = dt; }
    inline void setMethod(IntegrationMethod method) { if ( !frozen ) m_method = method; }
    inline void setWgPermute(bool permute) { if ( !frozen ) wg_permute = permute; }
    inline void setWgWavesPerEpoch(size_t num) { if ( !frozen ) wg_numWavesPerEpoch = num; }
    inline void setExpNumCandidates(size_t num) { if ( !frozen ) exp_numCandidates = num; }

    /// Get compile-time parameters
    inline QString modelfile() const { return p_modelfile; }
    inline double dt() const { return m_dt; }
    inline IntegrationMethod method() const { return m_method; }
    inline bool wgPermute() const { return wg_permute; }
    inline size_t wgWavesPerEpoch() const { return wg_numWavesPerEpoch; }
    inline size_t expNumCandidates() const { return exp_numCandidates; }

    QString dir() const; //!< @brief dir() returns the absolute path to the project directory

    /// Check if compilation has already happened
    inline bool isFrozen() const { return frozen; }

    /// Compile and load both Wavegen and Experiment libraries
    bool compile();

    /// Get the project objects
    MetaModel &model() const { return *m_model; }
    WavegenLibrary &wavegen() const { return *wglib; }
    ExperimentLibrary &experiment() const { return *explib; }

protected:
    QString p_modelfile;
    QString p_projectfile;

    double m_dt = 0.25;
    IntegrationMethod m_method = IntegrationMethod::RungeKutta4;

    bool wg_permute = false; //!< If true, parameters will be permuted, and only one waveform will be used per epoch
    size_t wg_numWavesPerEpoch = 10000; //!< [unpermuted only] Number of waveforms evaluated per epoch

    size_t exp_numCandidates = 10000;

    bool frozen = false;

    bool loadExisting;

    std::unique_ptr<MetaModel> m_model;
    std::unique_ptr<WavegenLibrary> wglib;
    std::unique_ptr<ExperimentLibrary> explib;
};

#endif // PROJECT_H
