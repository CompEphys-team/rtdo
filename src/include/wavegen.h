#ifndef WAVEGEN_H
#define WAVEGEN_H

#include "sessionworker.h"
#include <QVector>
#include <forward_list>
#include "wavegenlibrary.h"
#include "universallibrary.h"

class Wavegen : public SessionWorker
{
    Q_OBJECT
public:
    Wavegen(Session &session);

    const WavegenData &searchd;
    const StimulationData &stimd;

    WavegenLibrary &lib;
    UniversalLibrary &ulib;

    static inline size_t mape_multiplier(size_t precision) { return size_t(1) << precision; }

    struct Archive : public Result
    {
        std::list<MAPElite> elites;
        size_t precision = 0;
        size_t iterations = 0;
        int param;
        QVector<quint32> nCandidates, nInsertions, nReplacements, nElites;
        QVector<double> meanFitness, maxFitness;
        inline QString prettyName() const { return QString("%1 iterations").arg(iterations); }
        Archive(Result r = Result()) : Result(r) {}
        Archive(int param, WavegenData searchd, Result r = Result()) : Result(r), param(param)
        {
            int n = searchd.maxIterations * (param < 0 ? 1 : searchd.nGroupsPerWave);
            nCandidates.reserve(n);
            nInsertions.reserve(n);
            nReplacements.reserve(n);
            meanFitness.reserve(n);
            maxFitness.reserve(n);
        }

        QVector<double> deltabar;
    };

    inline const std::vector<Archive> &archives() const { return m_archives; }

    //! Returns the archive currently being processed.
    //! Use with caution (i.e. immediately after receiving progress signals), as race conditions apply.
    inline const Archive &currentArchive() const { return current; }

    /**
     * @brief getRandomStim generates a fully randomised stimulation according to the Wavegen's StimulationData
     */
    iStimulation getRandomStim(const StimulationData &stimd, const iStimData &istimd) const;

    inline QString actorName() const { return "Wavegen"; }
    bool execute(QString action, QString args, Result *res, QFile &file);

public slots: // Asynchronous calls that queue via Session
    /**
     * @brief clusterSearch searches for Stimulations using Elementary Effects detuning and a clustering algorithm.
     * All parameters are scored simultaneously. The resulting Archive's primary dimension, in addition to any dimensions set by the user,
     * is MAPEDimension::Func::EE_ParamIndex, so the Archive can be easily separated into parameter-specific selections.
     * The MAPElite fitness value is the normalised current deviation for the given parameter's detuning within a cluster
     * as identified by UniversalLibrary::cluster().
     */
    void clusterSearch();

    /**
     * @brief bubbleSearch searches for Stimulations using Elementary Effects detuning and a bubble algorithm.
     * All parameters are scored simultaneously. The resulting Archive's primary dimension, in addition to any dimensions set by the user,
     * is MAPEDimension::Func::EE_ParamIndex, so the Archive can be easily separated into parameter-specific selections.
     * The MAPElite fitness value is the mean ratio between the normalised current deviation for the target parameter's detuning and the mean
     * normalised current deviation for all parameter detunings, within a "bubble" as defined by the former rising above the latter.
     */
    void bubbleSearch();

public: // Synchronous calls
    /**
     * @brief diagnose runs an ad-hoc stimulation through a settled, detuned model group, populating lib.diagDelta with a time course of
     * raw deltaI = I_detuned - I_reference. lib.diagDelta is sized and ordered as (nParams+1) x I.duration, I_reference on parameter index 0.
     * Note, diagnose is performed synchronously and not logged.
     * @param I : a stimulation from any source.
     * @param dt : Temporarily overrides WavegenData::dt
     * @param simCycles : Number of RK4 cycles per dt
     */
    void diagnose(iStimulation I, double dt, int simCycles);

    /**
     * @brief getMeanParamError runs a number of randomly generated stimulations, returning the average error per cycle
     * produced by each parameter detuning.
     * Note, this standalone call does not do any preparatory work (initModels, detune, settle, ...)
     */
    std::vector<double> getMeanParamError();

signals:
    void done(int arg = -1);
    void startedSearch(int param);
    void searchTick(int epoch);

protected:
    friend class Session;
    Result *load(const QString &action, const QString &args, QFile &results, Result r);

    bool cluster_exec(QFile &file, Result *r);
    void cluster_save(QFile &file);
    Result *cluster_load(QFile &file, const QString &args, Result r);
    scalar cluster_scoreAndInsert(const std::vector<iStimulation> &stims, const int nStims, const std::vector<MAPEDimension> &dims);

    bool bubble_exec(QFile &file, Result *r);
    void bubble_save(QFile &file);
    Result *bubble_load(QFile &file, const QString &args, Result r);
    scalar bubble_scoreAndInsert(const std::vector<iStimulation> &stims, const int nStims, const std::vector<MAPEDimension> &dims);

    /**
     * @brief initModels initialises model parameters with random values. Typically, every WavegenData::nGroupsPerWave groups, there
     * is a base parameter set. If only randomised models are to be generated, supply withBase==false.
     * Changes: Host parameter values
     */
    void initModels(bool withBase = true);

    /**
     * @brief detune changes one adjustableParam per model, such that each model block has a tuned version [0]
     * and one detuned version for each adjustableParam.
     * Changes: Host values of adjustable parameters.
     * Note: Detuned values are based on the tuned model's values, such that multiple calls to detune do not change the outcome.
     */
    void detune();

    /**
     * @brief settle puts all models into a settled state clamped at StimulationData::baseV by simulating
     * for the configured time (see RunData::settleTime).
     * Changes: getErr to false, waveforms. Pushes full state to device, simulates, and saves final state on device.
     */
    void settle();

    /**
     * @brief pushStims pushes waveforms to the library, dispersing them as appropriate for the size of @p stim.
     * Changes: waveforms.
     * @param stim A vector with 1, lib.numGroups/searchd.nGroupsPerWave, or lib.numGroups elements.
     */
    void pushStims(const std::vector<iStimulation> &stim);

    /**
     * @brief mutate returns a mutant offspring of the @p parent.
     * @param parent is the primary parent stimulation of the offspring.
     * @param xoverParent is the secondary parent stimulation used for crossover mutation.
     */
    iStimulation mutate(const iStimulation &parent, const iStimulation &xoverParent);

    /// mutate helper functions
    void mutateCrossover(iStimulation&, const iStimulation&);
    void mutateVoltage(iStimulation&);
    void mutateNumber(iStimulation&);
    void mutateSwap(iStimulation&);
    void mutateTime(iStimulation&);
    void mutateType(iStimulation&);
    void construct_next_generation(std::vector<iStimulation> &stims);

    /// Elementary Effects helper functions
    void prepare_EE_models();
    void settle_EE_models();
    void pushStimsAndObserve(const std::vector<iStimulation> &stims, int nModelsPerStim, int blankCycles);
    QVector<double> getDeltabar();
    std::forward_list<MAPElite> sortCandidates(std::vector<std::forward_list<MAPElite>> &candidates_by_param, const std::vector<MAPEDimension> &dims);

    /**
     * @brief baseModelIndex calculates the modelspace index of a given group's tuned model.
     * @param group The global group index
     */
    inline int baseModelIndex(int group) const {
        return group % lib.numGroupsPerBlock                 // Group index within the block
                + (group/lib.numGroupsPerBlock) * lib.numModelsPerBlock; // Modelspace offset of the block this group belongs to
    }

    Archive current;

    std::vector<Archive> m_archives; //!< All archives

    static QString cluster_action, bubble_action;
    static quint32 cluster_magic, bubble_magic;
    static quint32 cluster_version, bubble_version;

    iStimData istimd;
};

#endif // WAVEGEN_H
