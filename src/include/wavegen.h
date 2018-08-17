#ifndef WAVEGEN_H
#define WAVEGEN_H

#include "sessionworker.h"
#include <QVector>
#include "wavegenlibrary.h"

class Wavegen : public SessionWorker
{
    Q_OBJECT
public:
    Wavegen(Session &session);

    const WavegenData &searchd;
    const StimulationData &stimd;

    WavegenLibrary &lib;

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
            nCandidates.reserve(searchd.maxIterations * searchd.nGroupsPerWave);
            nInsertions.reserve(searchd.maxIterations * searchd.nGroupsPerWave);
            nReplacements.reserve(searchd.maxIterations * searchd.nGroupsPerWave);
            meanFitness.reserve(searchd.maxIterations * searchd.nGroupsPerWave);
            maxFitness.reserve(searchd.maxIterations * searchd.nGroupsPerWave);
        }
    };

    inline const std::vector<Archive> &archives() const { return m_archives; }

    //! Returns the archive currently being processed.
    //! Use with caution (i.e. immediately after receiving progress signals), as race conditions apply.
    inline const Archive &currentArchive() const { return current; }

    /**
     * @brief getRandomStim generates a fully randomised stimulation according to the Wavegen's StimulationData
     */
    iStimulation getRandomStim() const;

    inline QString actorName() const { return "Wavegen"; }
    bool execute(QString action, QString args, Result *res, QFile &file);

public slots:
    /**
     * @brief adjustSigmas changes the perturbation factor for adjustableParams such that each perturbation causes
     * roughly the same error (current deviation, vs the corresponding tuned model, during a waveform injection).
     * A number (>= WavegenData::numSigmaAdjustWaveforms) of randomly generated waveforms is used to estimate the average
     * error for each adjustableParam, and an adjustment factor computed to bring the error closer to the median error
     * across all parameters. Multiple invocations may improve convergence.
     * Changes: getErr to true, host err to 0, targetParam to -1, calls @fn detune, @fn settle and @fn stimulate.
     */
    void adjustSigmas();

    /**
     * @brief search runs a MAP-Elites algorithm, evolving Stimulations that maximise fitness for a given parameter.
     * The algorithm will stop after searchd.maxIterations iterations, evaluating one final set of Stimulations before returning.
     * The archive of elite Stimulations are made available in archives() at the end of the search.
     * @see archives()
     * @param param is the index of the parameter to optimise for (0-based, same as for MetaModel::adjustableParams).
     */
    void search(int param);

    /**
     * @brief recalcIstimd populates iStimd from the latest searchd/stimd. Called automatically on session-registered changes.
     */
    void recalcIstimd();

signals:
    void done(int arg = -1);
    void startedSearch(int param);
    void searchTick(int epoch);

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results, Result r);

    /// Helper functions
    bool sigmaAdjust_exec(QFile &file, Result *r);
    void sigmaAdjust_save(QFile &file);
    void sigmaAdjust_load(QFile &file, Result r);
    void propagateAdjustedSigma();

    bool search_exec(QFile &file, Result *r);
    void search_save(QFile &file);
    void search_load(QFile &file, const QString &args, Result r);

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
     * @brief getMeanParamError runs a number of randomly generated stimulations, returning the average error per cycle
     * produced by each parameter detuning.
     */
    std::vector<double> getMeanParamError();

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

    /// MAP-Elites helper functions
    void mape_tournament(std::vector<iStimulation> &);
    void mape_insert(std::vector<MAPElite> &candidates);

    /**
     * @brief mape_bin returns a vector of discretised behavioural measures used as MAPE dimensions.
     * It adheres to the level of precision indicated in current.precision.
     */
    std::vector<size_t> mape_bin(const iStimulation &I);

    /**
     * @brief getSigmaMaxima generates sensible upper bounds on the perturbation factor for each adjustableParam
     */
    std::vector<double> getSigmaMaxima();

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

    static QString sigmaAdjust_action, search_action;
    static quint32 sigmaAdjust_magic, search_magic;
    static quint32 sigmaAdjust_version, search_version;

    struct iStimData
    {
        int iMinStep;
        int iDuration;
    };
    iStimData istimd;
};

#endif // WAVEGEN_H
