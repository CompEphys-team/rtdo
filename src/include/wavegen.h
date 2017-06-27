#ifndef WAVEGEN_H
#define WAVEGEN_H

#include "sessionworker.h"
#include <QVector>
#include "randutils.hpp"
#include "wavegenlibrary.h"

class Wavegen : public SessionWorker
{
    Q_OBJECT
public:
    Wavegen(Session &session);

    const WavegenData &searchd;
    const StimulationData &stimd;

    WavegenLibrary &lib;

    void abort(); //!< Abort all queued actions.

    static inline size_t mape_multiplier(size_t precision) { return size_t(1) << precision; }

    struct Archive {
        std::list<MAPElite> elites;
        size_t precision;
        size_t iterations;
        int param;
        WavegenData searchd;
        inline QString prettyName() const { return QString("%1 iterations").arg(iterations); }
    };

    inline const std::vector<Archive> &archives() const { return m_archives; }

    /**
     * @brief getRandomStim generates a fully randomised stimulation according to the Wavegen's StimulationData
     */
    Stimulation getRandomStim() const;

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

signals:
    void done(int arg = -1);
    void startedSearch(int param);
    void searchTick(int epoch);
    void didAbort();

protected slots:
    void clearAbort();

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results);
    inline QString actorName() const { return "Wavegen"; }

    /// Helper functions
    void sigmaAdjust_save(QFile &file);
    void sigmaAdjust_load(QFile &file);

    void search_save(QFile &file);
    void search_load(QFile &file, const QString &args);

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
     * @brief stimulate runs one full stimulation on every model. In permuted mode, all models receive the same stimulation,
     * namely the first element of @param stim. In unpermuted mode, each model group receives its corresponding stimulation
     * from @param stim.
     * Changes: waveforms.
     * @param stim A vector with at least 1 (permuted mode) or MetaModel::numGroups (unpermuted mode) elements.
     */
    void stimulate(const std::vector<Stimulation> &stim);

    /**
     * @brief mutate returns a mutant offspring of the parent referred to by @p parentIter.
     * @param parentIter is an iterator into mapeArchive
     * @param offset is the position of parentIter within mapeArchive, used for efficient crossover parent lookup.
     */
    Stimulation mutate(const Stimulation &parent, const Stimulation &xoverParent);

    /// mutate helper functions
    void mutateCrossover(Stimulation&, const Stimulation&);
    void mutateVoltage(Stimulation&);
    void mutateNumber(Stimulation&);
    void mutateSwap(Stimulation&);
    void mutateTime(Stimulation&);
    void mutateType(Stimulation&);

    /// MAP-Elites helper functions
    void mape_tournament(const std::vector<Stimulation> &);
    void mape_insert(std::vector<MAPElite> &candidates);

    /**
     * @brief mape_bin returns a vector of discretised behavioural measures used as MAPE dimensions.
     * It adheres to the level of precision indicated in mapeStats.precision.
     */
    std::vector<size_t> mape_bin(const Stimulation &I, const WaveStats &S);

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

    mutable randutils::mt19937_rng RNG;

    std::list<MAPElite> mapeArchive; //!< Elite archive of the current call to search().
    MAPEStats mapeStats; //!< Statistics of the most recent (or current) call to search().

    std::vector<Archive> m_archives; //!< All archives

    bool aborted;

    static QString sigmaAdjust_action, search_action;
    static quint32 sigmaAdjust_magic, search_magic;
    static quint32 sigmaAdjust_version, search_version;
};

#endif // WAVEGEN_H
