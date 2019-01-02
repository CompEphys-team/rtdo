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

    static const QString cluster_action, bubble_action;
    static constexpr quint32 search_magic = 0xc9fd545f;
    static constexpr quint32 search_version = 100;

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
     * @brief search() searches for Stimulations using Elementary Effects detuning and a MAPElites algorithm.
     * All parameters are scored simultaneously. The resulting Archive's primary dimension, in addition to any dimensions set by the user,
     * is MAPEDimension::Func::EE_ParamIndex, so the Archive can be easily separated into parameter-specific selections.
     * @param action (see Wavegen::*_action) defines the fitness function, as follows:
     * - cluster_action: The fitness value is the normalised current deviation for the given parameter's detuning within a cluster
     * as identified by UniversalLibrary::cluster().
     * - bubble_action: The fitness value is the mean ratio between the normalised current deviation for the target parameter's detuning and the mean
     * normalised current deviation for all parameter detunings, within a "bubble" as defined by the former rising above the latter. See also
     * UniversalLibrary::bubble().
     */
    void search(const QString &action);

signals:
    void done();
    void startedSearch(QString action);
    void searchTick(int epoch);

protected:
    friend class Session;
    Result *load(const QString &action, const QString &args, QFile &results, Result r);
    void save(QFile &file, const QString &action);

    scalar cluster_scoreAndInsert(const std::vector<iStimulation> &stims, const int nStims, const std::vector<MAPEDimension> &dims);
    scalar bubble_scoreAndInsert(const std::vector<iStimulation> &stims, const int nStims, const std::vector<MAPEDimension> &dims);

    void insertCandidates(std::forward_list<MAPElite> candidates);

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

    Archive current;

    std::vector<Archive> m_archives; //!< All archives

    iStimData istimd;
};

#endif // WAVEGEN_H
