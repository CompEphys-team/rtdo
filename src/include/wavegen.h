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


#ifndef WAVEGEN_H
#define WAVEGEN_H

#include "sessionworker.h"
#include <QVector>
#include <forward_list>
#include "universallibrary.h"

class Wavegen : public SessionWorker
{
    Q_OBJECT
public:
    Wavegen(Session &session);

    const WavegenData &searchd;
    const StimulationData &stimd;

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
        QString action;
        QVector<quint32> nCandidates, nInsertions, nReplacements, nElites;
        QVector<double> deltabar;
        double maxCurrent = 0;

        inline QString prettyName() const { return QString("%1 iterations").arg(iterations); }
        WavegenData searchd(const Session &s) const;

        Archive(Result r = Result()) : Result(r) {}
        Archive(QString action, WavegenData searchd, Result r = Result()) : Result(r), action(action)
        {
            nCandidates.reserve(searchd.maxIterations);
            nInsertions.reserve(searchd.maxIterations);
            nReplacements.reserve(searchd.maxIterations);
        }

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

    /**
     * @brief evaluatePremade simulates the given stimulations, observing them at the given times, and returns a MAPElite for each stim
     * that contains the observed deviations and current.
     * Synchronous.
     */
    std::vector<MAPElite> evaluatePremade(const std::vector<iStimulation> &stims, const std::vector<iObservations> &obs);

public slots:
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
     * Asynchronous.
     */
    void search(const QString &action);

    /**
     * @brief findObservations runs the bubble or cluster algorithm on the given stims, picking out the most suitable observations
     * @param stims: The set of iStimulations to be evaluated
     * @param action: The algorithm to be used (Wavegen::bubble_action or Wavegen::cluster_action)
     * @return A MAPElite for each stim, fully populated (excl. bin) with the highest-fitness observation sequence for the target param,
     * ordered by [targetParam][stimIdx]
     * Synchronous.
     */
    std::vector<std::vector<MAPElite>> findObservations(const std::vector<iStimulation> &stims, const QString &action);

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

    void rebinMeanCurrent(size_t meanCurrentDim, const std::vector<MAPEDimension> &dims);

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
    void prepare_models();
    void settle_models();
    void pushStimsAndObserve(const std::vector<iStimulation> &stims, int nModelsPerStim, int blankCycles);
    std::vector<double> getDeltabar();
    std::forward_list<MAPElite> sortCandidates(std::vector<std::forward_list<MAPElite>> &candidates_by_param, const std::vector<MAPEDimension> &dims);

    Archive current;

    std::vector<Archive> m_archives; //!< All archives

    iStimData istimd;
};

#endif // WAVEGEN_H
