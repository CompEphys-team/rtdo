#ifndef WAVEGEN_H
#define WAVEGEN_H

#include "types.h"
#include "randutils.hpp"
#include "wavegenconstructor.h"

class Wavegen : public WavegenConstructor
{
public:
    Wavegen(MetaModel &m, const std::string &dir, const StimulationData &p, const WavegenData &r);

    /**
     * @brief permute populates all models with a fresh permutation of adjustableParam values.
     * Changes: Host values of all adjustable parameters; forces resettling on next call to @fn settle.
     */
    void permute();

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
     * Changes: getErr to false, waveforms, t&iT. Pushes full state to device, simulates, and pulls full state to host.
     */
    void settle();

    /**
     * @brief restoreSettled restores the state variables of all models to the state they were in at the end of the last
     * call to @fn settle.
     * Changes: All state variables. Pushes full state to device.
     * @return false iff no settled state was found, e.g. after construction or a call to @fn permute.
     */
    bool restoreSettled();

    /**
     * @brief adjustSigmas changes the perturbation factor for adjustableParams such that each perturbation causes
     * roughly the same error (current deviation, vs the corresponding tuned model, during a waveform injection).
     * A number (>= WavegenData::numSigmaAdjustWaveforms) of randomly generated waveforms is used to estimate the average
     * error for each adjustableParam, and an adjustment factor computed to bring the error closer to the median error
     * across all parameters. Multiple invocations may improve convergence.
     * Changes: getErr to true, host err to 0, targetParam to -1, t&iT; calls @fn detune, @fn restoreSettled and @fn stimulate.
     */
    void adjustSigmas();

    /**
     * @brief stimulate runs one full stimulation on every model. In permuted mode, all models receive the same stimulation,
     * namely the first element of @param stim. In unpermuted mode, each model group receives its corresponding stimulation
     * from @param stim.
     * Changes: waveforms, final, t&iT.
     * @param stim A vector with at least 1 (permuted mode) or MetaModel::numGroups (unpermuted mode) elements.
     */
    void stimulate(const std::vector<Stimulation> &stim);

    /**
     * @brief search runs a MAP-Elites algorithm, evolving Stimulations that maximise fitness for a given parameter.
     * The algorithm will stop when r.stopFunc returns true, evaluating one final set of Stimulations before returning.
     * Runtime statistics and the archive of elite Stimulations are available until the next call to search.
     * @see mapeStats, mapeArchive
     * @param param is the index of the parameter to optimise for (0-based, same as for MetaModel::adjustableParams).
     */
    void search(int param);

    StimulationData p;
    WavegenData r;

protected:
    int blockSize; //!< Number of models (not groups!) per thread block
    int nModels; //!< Total number of models

    /**
     * @brief getRandomStim generates a fully randomised stimulation according to the Wavegen's StimulationData
     */
    Stimulation getRandomStim();

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
     * @brief getSigmaMaxima generates sensible upper bounds on the perturbation factor for each adjustableParam
     */
    static std::vector<double> getSigmaMaxima(const MetaModel &m);

    /**
     * @brief baseModelIndex calculates the modelspace index of a given group's tuned model.
     * @param group The global group index
     */
    inline int baseModelIndex(int group) const {
        return group % numGroupsPerBlock                 // Group index within the block
                + (group/numGroupsPerBlock) * blockSize; // Modelspace offset of the block this group belongs to
    }

    randutils::mt19937_rng RNG;

    std::vector<double> sigmaAdjust;
    std::vector<double> sigmax;

    std::list<std::vector<scalar>> settled;

public:
    std::list<MAPElite> mapeArchive; //!< Elite archive of the most recent (or current) call to search().
    MAPEStats mapeStats; //!< Statistics of the most recent (or current) call to search().
};

#endif // WAVEGEN_H
