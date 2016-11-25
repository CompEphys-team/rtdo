#ifndef WAVEGEN_H
#define WAVEGEN_H

#include "types.h"
#include "kernelhelper.h"
#include <random>
#include <list>

class Wavegen
{
public:
    Wavegen(MetaModel &m);

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
     * for the configured time (@see RunData::settleTime).
     * Changes: getErr to false, Vmem to baseV, t&iT. Pushes full state to device, simulates, and pulls full state to host.
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

    void stimulate(const Stimulation * stim);

    void search();

protected:
    MetaModel &m;
    int blockSize; //!< Number of models (not groups!) per thread block
    int nModels; //!< Total number of models

    /**
     * @brief getRandomStim generates a fully randomised stimulation according to the Wavegen's StimulationData
     */
    Stimulation getRandomStim();

    /**
     * @brief getSigmaMaxima generates sensible upper bounds on the perturbation factor for each adjustableParam
     */
    static std::vector<double> getSigmaMaxima(const MetaModel &m);

    std::default_random_engine gen;
    std::uniform_int_distribution<int> stepDist;
    std::uniform_int_distribution<int> coinflip;
    std::uniform_real_distribution<double> timeDist;
    std::uniform_real_distribution<double> VDist;

    std::vector<double> sigmaAdjust;
    std::vector<double> sigmax;

    std::list<std::vector<scalar>> settled;
};

#endif // WAVEGEN_H
