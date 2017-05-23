#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <iostream>
#include <list>
#include "util.h"

#ifndef scalar
#ifdef USEDOUBLE
typedef double scalar;
#else
typedef float scalar;
#endif
#endif

#ifdef __CUDACC__
#define CUDA_HOST_MEMBER __host__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_HOST_MEMBER
#define CUDA_CALLABLE_MEMBER
#endif

class Project; // Forward
class Session; // Forward

struct Stimulation
{
    scalar duration;
    scalar tObsBegin;
    scalar tObsEnd;
    scalar baseV;

    struct Step
    {
        scalar t;
        scalar V;
        bool ramp;
        CUDA_HOST_MEMBER bool operator==(const Step &other) const;
    };

    static constexpr size_t maxSteps = 10;
    Step steps[maxSteps];

    // Some functions that simplify handling steps ... almost as though it were a vector.
    CUDA_CALLABLE_MEMBER inline Step* begin() { return steps; }
    CUDA_CALLABLE_MEMBER inline const Step* begin() const { return steps; }
    CUDA_CALLABLE_MEMBER inline Step* end() { return steps + numSteps; }
    CUDA_CALLABLE_MEMBER inline const Step* end() const { return steps + numSteps; }
    CUDA_HOST_MEMBER void insert(Step* position, Step&& value);
    CUDA_HOST_MEMBER void erase(Step* position);
    CUDA_HOST_MEMBER inline void clear() { numSteps = 0; }
    CUDA_CALLABLE_MEMBER inline size_t size() const { return numSteps; }
    CUDA_CALLABLE_MEMBER inline bool empty() const { return numSteps == 0; }

    CUDA_HOST_MEMBER bool operator==(const Stimulation &other) const;

private:
    size_t numSteps;
};
std::ostream &operator<<(std::ostream&, const Stimulation&);
std::ostream &operator<<(std::ostream&, const Stimulation::Step&);

struct ChnData
{
    bool active;
    unsigned int idx;
    unsigned int range;
    unsigned int aref;
    double gain;
    double offset;
};

struct inChnData : public ChnData
{

};

struct outChnData : public ChnData
{

};

struct DAQData
{
    double dt;
    inChnData currentChn;
    inChnData voltageChn;
    outChnData stimChn;
};

struct CacheData
{
    unsigned int numTraces;
    bool averageWhileCollecting;
    bool useMedian;
};

struct ComediData : public DAQData
{
    std::string devname;
};

struct ThreadData
{
    int priority;
    int stackSize;
    int cpusAllowed;
    int policy;
    int maxMessageSize;
    unsigned long name;
};

enum class IntegrationMethod { ForwardEuler, RungeKutta4 };
std::ostream& operator<<(std::ostream& os, const IntegrationMethod &m);
std::istream& operator>>(std::istream& is, IntegrationMethod &m);

enum class ModuleType { Experiment = 0, Wavegen = 1 };
struct ModelData {
    double dt = 0.25;
    IntegrationMethod method = IntegrationMethod::RungeKutta4;
    QuotedString filepath;
    QuotedString dirpath;
};

struct Variable {
    Variable() {}
    Variable(std::string n, std::string c = "", std::string t = "scalar") : name(n), type(t), code(c), initial(0.) {}
    std::string name;
    std::string type;
    std::string code;
    double initial;

    scalar *v;
    inline scalar &operator[](std::size_t i) { return v[i]; }
};

struct StateVariable : public Variable {
    StateVariable() {}
    StateVariable(std::string n, std::string c = "", std::string t = "scalar") : Variable(n,c,t) {}
    std::vector<Variable> tmp;
};

struct AdjustableParam : public Variable {
    AdjustableParam() {}
    AdjustableParam(std::string n, std::string c = "", std::string t = "scalar") : Variable(n,c,t) {}
    double min;
    double max;
    double sigma;
    bool multiplicative;
    int wgPermutations;
    double wgSD;
    bool wgNormal;
};

struct MutationData
{
    // Relative likelihoods of changes to step properties:
    double lNumber = 1; // Add/remove a step
    double lLevel = 2; // Perturb a step's voltage by +- sdLevel (normal)
    double lType = 1; // Change ramp to step or vice versa
    double lTime = 2; // Perturb a step's time by +- sdTime (normal)
    double lSwap = 0.5; // Swap one step (voltage and ramp) against another
    double lCrossover = 0.2; // Recombine with a second parent; max once per mutation round

    double sdLevel = 10;
    double sdTime = 5;

    int n = 2; // Total number of mutations (+- std). A minimum of one mutation is enforced.
    double std = 2;
};

struct StimulationData
{
    int minSteps = 2;
    int maxSteps = 6;
    scalar minStepLength = 2;
    scalar duration = 300;
    scalar minVoltage = -100;
    scalar maxVoltage = 50;
    scalar baseV = -60;
    MutationData muta;
};

struct RunData
{
    int simCycles = 20;
    double clampGain = 1000;
    double accessResistance = 15; // MOhm
    double settleDuration = 100; // msec
};

struct WaveStats
{
    struct {
        int cycles;
        scalar rel;
    } current;

    struct {
        int cycles; //!< Duration of bubble
        scalar tEnd; //!< Time at end
    } best;

    scalar fitness; //!< Fitness of best bubble (= mean((target err / mean err) - 1) ; target err > mean err )

    int bubbles; //!< Number of bubbles

    inline WaveStats &operator+=(const WaveStats &rhs) {
        best.cycles += rhs.best.cycles;
        best.tEnd += rhs.best.tEnd;
        fitness += rhs.fitness;
        bubbles += rhs.bubbles;
        return *this;
    }
    inline WaveStats &operator/=(int n) {
        best.cycles /= n;
        best.tEnd /= n;
        fitness /= n;
        bubbles /= n;
        return *this;
    }
};
std::ostream &operator<<(std::ostream&os, const WaveStats&S);

struct MAPElite
{
    std::vector<size_t> bin;
    Stimulation wave;
    WaveStats stats;

    MAPElite() { stats.fitness = 0; }
    MAPElite(std::vector<size_t> bin, Stimulation wave, WaveStats stats) : bin(bin), wave(wave), stats(stats) {}

    /**
     * @brief compare performs a lexical comparison on bin.
     * @return true if @p rhs precedes the callee, false otherwise.
     */
    inline bool operator<(const MAPElite &rhs) const { return bin < rhs.bin; }

    /**
     * @brief compete compares the callee's fitness to that of @p rhs, replacing the callee's stats and wave if @p rhs is better.
     * @return true, if @p rhs beat and replaced the callee.
     */
    bool compete(const MAPElite &rhs);
};

struct MAPEStats
{
    size_t iterations = 0; //!< Total iterations completed
    size_t insertions = 0; //!< Total number of insertions into archive
    size_t population = 0; //!< Archive size
    size_t precision = 0; //!< Resolution level (resolution along any dimension == dim.resolution * 2^precision)
    size_t historicInsertions = 0; //!< Total insertions within recorded history
    std::list<MAPElite>::const_iterator bestWave; //!< Iterator to the current highest achieving Stimulation in Wavegen::mapeArchive

    struct History {
        size_t insertions = 0; //!< Insertions into the archive on this iteration
        size_t population = 0; //!< Archive size at the end of this iteration
        double bestFitness = 0.0; //!< Best fitness value (all-time)
    };
    std::vector<History> history;
    std::vector<History>::iterator histIter; //!< Points at the most recent history entry; advances forward on each iteration.

    MAPEStats(size_t sz, std::list<MAPElite>::const_iterator b) : bestWave(b), history(sz) {}
};

struct MAPEDimension
{
    enum class Func {
        BestBubbleDuration,
        BestBubbleTime,
        VoltageDeviation,
        VoltageIntegral
    };

    Func func;
    scalar min, max;
    size_t resolution; //!< initial number of bins

    /**
     * @brief bin classifies the given Stimulation/WaveStats pair along this dimension.
     * @param I: A Stimulation
     * @param S: The behavioural stats corresponding to the Stimulation
     * @param multiplier: Multiplier to the resolution (i.e., number of bins = multiplier * resolution)
     * @return The bin index this Stimulation/WaveStats pair belongs to.
     */
    size_t bin(const Stimulation &I, const WaveStats &S, size_t multiplier) const;
    size_t bin(scalar value, size_t multiplier) const; //!< As above, but with a fully processed behavioural value
    scalar bin_inverse(size_t bin, size_t multiplier) const; //!< Inverse function of bin(scalar, size_t), aliased to the lower boundary of the bin.
};
std::string toString(const MAPEDimension::Func &f);
std::ostream& operator<<(std::ostream& os, const MAPEDimension::Func &f);
std::istream& operator>>(std::istream& is, MAPEDimension::Func &f);

struct WavegenLibraryData
{
    bool permute = false; //!< If true, parameters will be permuted, and only one waveform will be used per epoch
    size_t numWavesPerEpoch = 10000; //!< [unpermuted only] Number of waveforms evaluated per epoch
};

struct WavegenData
{
    int numSigmaAdjustWaveforms = 1e5; //!< Number of random waveforms used to normalise the perturbation rate.
                                       //!< If parameters are not permuted, this number is rounded up to the
                                       //!< nearest multiple of the waveform population size.
    size_t nInitialWaves = 1e5; //!< Number of randomly initialised waveforms used to start the search
    std::vector<MAPEDimension> mapeDimensions; //!< List of dimensions along which stimulation behaviour is to be measured
    std::vector<size_t> precisionIncreaseEpochs; //!< Epochs on which MAPE precision/resolution is to double
    size_t maxIterations = 1000;
    size_t historySize = 20;
};

struct GAFitterSettings {
    size_t maxEpochs = 5000;

    int randomOrder = 1; //!< Stimulation order; 0: sequential, 1: random, 2: biased by error
    double orderBiasDecay = 0.2; //!< The bias for each error is a recursive average: bias = decay*error + (1-decay)*bias
    int orderBiasStartEpoch = 100; //!< The first epoch on which stimulation selection is biased. Before that, the order is sequential for the first nParam epochs, then random.

    size_t nElite = 100;
    size_t nReinit = 100;
    double crossover = 0.0;

    bool decaySigma = true;
    double sigmaInitial = 5;
    double sigmaHalflife = 700;

    int targetType = 0; //!< Parameter set used for simulated fitting; 0: initial, 1: targetValues, 2: randomised
    std::vector<scalar> targetValues;
};

#endif // TYPES_H
