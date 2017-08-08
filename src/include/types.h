#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <iostream>
#include <sstream>
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
    CUDA_HOST_MEMBER void insert(Step* position, const Step & value);
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
    bool active = true;
    unsigned int idx = 0;
    unsigned int range = 0;
    unsigned int aref = 0;
    double gain = 1.0;
    double offset = 0.0;
};

struct CacheData
{
    bool active = false;
    unsigned int numTraces = 10;
    bool averageWhileCollecting = true;
    bool useMedian = false;
};

enum class FilterMethod { MovingAverage, SavitzkyGolay23, SavitzkyGolay45 };
std::string toString(const FilterMethod &m);
std::ostream& operator<<(std::ostream& os, const FilterMethod &m);
std::istream& operator>>(std::istream& is, FilterMethod &m);

struct FilterData
{
    bool active = false;
    FilterMethod method;
    int samplesPerDt = 10;
    int width;
};

struct DAQData
{
    bool simulate;
    int devNo = 0;
    inline std::string devname() const { std::stringstream ss; ss << "/dev/comedi" << devNo; return ss.str(); }
    ChnData currentChn;
    ChnData voltageChn;
    ChnData stimChn;
    CacheData cache;
    FilterData filter;
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

enum class ModuleType { Experiment = 0, Wavegen = 1, Profiler = 2 };
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
    double min = 0;
    double max = 0;

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
    double sigma;
    double adjustedSigma;
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

struct Bubble
{
    int startCycle;
    int cycles;
    scalar value;
};

struct MAPElite
{
    std::vector<size_t> bin;
    Stimulation wave;
    scalar fitness;

    MAPElite() : fitness(0) {}
    MAPElite(std::vector<size_t> bin, Stimulation wave, scalar fitness) : bin(bin), wave(wave), fitness(fitness) {}

    /**
     * @brief compare performs a lexical comparison on bin.
     * @return true if @p rhs precedes the callee, false otherwise.
     */
    inline bool operator<(const MAPElite &rhs) const { return bin < rhs.bin; }

    /**
     * @brief compete compares the callee's fitness to that of @p rhs, replacing the callee's fitness and wave if @p rhs is better.
     * @return true, if @p rhs beat and replaced the callee.
     */
    bool compete(const MAPElite &rhs);
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
     * @brief bin classifies the given Stimulation along this dimension.
     * @param I: A Stimulation with a finalised observation window
     * @param multiplier: Multiplier to the resolution (i.e., number of bins = multiplier * resolution)
     * @return The bin index this Stimulation belongs to.
     */
    size_t bin(const Stimulation &I, size_t multiplier) const;
    size_t bin(scalar value, size_t multiplier) const; //!< As above, but with a fully processed behavioural value
    scalar bin_inverse(size_t bin, size_t multiplier) const; //!< Inverse function of bin(scalar, size_t), aliased to the lower boundary of the bin.
};
std::string toString(const MAPEDimension::Func &f);
std::ostream& operator<<(std::ostream& os, const MAPEDimension::Func &f);
std::istream& operator>>(std::istream& is, MAPEDimension::Func &f);

struct WavegenData
{
    int numSigmaAdjustWaveforms = 1e5; //!< Number of random waveforms used to normalise the perturbation rate.
                                       //!< If parameters are not permuted, this number is rounded up to the
                                       //!< nearest multiple of the waveform population size.
    size_t nInitialWaves = 1e5; //!< Number of randomly initialised waveforms used to start the search
    size_t nGroupsPerWave = 32; //!< Number of model groups (base + each param detuned) used for each waveform.
                                 //! The number must be a power of two or multiple of 32, as well as an integer divisor of project.wgNumGroups.
                                 //! Values that do not fulfill these conditions are rounded down to the next value that does.
                                 //! Groups are randomised within parameter range; optionally (see useBaseParameters),
                                 //! one group per wave is always the base model.
                                 //! Note that while nGroupsPerWave has a direct impact on runtime, the number of epochs/iterations
                                 //! is independent of it.
    bool useBaseParameters = true; //!< Indicates whether the base parameter set is included among the otherwise randomised parameter
                                   //! sets against which each waveform is evaluated.
    size_t nWavesPerEpoch = 8192; //!< Number of waveforms that constitute one "epoch" or iteration for the purposes of
                                   //! precisionIncreaseEpochs, maxIterations etc. nWavesPerEpoch is rounded up to the nearest
                                   //! multiple of project.wgNumGroups / round_down(nGroupsPerWave).
    bool rerandomiseParameters = false; //!< Indicates whether model parameters should be randomised at each epoch, rather than only
                                        //! once per run.
    std::vector<MAPEDimension> mapeDimensions; //!< List of dimensions along which stimulation behaviour is to be measured
    std::vector<size_t> precisionIncreaseEpochs; //!< Epochs on which MAPE precision/resolution is to double
    size_t maxIterations = 1000; //!< Total number of epochs (cf nWavesPerEpoch)
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


struct DataPoint { double t; double value; };

#endif // TYPES_H
