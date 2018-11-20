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

#ifdef USEDOUBLE
#ifndef scalar
typedef double scalar;
#endif
#define scalarmin(a,b) fmin(a,b)
#define scalarmax(a,b) fmax(a,b)
#define scalarfabs(a)  fabs(a)
#else
#ifndef scalar
typedef float scalar;
#endif
#define scalarmin(a,b) fminf(a,b)
#define scalarmax(a,b) fmaxf(a,b)
#define scalarfabs(a)  fabsf(a)
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
struct iStimulation; // Forward

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
    size_t numSteps;

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

    Stimulation() = default;
    Stimulation(const iStimulation &I, double dt);
};
std::ostream &operator<<(std::ostream&, const Stimulation&);
std::ostream &operator<<(std::ostream&, const Stimulation::Step&);

struct iStimulation
{
    int duration;
    int tObsBegin;
    int tObsEnd;
    scalar baseV;

    struct Step
    {
        int t;
        scalar V;
        bool ramp;
        CUDA_HOST_MEMBER bool operator==(const Step &other) const;
    };

    Step steps[Stimulation::maxSteps];
    size_t numSteps;

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

    CUDA_HOST_MEMBER bool operator==(const iStimulation &other) const;

    iStimulation() = default;
    iStimulation(const Stimulation &I, double dt);
    iStimulation(int) : iStimulation() {}
};
std::ostream &operator<<(std::ostream&, const iStimulation&);
std::ostream &operator<<(std::ostream&, const iStimulation::Step&);

struct iObservations
{
    static constexpr size_t maxObs = 10;
    int start[maxObs];
    int stop[maxObs];

    inline void operator=(double) {}
};
std::ostream &operator<<(std::ostream&, const iObservations&);

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
    bool useMedian = false;
    int timeout = 0;
};

enum class FilterMethod { MovingAverage, SavitzkyGolay23, SavitzkyGolay45, SavitzkyGolayEdge3, SavitzkyGolayEdge5 };
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

struct SimulatorData
{
    bool noise = false;
    double noiseTau = 0.01;
    double noiseStd = 2;
    int paramSet = 0; //!< 0: Initial values from model. 1: Random values, uniform in range. 2: Fixed values
    std::vector<scalar> paramValues;
    double outputResolution = 0;
};

struct DAQData
{
    int simulate; // -1: Recorded data; 0: live DAQ; >0: simulated model #(n-1)
    int devNo = 0;
    inline std::string devname() const { std::stringstream ss; ss << "/dev/comedi" << devNo; return ss.str(); }
    int throttle = 0;
    ChnData currentChn;
    ChnData voltageChn;
    ChnData V2Chan;
    ChnData vclampChan;
    ChnData cclampChan;
    CacheData cache;
    FilterData filter;
    SimulatorData simd;
    double varianceDuration = 50;
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

enum class IntegrationMethod { ForwardEuler = 0, RungeKutta4 = 1, RungeKuttaFehlberg45 = 2 };
std::ostream& operator<<(std::ostream& os, const IntegrationMethod &m);
std::istream& operator>>(std::istream& is, IntegrationMethod &m);

enum class ModuleType { Wavegen = 1, Profiler = 2, Universal = 3 };
struct ModelData {
    double dt = 0.25;
    IntegrationMethod method = IntegrationMethod::RungeKutta4;
    QuotedString filepath;
    QuotedString dirpath;
};

struct TypedVariableBase
{
    TypedVariableBase() {}
    TypedVariableBase(std::string n, std::string t) : name(n), type(t) {}
    std::string name;
    std::string type;
};

template <typename T>
struct TypedVariable : TypedVariableBase
{
    TypedVariable() {}
    TypedVariable(std::string n, std::string t = "scalar") : TypedVariableBase(n,t) {}

    T *v, *d_v;

    T *singular_v = nullptr;
    bool singular = false;

    inline T &operator[](std::size_t i) { return v[i]; }
    inline T &operator*() { return v[0]; }
};

struct Variable : public TypedVariable<scalar>
{
    Variable() {}
    Variable(std::string n, std::string c = "", std::string t = "scalar") : TypedVariable<scalar>(n,t), code(c), initial(0.) {}
    std::string code;
    double initial;
    double min = 0;
    double max = 0;
};

struct StateVariable : public Variable {
    StateVariable() {}
    StateVariable(std::string n, std::string c = "", std::string t = "scalar") : Variable(n,c,t) {}
    std::vector<Variable> tmp;
    double tolerance = 1e-3;
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
    double deltaBar = 1;
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
    double Imax = 0; // Current ceiling (+-), nA
    IntegrationMethod integrator = IntegrationMethod::RungeKutta4;
    double dt = 0.25;
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
    std::shared_ptr<iStimulation> wave;
    scalar fitness;
    std::vector<scalar> deviations;
    iObservations obs;

    MAPElite() : fitness(0), obs {{},{}} {}
    MAPElite(std::vector<size_t> bin, std::shared_ptr<iStimulation> wave, scalar fitness)
        : bin(bin), wave(wave), fitness(fitness), obs {{},{}}
    {
        obs.start[0] = wave->tObsBegin;
        obs.stop[0] = wave->tObsEnd;
    }
    MAPElite(std::vector<size_t> bin, std::shared_ptr<iStimulation> wave, scalar fitness, std::vector<scalar> deviations, iObservations obs)
        : bin(bin), wave(wave), fitness(fitness), deviations(deviations), obs(obs) {}

    /**
     * @brief compare performs a lexical comparison on bin.
     * @return true if @p rhs precedes the callee, false otherwise.
     */
    inline bool operator<(const MAPElite &rhs) const { return bin < rhs.bin; }

    /**
     * @brief compete compares the callee's fitness to that of @p rhs, replacing the callee's fitness, wave, obs and deviations if @p rhs is better.
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
        VoltageIntegral,

        EE_ParamIndex,
        EE_NumClusters,
        EE_ClusterIndex
    };

    Func func;
    scalar min, max;
    size_t resolution; //!< initial number of bins

    /**
     * @brief bin classifies the given Stimulation along this dimension. (non-EE functions only)
     * @param I: A Stimulation with a finalised observation window
     * @param multiplier: Multiplier to the resolution (i.e., number of bins = multiplier * resolution)
     * @return The bin index this Stimulation belongs to.
     */
    size_t bin(const iStimulation &I, size_t multiplier, double dt) const;
    size_t bin(scalar value, size_t multiplier) const; //!< As above, but with a fully processed behavioural value
    size_t bin(const iStimulation &I,
               size_t paramIdx, size_t clusterIdx, size_t nClusters,
               size_t multiplier, double dt) const; //!< As above, but for all functions
    scalar bin_inverse(size_t bin, size_t multiplier) const; //!< Inverse function of bin(scalar, size_t), aliased to the lower boundary of the bin.

    void setDefaultMinMax(StimulationData d, size_t nParams);
    inline bool hasVariableResolution() const { return !(func==Func::EE_ClusterIndex || func==Func::EE_NumClusters || func==Func::EE_ParamIndex); }
    inline size_t multiplier(int precisionLevel) const { return hasVariableResolution() ? (size_t(1)<<precisionLevel) : 1; }
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
    double noise_sd = 0.; //!< Expected instrument/environment current noise standard deviation (nA)

    int nTrajectories = 1; //!< Number of EE trajectories with independent starting points for each stimulation. Parameters are cycled across
                           //! trajectories (e.g. b012; b340; b123 etc., where b is a starting point, and the numbers indicate the parameter
                           //! detuned with respect to the precedent. With useBaseParameters==true, the initial model is used as starting point
                           //! until each parameter has been touched at least once. With rerandomiseParameters==true, each free starting point is
                           //! chosen independently, whereas otherwise, every stim gets the same set of starting points. The starting points never
                           //! change between epochs.
                           //! nTraj is adjusted down to fit nTraj*trajLen squarely into UniLib::NMODELS.
    int trajectoryLength = 32; //!< Length of each trajectory, including the starting point. Must be one of {2,4,8,16,32}.
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

    /**
     * @brief constraints: Candidate parameter constraint mode
     * 0: Model preset/initial range
     * 1: custom range (using min/max)
     * 2: fixed (manual, using fixedValue)
     * 3: target value (from DAQ; i.e. simulator value; canned .params file value; or model preset for live recording)
     */
    std::vector<int> constraints;
    std::vector<scalar> min, max, fixedValue;

    bool useLikelihood = false; // Retired 6 Nov 2018

    double cluster_blank_after_step = 5.;
    double cluster_min_dur = 5.;
    double cluster_fragment_dur = 0.5;
    double cluster_threshold = 0.95;

    bool useDE = false;
    bool useClustering = false;
    int mutationSelectivity = 2; // 0: Nonspecific mutation, 1: Graded mutation rates, 2: Target parameter only
};


struct DataPoint { double t; double value; };

struct Result { int resultIndex = -1; };

struct Settings : public Result
{
    WavegenData searchd;
    StimulationData stimd;
    RunData rund;
    DAQData daqd;
    GAFitterSettings gafs;
};

struct Section
{
    int start, end;
    std::vector<double> deviations; //!< Summed deviation per parameter
};

struct ClampParameters
{
    scalar clampGain, accessResistance;
    scalar VClamp0; // = VClamp linearly extrapolated to t=0
    scalar dVClamp; // = VClamp gradient, mV/ms

    CUDA_CALLABLE_MEMBER inline scalar getCurrent(scalar t, scalar V) const
    {
        return (clampGain * (VClamp0 + t*dVClamp - V) - V) / accessResistance;
    }
};

#endif // TYPES_H
