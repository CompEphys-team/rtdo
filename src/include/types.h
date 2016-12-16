#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <iostream>
#include <list>

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
enum class ModuleType { Experiment = 0, Wavegen = 1 };
struct ModelData {
    double dt;
    int npop;
    IntegrationMethod method;
    ModuleType type;
    bool permute;
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
    int simCycles;
    double clampGain;
    double accessResistance;
    double settleTime = 50; // Duration of initial simulation run to get settled state variable values
};

struct WaveStats
{
    struct Bubble //!< Streak of winning over all other deviations.
    {
        int cycles = 0; //!< Number of cycles won (= target param err > other param errs)
        scalar tEnd = 0; //!< Time at end
        scalar abs = 0; //!< Absolute distance to the next param err down (= target err - next err)
        scalar rel = 0; //!< Relative distance to the next param err down (= (target err - next err)/next err)
        scalar meanAbs = 0; //!< Absolute distance to mean param err
        scalar meanRel = 0; //!< Relative distance to mean param err
    };

    int bubbles = 0; //!< Number of bubbles. Note, all bubbles are contained within buds.
    Bubble totalBubble;
    Bubble currentBubble;
    Bubble longestBubble;
    Bubble bestAbsBubble;
    Bubble bestRelBubble;
    Bubble bestMeanAbsBubble;
    Bubble bestMeanRelBubble;

    int buds = 0; //!< Number of buds, which are winning streaks over the mean, rather than all, deviations.
    Bubble totalBud;
    Bubble currentBud;
    Bubble longestBud;
    Bubble bestAbsBud;
    Bubble bestRelBud;
    Bubble bestMeanAbsBud;
    Bubble bestMeanRelBud;
};
std::ostream &operator<<(std::ostream &os, const WaveStats::Bubble &B);
std::ostream &operator<<(std::ostream&os, const WaveStats&S);

struct MAPElite
{
    std::vector<size_t> bin;
    double fitness;
    Stimulation wave;
    WaveStats *stats;

    MAPElite(std::vector<size_t> bin, double fitness, Stimulation wave) : bin(bin), fitness(fitness), wave(wave), stats(nullptr) {}

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

struct MAPEStats
{
    size_t iterations = 0; //!< Total iterations completed
    size_t insertions = 0; //!< Total number of insertions into archive
    size_t population = 0; //!< Archive size
    size_t historicInsertions = 0; //!< Total insertions within recorded history
    std::list<MAPElite>::const_iterator bestWave; //!< Iterator to the current highest achieving Stimulation in Wavegen::mapeArchive
    WaveStats bestStats; //!< Behavioural statistics of the best wave (unpermuted Wavegen only)

    struct History {
        size_t insertions = 0; //!< Insertions into the archive on this iteration
        size_t population = 0; //!< Archive size at the end of this iteration
        double bestFitness = 0.0; //!< Best fitness value (all-time)
    };
    std::vector<History> history;
    std::vector<History>::iterator histIter; //!< Points at the most recent history entry; advances forward on each iteration.

    MAPEStats(size_t sz, std::list<MAPElite>::const_iterator b) : bestWave(b), history(sz) {}
};

class MAPEDimension;

struct WavegenData : public RunData
{
    int numSigmaAdjustWaveforms; //!< Number of random waveforms used to normalise the perturbation rate.
                                 //!< If the MetaModel is not permuted, this number is rounded up to the
                                 //!< nearest multiple of the population size.
    std::vector<std::shared_ptr<MAPEDimension>> dim; //!< MAP-Elites dimensions to use during search
    size_t nInitialWaves; //!< Number of randomly initialised waveforms used to start the search
    std::function<double(WaveStats const&)> fitnessFunc; //!< Return fitness based on performance statistics
    std::function<bool(MAPEStats const&)> stopFunc = [](MAPEStats const&){return true;}; //!< Return true to stop the search.
    size_t historySize;
};

#endif // TYPES_H
