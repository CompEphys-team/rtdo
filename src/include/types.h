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
    size_t precision = 0; //!< Precision level
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

struct WavegenData : public RunData
{
    bool permute;
    int numSigmaAdjustWaveforms; //!< Number of random waveforms used to normalise the perturbation rate.
                                 //!< If the MetaModel is not permuted, this number is rounded up to the
                                 //!< nearest multiple of the population size.
    size_t nInitialWaves; //!< Number of randomly initialised waveforms used to start the search

    /**
     * @brief binFunc returns a vector of discretised behavioural measures used as MAPE dimensions.
     * It should adhere to the level of precision indicated in the third function argument.
     */
    std::function<std::vector<size_t>(const Stimulation &, const WaveStats &, size_t precision)> binFunc;
    /**
     * @brief increasePrecision should return true if the algorithm is at a stage where precision should be increased
     * to the next level. Note that increasing precision causes the entire existing archive to be rebinned.
     */
    std::function<bool(const MAPEStats &)> increasePrecision;
    std::function<bool(MAPEStats const&)> stopFunc = [](MAPEStats const&){return true;}; //!< Return true to stop the search.
    size_t historySize;
};

#endif // TYPES_H
