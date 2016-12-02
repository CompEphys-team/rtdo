#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>

#ifndef scalar
#ifdef USEDOUBLE
typedef double scalar;
#else
typedef float scalar;
#endif
#endif

struct Stimulation
{
    double duration;
    double tObsBegin;
    double tObsEnd;
    double baseV;

    struct Step
    {
        double t;
        double V;
        bool ramp;
        bool operator==(const Step &other) const;
    };
    std::vector<Step> steps;

    bool operator==(const Stimulation &other) const;
};

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
    double minStepLength = 2;
    double duration = 300;
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

struct WavegenData : public RunData
{
    int numSigmaAdjustWaveforms; // Number of random waveforms used to normalise the perturbation rate.
                                 // If the MetaModel is not permuted, this number is rounded up to the
                                 // nearest multiple of the population size
};

#endif // TYPES_H
