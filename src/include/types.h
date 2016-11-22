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
    double t;
    double ot;
    double dur;
    double baseV;
    int N;
    std::vector<double> st;
    std::vector<double> V;
    std::vector<bool> ramp;

    bool operator==(const Stimulation &other) const;
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


#endif // TYPES_H
