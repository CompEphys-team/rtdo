#ifndef EXPERIMENTCONSTRUCTOR_H
#define EXPERIMENTCONSTRUCTOR_H

#include "metamodel.h"
#include "daq.h"
#include <functional>

class ExperimentLibrary
{
public:
    ExperimentLibrary(Project const& p);
    virtual ~ExperimentLibrary();

    void GeNN_modelDefinition(NNmodel &);

    inline DAQ *createSimulator() { return pointers.createSim(); }
    inline void destroySimulator(DAQ *sim) { pointers.destroySim(sim); }

    void setRunData(RunData rund); //!< Sets the RunData variables in the library, affecting all future calls to step().

    struct Pointers
    {
        int *simCycles;
        scalar *clampGain;
        scalar *accessResistance;
        scalar *Vmem;
        scalar *Imem;
        bool *VC;
        bool *getErr;

        scalar *err;
        scalar *d_err;

        scalar *t;
        unsigned long long *iT;

        void (*push)(void);
        void (*pull)(void);
        void (*step)(void);
        void (*reset)(void);
        DAQ *(*createSim)();
        void (*destroySim)(DAQ *);
        std::function<void(void)> pushErr;
        std::function<void(void)> pullErr;
    };

    const Project &project;

    MetaModel &model;

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;

    inline void push() { pointers.push(); }
    inline void pull() { pointers.pull(); }
    inline void step() { pointers.step(); }
    inline void reset() { pointers.reset(); }
    inline void pushErr() { pointers.pushErr(); }
    inline void pullErr() { pointers.pullErr(); }

private:
    void *loadLibrary(const std::string &directory);
    std::string simCode();
    std::string supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars);
    std::string daqCode();

    void *lib;

    Pointers (*populate)(std::vector<StateVariable>&, std::vector<AdjustableParam>&);
    Pointers pointers;

public:
    scalar &t;
    unsigned long long &iT;

    // Model globals
    int &simCycles;
    scalar &clampGain;
    scalar &accessResistance;
    scalar &Vmem;
    scalar &Imem;
    bool &VC;
    bool &getErr;

    // Model vars
    scalar *err;
};

#endif // EXPERIMENTCONSTRUCTOR_H
