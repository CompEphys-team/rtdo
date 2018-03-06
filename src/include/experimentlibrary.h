#ifndef EXPERIMENTCONSTRUCTOR_H
#define EXPERIMENTCONSTRUCTOR_H

#include "metamodel.h"
#include "daq.h"
#include <functional>

class ExperimentLibrary
{
public:
    ExperimentLibrary(Project const& p, bool compile);
    virtual ~ExperimentLibrary();

    void GeNN_modelDefinition(NNmodel &);

    inline DAQ *createSimulator(int simNo, Session &session, bool useRealismSettings) { return pointers.createSim(simNo, session, useRealismSettings); }
    inline void destroySimulator(DAQ *sim) { pointers.destroySim(sim); }

    void setRunData(RunData rund); //!< Sets the RunData variables in the library, affecting all future calls to step().

    struct Pointers
    {
        int *simCycles;
        scalar *clampGain;
        scalar *accessResistance;
        int *integrator;
        scalar *Vmem;
        scalar *Vprev;
        scalar *Imem;
        bool *VC;
        bool *getErr;
        scalar *VClamp0;
        scalar *dVClamp;
        scalar *tStep;

        scalar *err;
        scalar *d_err;
        scalar *meta_hP, *d_meta_hP;

        scalar *t;
        unsigned long long *iT;

        void (*push)(void);
        void (*pull)(void);
        void (*step)(void);
        void (*reset)(void);
        DAQ *(*createSim)(int simNo, Session&, bool);
        void (*destroySim)(DAQ *);
        std::function<void(void)> pushErr;
        std::function<void(void)> pullErr;
        std::function<void(Variable &)> pushV;
        std::function<void(Variable &)> pullV;
    };

    const Project &project;

    const MetaModel &model;

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;

    inline void push() { pointers.push(); }
    inline void pull() { pointers.pull(); }
    inline void step() { pointers.step(); }
    inline void reset() { pointers.reset(); }
    inline void pushErr() { pointers.pushErr(); }
    inline void pullErr() { pointers.pullErr(); }
    inline void push(Variable &v) { pointers.pushV(v); }
    inline void pull(Variable &v) { pointers.pullV(v); }

private:
    void *load();
    void *compile_and_load();
    std::string simCode();
    std::string supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars);

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
    int &integrator;
    scalar &Vmem;
    scalar &Vprev;
    scalar &Imem;
    bool &VC;
    bool &getErr;
    scalar &VClamp0;
    scalar &dVClamp;
    scalar &tStep;

    // Model vars
    scalar *err;
    scalar *meta_hP;
};

#endif // EXPERIMENTCONSTRUCTOR_H
