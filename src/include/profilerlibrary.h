#ifndef PROFILERLIBRARY_H
#define PROFILERLIBRARY_H

#include "metamodel.h"
#include <functional>

class ProfilerLibrary
{
public:
    ProfilerLibrary(Project const& p, bool compile);
    virtual ~ProfilerLibrary();

    void GeNN_modelDefinition(NNmodel &);

    void setRunData(RunData rund); //!< Sets the RunData variables in the library, affecting all future calls to step().

    struct Pointers
    {
        scalar *dt;
        int *samplingInterval;
        scalar *clampGain;
        scalar *accessResistance;
        scalar **current;
        bool *settling;

        std::vector<scalar*> d_param;

        void (*push)(void);
        void (*pull)(void);
        void (*pushStim)(const iStimulation &);
        void (*step)(void);
        void (*doProfile)(Pointers&, size_t, unsigned int, double&, double&);
        void (*reset)(void);
    };

    const Project &project;

    const MetaModel &model;

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;

    inline void push() { pointers.push(); }
    inline void pull() { pointers.pull(); }
    inline void reset() { pointers.reset(); }

private:
    void *load();
    void *compile_and_load();
    std::string simCode();
    std::string supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars);

    void *lib;

    Pointers (*populate)(std::vector<StateVariable>&, std::vector<AdjustableParam>&);
    Pointers pointers;

public:
    // Model globals
    scalar &dt;
    int &samplingInterval;
    scalar &clampGain;
    scalar &accessResistance;

    void settle(iStimulation stim); // Settle (i.e. apply stim), saving the state back into device memory.
    void profile(iStimulation stim, size_t targetParam, double &accuracy, double &median_norm_gradient);
};

#endif // PROFILERLIBRARY_H
