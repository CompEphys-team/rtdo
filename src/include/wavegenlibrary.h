#ifndef WAVEGENLIBRARY_H
#define WAVEGENLIBRARY_H

#include "metamodel.h"
#include <functional>

class WavegenLibrary
{
public:
    WavegenLibrary(const Project &p, bool compile);
    ~WavegenLibrary();

    void GeNN_modelDefinition(NNmodel &);

    void setRunData(RunData rund); //!< Sets the RunData variables in the library, affecting all future calls to step().

    struct Pointers
    {
        int *simCycles;
        scalar *clampGain;
        scalar *accessResistance;
        int *targetParam;
        bool *final;
        bool *getErr;

        scalar *err;
        scalar *d_err;

        Stimulation *waveforms;
        Stimulation *d_waveforms;
        WaveStats *wavestats;
        WaveStats *clear_wavestats;
        WaveStats *d_wavestats;

        scalar *t;
        unsigned long long *iT;

        void (*push)(void);
        void (*pull)(void);
        void (*step)(void);
        void (*reset)(void);
        std::function<void(void)> pullStats;
        std::function<void(void)> clearStats;
        std::function<void(void)> pushWaveforms;
        std::function<void(void)> pullWaveforms;
        std::function<void(void)> pushErr;
        std::function<void(void)> pullErr;
    };

    const Project &project;

    MetaModel &model;

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;
    std::vector<Variable> currents;

    int numGroupsPerBlock; //!< Exposes the number of model groups interleaved in each block.
    int numModelsPerBlock; //!< Exposes the total number of models in each thread block.
    int numGroups; //!< Exposes the total number of model groups.
    int numBlocks; //!< Exposes the total number of thread blocks, all of which are fully occupied.
    int numModels; //!< Exposes the total number of models.
    /// Note: A "group" consists of nParams+1 models: A base model, and one detuned model in each parameter.

    inline void push() { pointers.push(); }
    inline void pull() { pointers.pull(); }
    inline void step() { pointers.step(); }
    inline void reset() { pointers.reset(); }
    inline void clearStats() { pointers.clearStats(); }
    inline void pullStats() { pointers.pullStats(); }
    inline void pushWaveforms() { pointers.pushWaveforms(); }
    inline void pullWaveforms() { pointers.pullWaveforms(); }
    inline void pushErr() { pointers.pushErr(); }
    inline void pullErr() { pointers.pullErr(); }

private:
    void *load();
    void *compile_and_load();
    std::string simCode();
    std::string supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars);

    void *lib;

    Pointers (*populate)(WavegenLibrary &);
    Pointers pointers;

public:
    scalar &t;
    unsigned long long &iT;

    // Model globals
    int &simCycles;
    scalar &clampGain;
    scalar &accessResistance;
    int &targetParam;
    bool &final;
    bool &getErr;

    // Model vars
    scalar *err;

    // Group vars
    Stimulation *waveforms;
    WaveStats *wavestats;
};

#endif // WAVEGENLIBRARY_H
