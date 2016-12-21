#ifndef WAVEGENCONSTRUCTOR_H
#define WAVEGENCONSTRUCTOR_H

#include "metamodel.h"
#include <functional>

class WavegenConstructor
{
public:
    WavegenConstructor(MetaModel &m, const std::string &directory);

    void GeNN_modelDefinition(NNmodel &);

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

protected:
    MetaModel &m;

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;
    std::vector<Variable> currents;

    int numGroupsPerBlock; //!< Exposes the number of model groups interleaved in each block.
    int numGroups; //!< Exposes the total number of model groups.
    int numBlocks; //!< Exposes the total number of thread blocks, all of which are fully occupied.
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
    void *loadLibrary(const std::string &directory);
    std::string simCode();
    std::string supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars);

    void *lib;

    Pointers (*populate)(std::vector<StateVariable>&, std::vector<AdjustableParam>&, std::vector<Variable>&);
    Pointers pointers;

protected:
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

#endif // WAVEGENCONSTRUCTOR_H
