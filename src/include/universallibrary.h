#ifndef UNIVERSALLIBRARY_H
#define UNIVERSALLIBRARY_H

#include "metamodel.h"
#include "daq.h"
#include <functional>

/// Flags and bitmasks for the `assignment` parameter:
// Bits 0-1: Which target to compare to (summary)
// Caution: _LANE0 and _PREVTHREAD require identical stim/obs and rund within a warp.
#define ASSIGNMENT_SUMMARY_COMPARE_NONE         static_cast<unsigned int>(0x0) /* report raw current */
#define ASSIGNMENT_SUMMARY_COMPARE_TARGET       static_cast<unsigned int>(0x1) /* compare against lib.target */
#define ASSIGNMENT_SUMMARY_COMPARE_LANE0        static_cast<unsigned int>(0x2) /* compare against first lane in warp; first lane does _NONE */
#define ASSIGNMENT_SUMMARY_COMPARE_PREVTHREAD   static_cast<unsigned int>(0x3) /* compare against preceding lane in warp; first lane does _NONE */
#define ASSIGNMENT_SUMMARY_COMPARE_MASK         static_cast<unsigned int>(0x3) /* (mask) */

// Bit 2: Whether to report summary value
#define ASSIGNMENT_REPORT_SUMMARY               static_cast<unsigned int>(0x1 << 2) /* calculate sum over observed in lib.summary */

// Bit 3: Report sum of (squared | abs) values in summary
#define ASSIGNMENT_SUMMARY_SQUARED              static_cast<unsigned int>(0x1 << 3) /* On: sum of squares; Off: sum of absolute values */

// Bit 4: Divide summary by number of observations
#define ASSIGNMENT_SUMMARY_AVERAGE              static_cast<unsigned int>(0x1 << 4) /* Average summary over number of samples */

// Bit 5: Whether to report time series
#define ASSIGNMENT_REPORT_TIMESERIES            static_cast<unsigned int>(0x1 << 5) /* Report a time series in lib.output */

// Bit 6: How to report time series
#define ASSIGNMENT_TIMESERIES_COMPACT           static_cast<unsigned int>(0x1 << 6) /* Omit unreported samples entirely (Off: sparse) */

// Bit 7: Report absolute values in time series
#define ASSIGNMENT_TIMESERIES_ABS               static_cast<unsigned int>(0x1 << 7) /* Report absolute value */

// Bit 8: Maintain state at end of stimulation
#define ASSIGNMENT_MAINTAIN_STATE               static_cast<unsigned int>(0x1 << 8) /* Maintain model state after stimulation */

// Bit 9: Skip all further processing after settling (Note: This resets `summary` to 0 regardless of MAINTAIN_STATE or other instructions)
#define ASSIGNMENT_SETTLE_ONLY                  static_cast<unsigned int>(0x1 << 9) /* Ignore stimulation, settle for iSettleDuration only */

// Bit 10: Zero all timeseries samples that would not otherwise be touched in sparse mode
#define ASSIGNMENT_TIMESERIES_ZERO_UNTOUCHED_SAMPLES    static_cast<unsigned int>(0x1 << 10) /* Sparse mode: set unobserved samples to 0 */

// Bits 11-12: Which target to compare to (time series)
// Caution: _LANE0 and _PREVTHREAD require identical stim/obs and rund within a warp.
#define ASSIGNMENT_TIMESERIES_COMPARE_NONE              static_cast<unsigned int>(0x0 << 11) /* report raw current */
#define ASSIGNMENT_TIMESERIES_COMPARE_TARGET            static_cast<unsigned int>(0x1 << 11) /* compare against lib.target */
#define ASSIGNMENT_TIMESERIES_COMPARE_LANE0             static_cast<unsigned int>(0x2 << 11) /* compare against first lane in warp; first lane does _NONE */
#define ASSIGNMENT_TIMESERIES_COMPARE_PREVTHREAD        static_cast<unsigned int>(0x3 << 11) /* compare against preceding lane in warp; first lane does _NONE */
#define ASSIGNMENT_TIMESERIES_COMPARE_MASK              static_cast<unsigned int>(0x3 << 11) /* (mask) */

// Bit 13: Whether to maintain summary value across calls, rather than resetting to 0
#define ASSIGNMENT_SUMMARY_PERSIST              static_cast<unsigned int>(0x1 << 13) /* Retain summary value across calls */

// Bits 14-16: Singular stim/rund/target
#define ASSIGNMENT_SINGULAR_STIM                 static_cast<unsigned int>(0x1 << 14) /* Load stim/obs from singular __constant__ var */
#define ASSIGNMENT_SINGULAR_RUND                 static_cast<unsigned int>(0x1 << 15) /* Load rundata from singular __constant__ var */
#define ASSIGNMENT_SINGULAR_TARGET                 static_cast<unsigned int>(0x1 << 16) /* Load target from singular __constant__ var */

class UniversalLibrary
{
public:
    UniversalLibrary(Project &p, bool compile, bool light = false);
    virtual ~UniversalLibrary();

    void GeNN_modelDefinition(NNmodel &);

    inline DAQ *createSimulator(int simNo, Session &session, const Settings &settings, bool useRealismSettings) {
        return isLight ? nullptr : pointers.createSim(simNo, session, settings, useRealismSettings);
    }
    inline void destroySimulator(DAQ *sim) { if ( !isLight ) pointers.destroySim(sim); }

    struct Pointers
    {
        int *simCycles;
        IntegrationMethod *integrator;
        unsigned int *assignment;
        int *targetStride;

        std::function<void(void*, void*, size_t)> pushV;
        std::function<void(void*, void*, size_t)> pullV;
        void (*run)(void);
        void (*reset)(void);
        DAQ *(*createSim)(int simNo, Session&, const Settings&, bool);
        void (*destroySim)(DAQ *);

        void (*resizeTarget)(size_t);
        void(*pushTarget)(void);
        scalar **target;

        void (*resizeOutput)(size_t);
        void(*pullOutput)(void);
        scalar **output;

        void (*profile)(int nSamples, int stride, scalar *d_targetParam, double &accuracy, double &median_norm_gradient);

        int (*cluster)(int trajLen, int nTraj, int duration, int secLen, scalar dotp_threshold, int minClusterLen,
                        std::vector<double> deltabar, bool pull);
        void (*pullClusters)(int nStims);
        int (*pullPrimitives)(int nStims, int duration, int secLen);
        std::vector<double> (*find_deltabar)(int trajLen, int nTraj, int duration);
        scalar **clusters;
        scalar **clusterCurrent;
        scalar **clusterPrimitives;
        iObservations **clusterObs;

        void (*observe_no_steps)(int blankCycles);
    };

    Project &project;
    MetaModel &model;
    const size_t NMODELS;

    static constexpr int maxClusters = 32;

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;

    // Library-defined model variables
    // Setting singular_* on turns these into single-value __constant__ parameters rather than one-per-model.
    // When set thus, access the single value through var[0] or *var. Pushing remains necessary, but is automagically delegated.
    TypedVariable<iStimulation> stim;
    TypedVariable<iObservations> obs;
    inline void setSingularStim(bool on = true) {
        stim.singular = obs.singular = on;
        assignment_base = on ? (assignment_base | ASSIGNMENT_SINGULAR_STIM) : (assignment_base & ~ASSIGNMENT_SINGULAR_STIM);
    }

    TypedVariable<scalar> clampGain;
    TypedVariable<scalar> accessResistance;
    TypedVariable<int> iSettleDuration;
    TypedVariable<scalar> Imax;
    TypedVariable<scalar> dt;
    inline void setSingularRund(bool on = true) {
        clampGain.singular = accessResistance.singular = iSettleDuration.singular = Imax.singular = dt.singular = on;
        assignment_base = on ? (assignment_base | ASSIGNMENT_SINGULAR_RUND) : (assignment_base & ~ASSIGNMENT_SINGULAR_RUND);
    }

    TypedVariable<size_t> targetOffset;
    inline void setSingularTarget(bool on = true) {
        targetOffset.singular = on;
        assignment_base = on ? (assignment_base | ASSIGNMENT_SINGULAR_TARGET) : (assignment_base & ~ASSIGNMENT_SINGULAR_TARGET);
    }

    TypedVariable<double> summary;

    void push();
    void pull();
    template <typename T>
    inline void push(TypedVariable<T> &var) {
        pointers.pushV(var.v,
                       var.singular ? var.singular_v : var.d_v,
                       sizeof(T) * (var.singular ? 1 : NMODELS) );
    }
    template <typename T>
    inline void pull(TypedVariable<T> &var) {
        pointers.pullV(var.v,
                       var.singular ? var.singular_v : var.d_v,
                       sizeof(T) * (var.singular ? 1 : NMODELS) );
    }
    inline void run() { pointers.run(); }
    inline void reset() { pointers.reset(); }

    /// Allocates space for nTraces traces of length nSamples and sets targetStride = nTraces
    void resizeTarget(size_t nTraces, size_t nSamples);
    inline void pushTarget() { pointers.pushTarget(); }

    /// Allocates space for NMODELS traces of length nSamples
    void resizeOutput(size_t nSamples);
    inline void pullOutput() { pointers.pullOutput(); }

    void setRundata(size_t modelIndex, const RunData &rund);

    /// post-run() workhorse for SamplingProfiler
    /// Expects assignments TIMESERIES_COMPARE_NONE | TIMESERIES_COMPACT with adjacent pairs of tuned/detuned models
    inline void profile(int nSamples, int stride, size_t targetParam, double &accuracy, double &median_norm_gradient) {
        pointers.profile(nSamples, stride, adjustableParams[targetParam].d_v, accuracy, median_norm_gradient);
    }

    /// post-run() workhorse for elementary effects wavegen
    /// Expects assignment TIMESERIES_COMPARE_NONE with models sequentially detuned along and across their ee trajectories, as well as
    /// an appropriate individual obs in each stim's first trajectory starting point model
    /// @return nPartitions, as required for pullClusters() and further processing of clusterMasks: The number of partitions of 32 secs each
    inline int cluster(int trajLen, /* length of EE trajectory (power of 2, <=32) */
                        int nTraj, /* Number of EE trajectories */
                        int duration, /* nSamples of each trace, cf. resizeOutput() */
                        int secLen, /* sampling resolution (in ticks) */
                        scalar dotp_threshold, /* Minimum scalar product for two sections to be considered to belong to the same cluster */
                        int minClusterLen, /* Minimum duration for a cluster to be considered valid (in ticks) */
                        std::vector<double> deltabar, /* RMS current deviation, cf. find_deltabar() */
                        bool pull_results) /* True to copy results immediately; false to defer to pullClusters(), allowing time for CPU processing */ {
        return pointers.cluster(trajLen, nTraj, duration, secLen, dotp_threshold, minClusterLen, deltabar, pull_results);
    }
    /// Copy cluster() results to clusters, clusterMasks and clusterCurrent.
    /// Note, nStims = NMODELS/(trajLen*nTraj); nPartitions = ((duration+secLen-1)/secLen+31)/32.
    inline void pullClusters(int nStims) { pointers.pullClusters(nStims); }
    /// Copy cluster() intermediates (section primitives) to clusterPrimitives, and return the section stride
    inline int pullPrimitives(int nStims, int duration, int secLen) { return pointers.pullPrimitives(nStims, duration, secLen); }

    /// post-run() calculation of RMS current deviation from each detuned parameter. Reports the RMSD per tick, per single detuning,
    /// as required by cluster().
    /// Expects assignment TIMESERIES_COMPARE_PREVTHREAD with models sequentially detuned along and across their ee trajectories, as well as
    /// an appropriate individual obs in each stim's first trajectory starting point model
    inline std::vector<double> find_deltabar(int trajLen, int nTraj, int duration) { return pointers.find_deltabar(trajLen, nTraj, duration); }

    /// Utility call to add full-stim, step-blanked observation windows to the (model-individual) stims residing on the GPU
    inline void observe_no_steps(int blankCycles) { pointers.observe_no_steps(blankCycles); }

private:
    void *load();
    void *compile_and_load();
    std::string simCode();
    std::string supportCode(const std::vector<Variable> &globals);

    void *lib;

    Pointers (*populate)(UniversalLibrary&);
    Pointers pointers;

    bool isLight;
    int dummyInt;
    unsigned int dummyUInt;
    IntegrationMethod dummyIntegrator;
    scalar *dummyScalarPtr = nullptr;
    int *dummyIntPtr = nullptr;
    unsigned int *dummyUIntPtr = nullptr;
    iObservations *dummyObsPtr = nullptr;

public:
    // Globals
    int &simCycles;
    IntegrationMethod &integrator;
    unsigned int &assignment;
    int &targetStride;

    scalar *&target;
    scalar *&output;

    scalar *&clusters;
    scalar *&clusterCurrent; //!< Layout: [stimIdx][clusterIdx]
    scalar *&clusterPrimitives; //!< Layout: [stimIdx][paramIdx][secIdx]. Get the section stride (padded nSecs) from the call to pullPrimitives().
    iObservations *&clusterObs; //!< Layout: [stimIdx][clusterIdx]

    unsigned int assignment_base = 0;
};

#endif // UNIVERSALLIBRARY_H
