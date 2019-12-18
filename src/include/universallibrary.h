/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


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
#define ASSIGNMENT_SUMMARY_COMPARE_PREVTHREAD   static_cast<unsigned int>(0x3) /* compare against preceding lane in warp, wrapping around */
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

// Bit 9: Don't integrate past settling
#define ASSIGNMENT_SETTLE_ONLY                  static_cast<unsigned int>(0x3 << 8) /* Ignore stimulation, settle for iSettleDuration only; implies MAINTAIN_STATE and (effectively) SUMMARY_PERSIST */

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
#define ASSIGNMENT_SUMMARY_PERSIST              static_cast<unsigned int>(0x1 << 13) /* Maintain and accumulate summary value across calls. Do not use with SUMMARY_AVERAGE */

// Bits 14-16: Singular stim/rund/target
#define ASSIGNMENT_SINGULAR_STIM                 static_cast<unsigned int>(0x1 << 14) /* Load stim/obs from singular __constant__ var */
#define ASSIGNMENT_SINGULAR_RUND                 static_cast<unsigned int>(0x1 << 15) /* Load rundata from singular __constant__ var */
#define ASSIGNMENT_SINGULAR_TARGET                 static_cast<unsigned int>(0x1 << 16) /* Load target from singular __constant__ var */

// Bits 17-18: Noisy integration during observed periods
#define ASSIGNMENT_NOISY_OBSERVATION            static_cast<unsigned int>(0x1 << 17) /* Add white or O-U noise during observed periods */
#define ASSIGNMENT_NOISY_CHANNELS               static_cast<unsigned int>(0x3 << 17) /* Add gating noise during observed periods; implies NOISY_OBSERVATION */

// Bits 19-26: Current and pattern clamp
#define ASSIGNMENT_CURRENTCLAMP                 static_cast<unsigned int>(0x1 << 19) /* Run in pure current clamp; reports voltage instead of current */
#define ASSIGNMENT_PATTERNCLAMP                 static_cast<unsigned int>(0x1 << 20) /* Run in pattern clamp against the specified target; reports voltage or pin current */
#define ASSIGNMENT_PC_REPORT_PIN                static_cast<unsigned int>(0x1 << 21) /* Report pattern clamp pin current, rather than voltage */

#define ASSIGNMENT_PC_PIN__SHIFT 22
#define ASSIGNMENT_PC_PIN_32    static_cast<unsigned int>(0x1f << ASSIGNMENT_PC_PIN__SHIFT) /* Pin full warp to lane 0 voltage, rather than lib.target; lane 0 not pinned */
#define ASSIGNMENT_PC_PIN_16    static_cast<unsigned int>(0x0f << ASSIGNMENT_PC_PIN__SHIFT) /* Pin half warp to local lane 0 voltage, rather than lib.target; lane 0 not pinned */
#define ASSIGNMENT_PC_PIN_8     static_cast<unsigned int>(0x07 << ASSIGNMENT_PC_PIN__SHIFT) /* Pin 7 following threads to local lane 0 voltage, rather than lib.target; lane 0 not pinned */
#define ASSIGNMENT_PC_PIN_4     static_cast<unsigned int>(0x03 << ASSIGNMENT_PC_PIN__SHIFT) /* Pin 3 following threads to local lane 0 voltage, rather than lib.target; lane 0 not pinned */
#define ASSIGNMENT_PC_PIN_2     static_cast<unsigned int>(0x01 << ASSIGNMENT_PC_PIN__SHIFT) /* Pin odd threads to their next lower even thread's voltage, rather than lib.target; even threads not pinned */

// Bit 27: Clamp gain decay
#define ASSIGNMENT_CLAMP_GAIN_DECAY             static_cast<unsigned int>(0x1 << 27) /* Decay clamp gain to zero over the duration of PC settling */

// Bit 28: Override iSettleDuration
#define ASSIGNMENT_NO_SETTLING                  static_cast<unsigned int>(0x1 << 28) /* No settling regardless of iSettleDuration value */

// Bit 29: Multiplex subset of models
// Models (and RunData, if singular) are loaded as id % cl_blocksize; stim and target (if singular) are loaded as id / cl_blocksize.
#define ASSIGNMENT_SUBSET_MUX                   static_cast<unsigned int>(0x1 << 29) /* Multiplex over the subset [0,cl_blocksize] */

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
        scalar *noiseExp;
        scalar *noiseAmplitude;
        int *summaryOffset;
        int *cl_blocksize;

        std::function<void(void*, void*, size_t, int)> pushV;
        std::function<void(void*, void*, size_t, int)> pullV;
        void (*run)(int, unsigned int);
        void (*reset)(void);

        void (*sync)(unsigned int);
        void (*resetEvents)(unsigned int);
        unsigned int (*recordEvent)(unsigned int);
        void (*waitEvent)(unsigned int, unsigned int);

        DAQ *(*createSim)(int simNo, Session&, const Settings&, bool);
        void (*destroySim)(DAQ *);

        void (*resizeTarget)(size_t);
        void(*pushTarget)(int, size_t, size_t);
        scalar **target;

        void (*resizeOutput)(size_t);
        void(*pullOutput)(int);
        scalar **output;

        void (*resizeSummary)(size_t);
        void(*pullSummary)(int, size_t, size_t);
        scalar **summary;

        void (*profile)(int nSamples, const std::vector<AdjustableParam> &params, size_t targetParam, std::vector<scalar> weight,
                        double &rho_weighted, double &rho_unweighted, double &rho_target_only,
                        double &grad_weighted, double &grad_unweighted, double &grad_target_only,
                        std::vector<double> &invariants,
                        bool VC);

        void (*cluster)(int trajLen, int nTraj, int duration, int secLen, scalar dotp_threshold, int minClusterLen,
                        std::vector<double> deltabar, const MetaModel &, bool VC, bool pull);
        void (*pullClusters)(int nStims, bool includeCurrent);
        int (*pullPrimitives)(int nStims, int duration, int secLen);
        std::vector<double> (*find_deltabar)(int trajLen, int nTraj, const MetaModel &);
        scalar **clusters;
        scalar **clusterCurrent;
        scalar **clusterPrimitives;
        iObservations **clusterObs;

        void (*bubble)(int trajLen, int nTraj, int duration, int secLen, std::vector<double> deltabar, const MetaModel &, bool VC, bool pull_results);
        void (*pullBubbles)(int nStims, bool includeCurrent);
        Bubble **bubbles;

        void (*observe_no_steps)(int blankCycles);
        void (*genRandom)(unsigned int n, scalar mean, scalar sd, unsigned long long seed);
        void (*get_posthoc_deviations)(int trajLen, int nTraj, unsigned int nStims, std::vector<double> deltabar, const MetaModel &, bool VC);

        std::vector<scalar> (*principal_components)(int L, scalar *d_X, int ldX, int n, int p);
        void (*copy_param)(int i, scalar *d_v);
        scalar **PCA_TL;

        double (*get_mean_distance)(const AdjustableParam &);

        scalar *(*cl_compare_to_target)(int nSamples, ClosedLoopData d, double dt, bool reset_summary, scalar *target);
        std::vector<std::tuple<scalar, scalar, scalar, scalar>> (*cl_compare_models)(int nStims, unsigned int nSamples, ClosedLoopData d, double dt);
        scalar (*cl_dmap_hi)(scalar lo, scalar step);
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

    void push();
    void pull();
    template <typename T>
    inline void push(TypedVariable<T> &var, int streamId = -1) { //!< Synchronous, unless streamId >= 0
        pointers.pushV(var.v,
                       var.singular ? var.singular_v : var.d_v,
                       sizeof(T) * (var.singular ? 1 : NMODELS),
                       streamId);
    }
    template <typename T>
    inline void pull(TypedVariable<T> &var, int streamId = -1) { //!< Synchronous, unless streamId >= 0
        pointers.pullV(var.v,
                       var.singular ? var.singular_v : var.d_v,
                       sizeof(T) * (var.singular ? 1 : NMODELS),
                       streamId);
    }
    inline void run(int iT = 0, unsigned int streamId = 0) { pointers.run(iT, streamId); }
    inline void reset() { pointers.reset(); }
    void pushParams();

    inline void sync(unsigned int streamId = 0) { pointers.sync(streamId); }
    inline void resetEvents(unsigned int nExpected = 0) { pointers.resetEvents(nExpected); } //!< Resets the consecutive events counter and prepares nExpected events for immediate use.
    inline unsigned int recordEvent(unsigned int streamId = 0) { return pointers.recordEvent(streamId); } //!< Records a new event on the given stream and returns its handle.
    inline void waitEvent(unsigned int eventHandle, unsigned int streamId = 0) { pointers.waitEvent(eventHandle, streamId); }

    /// Allocates space for nTraces traces of length nSamples and sets targetStride = nTraces
    void resizeTarget(size_t nTraces, size_t nSamples);
    inline void pushTarget(int streamId = -1, size_t nSamples = 0, size_t offset = 0) { pointers.pushTarget(streamId, nSamples, offset); } //!< Synchronous, unless streamId >= 0

    /// Allocates space for NMODELS traces of length nSamples
    void resizeOutput(size_t nSamples);
    inline void pullOutput(int streamId = -1) { pointers.pullOutput(streamId); } //!< Synchronous, unless streamId >= 0

    /// Allocates space for n summaries per model (initial size is 1). Use summaryOffset (0,1,2, not 0,NMODELS,2*NMODELS) to separate summary outputs eg across streams
    inline void resizeSummary(size_t n) { pointers.resizeSummary(n * NMODELS); }
    inline void pullSummary(int streamId = -1, size_t offset = 0) { pointers.pullSummary(streamId, NMODELS, offset*NMODELS); } //!< Synchronous, unless streamId >= 0

    void setRundata(size_t modelIndex, const RunData &rund);

    /// post-run() workhorse for SamplingProfiler
    /// Expects assignments TIMESERIES_COMPARE_NONE | TIMESERIES_COMPACT
    /// @a invariants should be an empty array when a new parameter set is presented. It will be populated with intermediate values
    /// related to unweighted and target-only distance measures, which can be reused on subsequent calls for the same parameter set
    /// for a reduction in computational load.
    /// @a VC==false additionally expects PATTERNCLAMP | PC_PIN_32. Additionally, cold-starting and switching between different invariant sets is not supported.
    inline void profile(int nSamples, size_t targetParam, std::vector<scalar> weight,
                        double &rho_weighted, double &rho_unweighted, double &rho_target_only,
                        double &grad_weighted, double &grad_unweighted, double &grad_target_only,
                        std::vector<double> &invariants, bool VC) {
        pointers.profile(nSamples, adjustableParams, targetParam, weight,
                         rho_weighted, rho_unweighted, rho_target_only,
                         grad_weighted, grad_unweighted, grad_target_only,
                         invariants, VC);
    }

    /// post-run() workhorse for cluster-based wavegen
    /// Expects assignment TIMESERIES_COMPARE_NONE with models sequentially detuned along and across their ee trajectories, as well as
    /// an appropriate individual obs in each stim's first trajectory starting point model
    /// @return nPartitions, as required for pullClusters() and further processing of clusterMasks: The number of partitions of 32 secs each
    inline void cluster(int trajLen, /* length of EE trajectory (power of 2, <=32) */
                        int nTraj, /* Number of EE trajectories */
                        int duration, /* nSamples of each trace, cf. resizeOutput() */
                        int secLen, /* sampling resolution (in ticks) */
                        scalar dotp_threshold, /* Minimum scalar product for two sections to be considered to belong to the same cluster */
                        int minClusterLen, /* Minimum duration for a cluster to be considered valid (in ticks) */
                        std::vector<double> deltabar, /* RMS current deviation, cf. find_deltabar() */
                        bool VC, /* Voltage clamp / pattern clamp switch */
                        bool pull_results) /* True to copy results immediately; false to defer to pullClusters(), allowing time for CPU processing */ {
        pointers.cluster(trajLen, nTraj, duration, secLen, dotp_threshold, minClusterLen, deltabar, model, VC, pull_results);
    }
    /// Copy cluster() results to clusters, clusterMasks and clusterCurrent.
    /// Note, nStims = NMODELS/(trajLen*nTraj)
    inline void pullClusters(int nStims, bool includeCurrent) { pointers.pullClusters(nStims, includeCurrent); }
    /// Copy cluster() intermediates (section primitives) to clusterPrimitives, and return the section stride
    inline int pullPrimitives(int nStims, int duration, int secLen) { return pointers.pullPrimitives(nStims, duration, secLen); }

    /// post-run() workhorse for bubble-based wavegen
    /// Expects assignment TIMESERIES_COMPARE_NONE with models sequentially detuned along and across their ee trajectories, as well as
    /// an appropriate individual obs in each stim's first trajectory starting point model
    inline void bubble(int trajLen, /* length of EE trajectory (power of 2, <=32) */
                       int nTraj, /* Number of EE trajectories */
                       int duration,
                       int secLen,
                       std::vector<double> deltabar,
                       bool VC,
                       bool pull_results) {
        pointers.bubble(trajLen, nTraj, duration, secLen, deltabar, model, VC, pull_results);
    }
    /// Copy bubble() results to bubbles, clusters, and clusterCurrent.
    inline void pullBubbles(int nStims, bool includeCurrent) { pointers.pullBubbles(nStims, includeCurrent); }

    /// post-run() calculation of RMS current deviation from each detuned parameter. Reports the RMSD per tick, per single detuning,
    /// as required by cluster().
    /// VC: Expects assignment TIMESERIES_COMPARE_PREVTHREAD with models sequentially detuned along and across their ee trajectories, as well as
    /// an appropriate individual obs in each stim's first trajectory starting point model
    /// CC: Use TIMESERIES_COMPARE_NONE with one-at-a-time detuning
    inline std::vector<double> find_deltabar(int trajLen, int nTraj) { return pointers.find_deltabar(trajLen, nTraj, model); }

    /// Utility call to add full-stim, step-blanked observation windows to the (model-individual) stims residing on the GPU
    inline void observe_no_steps(int blankCycles) { pointers.observe_no_steps(blankCycles); }

    /// Utility call to generate a lot of normally distributed scalars in device memory for use with ASSIGNMENT_NOISY_OBSERVATION
    /// The RNG is reset to a new seed only if seed != 0; otherwise, the existing RNG state is used.
    inline void generate_random_samples(unsigned int n, scalar mean, scalar sd, unsigned long long seed = 0) { pointers.genRandom(n, mean, sd, seed); }

    /// post-run() workhorse to get deviations and currents for predefined stims.
    /// Pulls results to clusters as [stimIdx][paramIdx], and to clusterCurrent as [stimIdx], both with stride nStims.
    inline void get_posthoc_deviations(int trajLen, int nTraj, unsigned int nStims, std::vector<double> deltabar, bool VC) {
        pointers.get_posthoc_deviations(trajLen, nTraj, nStims, deltabar, model, VC);
    }

    /// Utility call to perform PCA on adjustableParams.
    /// Stores a projection to the first @a L PCs in PCA_TL, and returns (all) singular values.
    /// Use @arg nIgnored to exclude the given number of rows at the end of the model population from the PCA.
    inline std::vector<scalar> get_params_principal_components(int L, int nIgnored = 0) {
        for ( size_t i = 0; i < adjustableParams.size(); i++ )
            pointers.copy_param(i, adjustableParams[i].d_v);
        auto ret = pointers.principal_components(L, nullptr, NMODELS, NMODELS-nIgnored, adjustableParams.size());
        PCA_TL_size = NMODELS-nIgnored;
        return ret;
    }

    /// Utility call to compute mean distance within the population along the given parameter axis
    inline double get_mean_distance(size_t targetParam) { return pointers.get_mean_distance(adjustableParams.at(targetParam)); }

    /// Compares model timeseries to the timeseries provided in lib.target (host-side; respects targetOffset[0]), reporting the error in lib.summary (device-side).
    /// Returns a pointer to the partial (trace, sdf, and dmaps) errors across all models, laid out as [errtype*NMODELS + modelIdx].
    inline scalar *cl_compare_to_target(int nSamples, ClosedLoopData d, double dt, bool reset_summary) {
        return pointers.cl_compare_to_target(nSamples, d, dt, reset_summary, target + targetOffset[0]);
    }

    /// Compares model timeseries for closed-loop stimulus selection after multiplexed stim evaluation.
    inline std::vector<std::tuple<scalar, scalar, scalar, scalar>> cl_compare_models(int nStims, unsigned int nSamples, ClosedLoopData d, double dt) {
        return pointers.cl_compare_models(nStims, nSamples, d, dt);
    }

    /// Utility to find dmap upper limit
    inline scalar cl_dmap_hi(scalar lo, scalar step) { return pointers.cl_dmap_hi(lo, step); }

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
    Bubble *dummyBubblePtr = nullptr;
    scalar dummyScalar;

    scalar *lightParams, *lightSummary;

public:
    // Globals
    int &simCycles;
    IntegrationMethod &integrator;
    unsigned int &assignment;
    int &targetStride;
    scalar &noiseExp;
    scalar &noiseAmplitude;
    int &summaryOffset;
    int &cl_blocksize;

    scalar *&target;
    scalar *&output;
    scalar *&summary;

    scalar *&clusters; //!< Layout: [stimIdx][clusterIdx][paramIdx] (after clustering) or [stimIdx][targetParamIdx][paramIdx] (after bubbling)
    scalar *&clusterCurrent; //!< Layout: [stimIdx][clusterIdx] (after clustering) or [stimIdx][targetParamIdx] (after bubbling)
    scalar *&clusterPrimitives; //!< Layout: [stimIdx][paramIdx][secIdx]. Get the section stride (padded nSecs) from the call to pullPrimitives().
    iObservations *&clusterObs; //!< Layout: [stimIdx][clusterIdx]

    Bubble *&bubbles; //!< Layout: [stimIdx][targetParamIdx]

    scalar *&PCA_TL;
    int PCA_TL_size = 0; // Number of projected points (and leading dimension of PCA_TL)

    unsigned int assignment_base = 0;
};

#endif // UNIVERSALLIBRARY_H
