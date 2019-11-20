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


#include "universallibrary.h"
#include "modelSpec.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include "global.h"
#include "cuda_helper.h"
#include <dlfcn.h>
#include "project.h"

#define SUFFIX "UNI"

static UniversalLibrary *_this;
static void redirect(NNmodel &n) { _this->GeNN_modelDefinition(n); }

UniversalLibrary::UniversalLibrary(Project & p, bool compile, bool light) :
    project(p),
    model(project.model()),
    NMODELS(p.expNumCandidates()),
    stateVariables(model.stateVariables),
    adjustableParams(model.adjustableParams),

    stim("stim", "iStimulation"),
    obs("obs", "iObservations"),
    clampGain("clampGain"),
    accessResistance("accessResistance"),
    iSettleDuration("iSettleDuration", "int"),
    Imax("Imax"),
    dt("dt"),
    targetOffset("targetOffset", "size_t"),

    lib(light ? nullptr : compile ? compile_and_load() : load()),
    populate(light ? nullptr : (decltype(populate))dlsym(lib, "populate")),
    pointers(light ? Pointers() : populate(*this)),
    isLight(light),

    simCycles(light ? dummyInt : *pointers.simCycles),
    integrator(light ? dummyIntegrator : *pointers.integrator),
    assignment(light ? dummyUInt : *pointers.assignment),
    targetStride(light ? dummyInt : *pointers.targetStride),
    noiseExp(light ? dummyScalar : *pointers.noiseExp),
    noiseAmplitude(light ? dummyScalar : *pointers.noiseAmplitude),
    summaryOffset(light ? dummyInt : *pointers.summaryOffset),
    cl_blocksize(light ? dummyInt : *pointers.cl_blocksize),

    target(light ? dummyScalarPtr : *pointers.target),
    output(light ? dummyScalarPtr : *pointers.output),
    summary(light ? dummyScalarPtr : *pointers.summary),

    clusters(light ? dummyScalarPtr : *pointers.clusters),
    clusterCurrent(light ? dummyScalarPtr : *pointers.clusterCurrent),
    clusterPrimitives(light ? dummyScalarPtr : *pointers.clusterPrimitives),
    clusterObs(light ? dummyObsPtr : *pointers.clusterObs),

    bubbles(light ? dummyBubblePtr : *pointers.bubbles),

    PCA_TL(light ? dummyScalarPtr : *pointers.PCA_TL)
{
}

UniversalLibrary::~UniversalLibrary()
{
    if ( isLight )
        return;
    void (*libExit)(Pointers&);
    if ( (libExit = (decltype(libExit))dlsym(lib, "libExit")) )
        libExit(pointers);
    if ( !--MetaModel::numLibs ) {
        void (*resetDevice)();
        if ( (resetDevice = (decltype(resetDevice))dlsym(lib, "resetDevice")) )
            resetDevice();
    }
    dlclose(lib);
}

void *UniversalLibrary::compile_and_load()
{
    std::string directory = project.dir().toStdString();

    // Generate code
    _this = this;
    MetaModel::modelDef = redirect;

    model.isUniversalLib = true;
    model.save_state_condition = "$(assignment) & ASSIGNMENT_MAINTAIN_STATE";
    model.save_selectively = true;
    model.save_selection = {};
    model.singular_stim_vars = {&stim, &obs};
    model.singular_clamp_vars = {&clampGain, &accessResistance};
    model.singular_rund_vars = {&iSettleDuration, &Imax, &dt};
    model.singular_target_vars = {&targetOffset};

    generateCode(directory, model);

    model.isUniversalLib = false;
    model.save_state_condition = "";
    model.save_selectively = false;

    // Compile
    std::string dir = directory + "/" + model.name() + "_CODE";
    std::ofstream makefile(dir + "/Makefile", std::ios_base::app);
    makefile << endl;
    makefile << "runner.so: runner.o" << endl;
    makefile << "\t$(CXX) -o $@ $< -shared " << CULIBS << endl;
    std::stringstream cmd;
    cmd << "cd " << dir << " && make runner.so";
    if ( system(cmd.str().c_str()) )
        throw std::runtime_error("Code compile failed.");

    // Load library
    return load();
}

void *UniversalLibrary::load()
{
    std::string libfile = project.dir().toStdString() + "/" + model.name() + "_CODE/runner.so";
    dlerror();
    void *libp;
    if ( ! (libp = dlopen(libfile.c_str(), RTLD_NOW)) )
        throw std::runtime_error(std::string("Library load failed: ") + dlerror());

    ++MetaModel::numLibs;
    return libp;
}

void UniversalLibrary::GeNN_modelDefinition(NNmodel &nn)
{
    std::vector<double> fixedParamIni, variableIni;
    neuronModel n = model.generate(nn, fixedParamIni, variableIni);

    std::vector<Variable> globals = {
        Variable("simCycles", "", "int"),
        Variable("integrator", "", "IntegrationMethod"),
        Variable("assignment", "", "unsigned int"),
        Variable("targetStride", "", "int"),
        Variable("noiseExp"),
        Variable("noiseAmplitude"),
        Variable("summaryOffset", "", "int"),
        Variable("cl_blocksize", "", "int")
    };
    for ( Variable &p : globals ) {
        n.extraGlobalNeuronKernelParameters.push_back(p.name);
        n.extraGlobalNeuronKernelParameterTypes.push_back(p.type);
    }

    n.varNames.push_back(stim.name);
        n.varTypes.push_back(stim.type);
        variableIni.push_back(0.);
    n.varNames.push_back(obs.name);
        n.varTypes.push_back(obs.type);
        variableIni.push_back(0.);
    n.varNames.push_back(clampGain.name);
        n.varTypes.push_back(clampGain.type);
        variableIni.push_back(0.);
    n.varNames.push_back(accessResistance.name);
        n.varTypes.push_back(accessResistance.type);
        variableIni.push_back(0.);
    n.varNames.push_back(iSettleDuration.name);
        n.varTypes.push_back(iSettleDuration.type);
        variableIni.push_back(0.);
    n.varNames.push_back(Imax.name);
        n.varTypes.push_back(Imax.type);
        variableIni.push_back(0.);
    n.varNames.push_back(dt.name);
        n.varTypes.push_back(dt.type);
        variableIni.push_back(0.);
    n.varNames.push_back(targetOffset.name);
        n.varTypes.push_back(targetOffset.type);
        variableIni.push_back(0.);

    n.simCode = simCode();
    n.supportCode = supportCode(globals);

    int numModels = nModels.size();
    nModels.push_back(n);
    nn.setName(model.name());
    nn.addNeuronPopulation(SUFFIX, NMODELS, numModels, fixedParamIni, variableIni);

    nn.finalize();
}

std::string UniversalLibrary::simCode()
{
    std::stringstream ss;
    ss << endl << "#ifndef _" << model.name() << "_neuronFnct_cc" << endl;

    ss << R"EOF(
// Ensure inclusion of unreferenced: $(cl_blocksize)

scalar mdt = $(dt) / $(simCycles);
scalar hP = 0.;

int tStep = 0;
int nSamples = 0;
int nextObs = 0;
scalar t = iT * $(dt);
double summary = 0;

scalar noiseI[3];
if ( $(assignment) & ASSIGNMENT_NOISY_OBSERVATION )
    noiseI[0] = dd_random[id];

scalar V = state.V;
if ( $(assignment) & ASSIGNMENT_CURRENTCLAMP )
    clamp.clamp = ClampParameters::ClampType::Current;
else if ( $(assignment) & ASSIGNMENT_PATTERNCLAMP ) {
    if ( ($(assignment) & ASSIGNMENT_PC_PIN_2) && (threadIdx.x & 31 & ($(assignment) >> ASSIGNMENT_PC_PIN__SHIFT)) == 0 )
        clamp.clamp = ClampParameters::ClampType::Current;
    else
        clamp.clamp = ClampParameters::ClampType::Pattern;
}

// Settle
if ( iT == 0 && $(iSettleDuration) > 0 ) {
    if ( $(assignment) & ASSIGNMENT_PATTERNCLAMP ) {
        clamp.dIClamp = 0.0;
        clamp.IClamp0 = $(stim).baseV;
        for ( iT = -$(iSettleDuration); iT < 0; iT++ ) {
            scalar dV;
            if ( $(assignment) & ASSIGNMENT_PC_PIN_2 ) {
                unsigned short target = threadIdx.x & 31 & ~($(assignment) >> ASSIGNMENT_PC_PIN__SHIFT);
                dV = V - __shfl_sync(0xffffffff, state.V, target);
                V = __shfl_sync(0xffffffff, state.V, target);
            } else {
                V = dd_target[$(targetOffset) + $(targetStride)*(iT+$(iSettleDuration))];
                dV = dd_target[$(targetOffset) + $(targetStride)*(iT+1+$(iSettleDuration))] - V;
            }
            clamp.dVClamp = dV / $(dt);
            clamp.VClamp0 = V - iT * dV;
            integrate(1, $(integrator), $(simCycles), iT * $(dt), $(dt), mdt, hP, state, params, clamp);
        }
    } else if ( $(assignment) & ASSIGNMENT_CURRENTCLAMP ) {
        clamp.dIClamp = 0.0;
        clamp.IClamp0 = $(stim).baseV;
        integrate($(iSettleDuration), $(integrator), $(simCycles), -$(iSettleDuration)*$(dt), $(dt), mdt, hP, state, params, clamp);
    } else {
        clamp.dVClamp = 0.0;
        clamp.VClamp0 = $(stim).baseV;
        integrate($(iSettleDuration), $(integrator), $(simCycles), -$(iSettleDuration)*$(dt), $(dt), mdt, hP, state, params, clamp);
    }
} else if ( iT > 0 ) {
    while ( nextObs < iObservations::maxObs && iT > $(obs).stop[nextObs] )
        ++nextObs;
}

while ( !($(assignment)&ASSIGNMENT_SETTLE_ONLY)
        && iT < $(stim).duration
        && nextObs < iObservations::maxObs ) {

    // Integrate unobserved
    while ( iT < $(obs).start[nextObs] ) {
        if ( $(assignment) & (ASSIGNMENT_CURRENTCLAMP | ASSIGNMENT_PATTERNCLAMP) )
            getiCommandSegment($(stim), iT, $(stim).duration - iT, $(dt), clamp.IClamp0, clamp.dIClamp, tStep);
        else
            getiCommandSegment($(stim), iT, $(stim).duration - iT, $(dt), clamp.VClamp0, clamp.dVClamp, tStep);
        tStep = min(tStep, $(obs).start[nextObs] - iT);

        if ( $(assignment) & ASSIGNMENT_PATTERNCLAMP ) {
            while ( tStep ) {
                scalar dV;
                if ( $(assignment) & ASSIGNMENT_PC_PIN_2 ) {
                    unsigned short target = threadIdx.x & 31 & ~($(assignment) >> ASSIGNMENT_PC_PIN__SHIFT);
                    dV = V - __shfl_sync(0xffffffff, state.V, target);
                    V = __shfl_sync(0xffffffff, state.V, target);
                } else {
                    V = dd_target[$(targetOffset) + $(targetStride)*iT];
                    dV = dd_target[$(targetOffset) + $(targetStride)*(iT+1)] - V;
                }
                clamp.dVClamp = dV / $(dt);
                clamp.VClamp0 = V - iT * dV;

                integrate(1, $(integrator), $(simCycles), iT * $(dt), $(dt), mdt, hP, state, params, clamp);
                --tStep;
                ++iT;
            }
        } else {
            integrate(tStep, $(integrator), $(simCycles), t, $(dt), mdt, hP, state, params, clamp);
            iT += tStep;
        }
        t = iT * $(dt);

        if ( ($(assignment) & ASSIGNMENT_REPORT_TIMESERIES)
          && !($(assignment) & ASSIGNMENT_TIMESERIES_COMPACT)
          && ($(assignment) & ASSIGNMENT_TIMESERIES_ZERO_UNTOUCHED_SAMPLES) )
            for ( int i = -tStep; i < 0; i++ )
                dd_timeseries[id + NMODELS*(iT+i)] = 0.;
    }

    // Process results while integrating stepwise with $(dt)
    while ( iT < $(obs).stop[nextObs] ) {
        if ( $(assignment) & (ASSIGNMENT_CURRENTCLAMP | ASSIGNMENT_PATTERNCLAMP) )
            getiCommandSegment($(stim), iT, $(stim).duration - iT, $(dt), clamp.IClamp0, clamp.dIClamp, tStep);
        else
            getiCommandSegment($(stim), iT, $(stim).duration - iT, $(dt), clamp.VClamp0, clamp.dVClamp, tStep);
        tStep = min(tStep, $(obs).stop[nextObs] - iT);
        while ( tStep ) {
            // Process results
            scalar diff, value;
            if ( $(assignment) & ASSIGNMENT_CURRENTCLAMP ) {
                value = state.V;
            } else if ( $(assignment) & ASSIGNMENT_PATTERNCLAMP ) {
                scalar dV;
                if ( $(assignment) & ASSIGNMENT_PC_PIN_2 ) {
                    unsigned short target = threadIdx.x & 31 & ~($(assignment) >> ASSIGNMENT_PC_PIN__SHIFT);
                    dV = V - __shfl_sync(0xffffffff, state.V, target);
                    V = __shfl_sync(0xffffffff, state.V, target);
                } else {
                    V = dd_target[$(targetOffset) + $(targetStride)*iT];
                    dV = dd_target[$(targetOffset) + $(targetStride)*(iT+1)] - V;
                }
                clamp.dVClamp = dV / $(dt);
                clamp.VClamp0 = V - iT * dV;
                value = ($(assignment) & ASSIGNMENT_PC_REPORT_PIN) ? clamp.getPinCurrent(t, state.V) : state.V;
            } else {
                value = clip(clamp.getCurrent(t, state.V), $(Imax));
            }

            // Report time series
            if ( $(assignment) & ASSIGNMENT_REPORT_TIMESERIES ) {
                scalar *out = dd_timeseries;
                if ( $(assignment) & ASSIGNMENT_TIMESERIES_COMPACT )
                    out += id + NMODELS*nSamples;
                else
                    out += id + NMODELS*iT;

                switch ( $(assignment) & ASSIGNMENT_TIMESERIES_COMPARE_MASK ) {
                case ASSIGNMENT_TIMESERIES_COMPARE_TARGET:
                    diff = dd_target[$(targetOffset) + $(targetStride)*iT] - value;
                    break;
                case ASSIGNMENT_TIMESERIES_COMPARE_LANE0:
                    diff = __shfl_sync(0xffffffff, value, 0) - value;
                    if ( (threadIdx.x & 31) == 0 )
                        diff = value;
                    break;
                case ASSIGNMENT_TIMESERIES_COMPARE_PREVTHREAD:
                    diff = __shfl_up_sync(0xffffffff, value, 1) - value;
                    if ( (threadIdx.x & 31) == 0 )
                        diff = value;
                    break;
                case ASSIGNMENT_TIMESERIES_COMPARE_NONE:
                default:
                    diff = value;
                    break;
                }

                *out = ($(assignment) & ASSIGNMENT_TIMESERIES_ABS) ? fabs(diff) : diff;
            }

            // Accumulate summary
            if ( $(assignment) & ASSIGNMENT_REPORT_SUMMARY ) {
                switch ( $(assignment) & ASSIGNMENT_SUMMARY_COMPARE_MASK ) {
                case ASSIGNMENT_SUMMARY_COMPARE_TARGET:
                    diff = dd_target[$(targetOffset) + $(targetStride)*iT] - value;
                    break;
                case ASSIGNMENT_SUMMARY_COMPARE_LANE0:
                    diff = __shfl_sync(0xffffffff, value, 0) - value;
                    if ( (threadIdx.x & 31) == 0 )
                        diff = value;
                    break;
                case ASSIGNMENT_SUMMARY_COMPARE_PREVTHREAD:
                    diff = __shfl_sync(0xffffffff, value, (threadIdx.x&31)-1) - value;
                    break;
                case ASSIGNMENT_SUMMARY_COMPARE_NONE:
                default:
                    diff = value;
                    break;
                }

                summary += ($(assignment) & ASSIGNMENT_SUMMARY_SQUARED) ? (double(diff)*diff) : fabs(diff);
            }

            if ( ($(assignment) & ASSIGNMENT_REPORT_FIRST_SPIKE) && value > -10.0 && summary == 0 ) {
                summary = t - dd_target[0];
                if ( $(assignment) & ASSIGNMENT_SUMMARY_SQUARED )
                    summary *= summary;
                iT = $(obs).stop[nextObs];
                break;
            }

            // Integrate forward by one $(dt)
            if ( $(assignment) & ASSIGNMENT_NOISY_OBSERVATION ) {
                scalar *offset_rand = dd_random + nSamples * 2*$(simCycles) * NMODELS + id;
                scalar A = $(noiseAmplitude);
                if ( ($(assignment) & ASSIGNMENT_NOISY_CHANNELS) == ASSIGNMENT_NOISY_CHANNELS )
                    A = sqrt(A*A + state.state__variance(params));
                for ( int mt = 0; mt < $(simCycles); mt++ ) {
                    noiseI[1] = noiseI[0] * $(noiseExp) + offset_rand[(2*mt+1) * NMODELS] * A;
                    noiseI[2] = noiseI[1] * $(noiseExp) + offset_rand[(2*mt+2) * NMODELS] * A;
                    RK4(t + mt*mdt, mdt, state, params, clamp, noiseI);
                    noiseI[0] = noiseI[2];
                }
            } else {
                integrate(1, $(integrator), $(simCycles), t, $(dt), mdt, hP, state, params, clamp);
            }
            ++nSamples;
            --tStep;
            ++iT;
            t = iT * $(dt);
        }
    }
    ++nextObs;
}

if ( ($(assignment) & ASSIGNMENT_REPORT_FIRST_SPIKE) && summary == 0 && t == iT*$(dt) ) { // Note: Test for no spike requires break in spike detection, otherwise t is updated correctly
    summary = iT*$(dt) - dd_target[0];
    if ( $(assignment) & ASSIGNMENT_SUMMARY_SQUARED )
        summary *= summary;
}

if ( $(assignment) & (ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_REPORT_FIRST_SPIKE) && !($(assignment) & ASSIGNMENT_SETTLE_ONLY) ) {
    if ( $(assignment) & ASSIGNMENT_SUMMARY_AVERAGE )
        summary /= nSamples;
    if ( $(assignment) & ASSIGNMENT_SUMMARY_PERSIST )
        summary += dd_summary[$(summaryOffset)*NMODELS + id];
    dd_summary[$(summaryOffset)*NMODELS + id] = summary;
}

#endif
)EOF";

    return ss.str();
}

std::string UniversalLibrary::supportCode(const std::vector<Variable> &globals)
{
    std::stringstream ss;

    ss << "} // break namespace for STL includes:" << endl;

    ss << "#include \"definitions.h\"" << endl;
    ss << "#include \"universallibrary.h\"" << endl;
    ss << "#include \"../core/supportcode.cpp\"" << endl;
    ss << endl;
    ss << model.supportCode() << endl;
    ss << project.simulatorCode() << endl;
    ss << endl;
    ss << "#define NMODELS " << NMODELS << endl;
    ss << "#define NPARAMS " << adjustableParams.size() << endl;
    ss << "#define MAXCLUSTERS " << maxClusters << endl;
    ss << "#include \"../cuda/universallibrary.cu\"" << endl;
    ss << endl;

    ss << "void runStreamed(int iT, unsigned int streamId) {" << endl
       << "    calcNeurons <<< nGrid_calcNeurons, nThreads_calcNeurons, 0, getLibStream(streamId) >>> (";
    for ( const Variable &p : globals )
        ss << p.name << SUFFIX << ", ";
    ss << "iT);\n"
       << "}" << endl;

    ss << "extern \"C\" UniversalLibrary::Pointers populate(UniversalLibrary &lib) {" << endl;
    ss << "    UniversalLibrary::Pointers pointers;" << endl;
    ss << "    libInit(lib, pointers);" << endl;
    ss << "    resizeSummary(NMODELS);" << endl;

    int i = 0;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    lib.stateVariables[" << i << "].v = " << v.name << SUFFIX << ";" << endl;
        ss << "    lib.stateVariables[" << i << "].d_v = d_" << v.name << SUFFIX << ";" << endl;
        ++i;
    }
    i = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    lib.adjustableParams[" << i << "].v = " << p.name << SUFFIX << ";" << endl;
        ss << "    lib.adjustableParams[" << i << "].d_v = d_" << p.name << SUFFIX << ";" << endl;
        ++i;
    }
    ss << endl;
    for ( const Variable &p : globals ) {
        ss << "    pointers." << p.name << " =& " << p.name << SUFFIX << ";" << endl;
    }
    ss << "    *pointers.cl_blocksize = NMODELS;" << endl;

    ss << "    lib.stim.v = stim" << SUFFIX << ";" << endl;
    ss << "    lib.stim.d_v = d_stim" << SUFFIX << ";" << endl;
    ss << "    lib.obs.v = obs" << SUFFIX << ";" << endl;
    ss << "    lib.obs.d_v = d_obs" << SUFFIX << ";" << endl;
    ss << "    lib.clampGain.v = clampGain" << SUFFIX << ";" << endl;
    ss << "    lib.clampGain.d_v = d_clampGain" << SUFFIX << ";" << endl;
    ss << "    lib.accessResistance.v = accessResistance" << SUFFIX << ";" << endl;
    ss << "    lib.accessResistance.d_v = d_accessResistance" << SUFFIX << ";" << endl;
    ss << "    lib.iSettleDuration.v = iSettleDuration" << SUFFIX << ";" << endl;
    ss << "    lib.iSettleDuration.d_v = d_iSettleDuration" << SUFFIX << ";" << endl;
    ss << "    lib.Imax.v = Imax" << SUFFIX << ";" << endl;
    ss << "    lib.Imax.d_v = d_Imax" << SUFFIX << ";" << endl;
    ss << "    lib.dt.v = dt" << SUFFIX << ";" << endl;
    ss << "    lib.dt.d_v = d_dt" << SUFFIX << ";" << endl;
    ss << "    lib.targetOffset.v = targetOffset" << SUFFIX << ";" << endl;
    ss << "    lib.targetOffset.d_v = d_targetOffset" << SUFFIX << ";" << endl;

    ss << endl;
    ss << "    pointers.run =& runStreamed;" << endl;
    ss << "    pointers.reset =& initialize;" << endl;
    ss << "    pointers.sync =& libSync;" << endl;
    ss << "    pointers.resetEvents =& libResetEvents;" << endl;
    ss << "    pointers.recordEvent =& libRecordEvent;" << endl;
    ss << "    pointers.waitEvent =& libWaitEvent;" << endl;
    ss << "    pointers.createSim =& createSim;" << endl;
    ss << "    pointers.destroySim =& destroySim;" << endl;
    ss << "    pointers.resizeTarget =& resizeTarget;" << endl;
    ss << "    pointers.pushTarget =& pushTarget;" << endl;
    ss << "    pointers.resizeOutput =& resizeOutput;" << endl;
    ss << "    pointers.pullOutput =& pullOutput;" << endl;
    ss << "    pointers.resizeSummary =& resizeSummary;" << endl;
    ss << "    pointers.pullSummary =& pullSummary;" << endl;
    ss << "    pointers.profile =& profile;" << endl;
    ss << "    pointers.cluster =& cluster;" << endl;
    ss << "    pointers.pullClusters =& pullClusters;" << endl;
    ss << "    pointers.pullPrimitives =& pullPrimitives;" << endl;
    ss << "    pointers.bubble =& bubble;" << endl;
    ss << "    pointers.pullBubbles =& pullBubbles;" << endl;
    ss << "    pointers.find_deltabar =& find_deltabar;" << endl;
    ss << "    pointers.observe_no_steps =& observe_no_steps;" << endl;
    ss << "    pointers.genRandom =& genRandom;" << endl;
    ss << "    pointers.get_posthoc_deviations =& get_posthoc_deviations;" << endl;
    ss << "    pointers.principal_components =& principal_components;" << endl;
    ss << "    pointers.get_mean_distance =& get_mean_distance;" << endl;
    ss << "    pointers.copy_param =& copy_param;" << endl;
    ss << "    pointers.cl_compare_to_target =& cl_compare_to_target;" << endl;
    ss << "    pointers.cl_compare_models =& cl_compare_models;" << endl;
    ss << "    return pointers;" << endl;
    ss << "}" << endl;

    ss << endl;
    ss << "namespace " << SUFFIX << "_neuron {" << endl;

    return ss.str();
}

void UniversalLibrary::resizeTarget(size_t nTraces, size_t nSamples)
{
    pointers.resizeTarget(nSamples*nTraces);
    targetStride = nTraces;
}

void UniversalLibrary::resizeOutput(size_t nSamples)
{
    pointers.resizeOutput(nSamples * NMODELS);
}

void UniversalLibrary::setRundata(size_t modelIndex, const RunData &rund)
{
    clampGain[modelIndex] = rund.clampGain;
    accessResistance[modelIndex] = rund.accessResistance;
    iSettleDuration[modelIndex] = rund.settleDuration/rund.dt;
    Imax[modelIndex] = rund.Imax;
    dt[modelIndex] = rund.dt;
}

void UniversalLibrary::push()
{
    for ( Variable &v : stateVariables )
        push(v);
    pushParams();
    push(stim);
    push(obs);
    push(clampGain);
    push(accessResistance);
    push(iSettleDuration);
    push(Imax);
    push(dt);
    push(targetOffset);
}

void UniversalLibrary::pushParams()
{
    for ( Variable &v : adjustableParams )
        push(v);
}

void UniversalLibrary::pull()
{
    for ( Variable &v : stateVariables )
        pull(v);
    for ( Variable &v : adjustableParams )
        pull(v);
    pull(stim);
    pull(obs);
    pull(clampGain);
    pull(accessResistance);
    pull(iSettleDuration);
    pull(Imax);
    pull(dt);
    pull(targetOffset);
}
