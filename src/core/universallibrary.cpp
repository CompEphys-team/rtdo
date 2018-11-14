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
    summary("summary", "double"),

    lib(light ? nullptr : compile ? compile_and_load() : load()),
    populate(light ? nullptr : (decltype(populate))dlsym(lib, "populate")),
    pointers(light ? Pointers() : populate(*this)),
    isLight(light),

    simCycles(light ? dummyInt : *pointers.simCycles),
    integrator(light ? dummyIntegrator : *pointers.integrator),
    assignment(light ? dummyUInt : *pointers.assignment),
    targetStride(light ? dummyInt : *pointers.targetStride),

    target(light ? dummyScalarPtr : *pointers.target),
    output(light ? dummyScalarPtr : *pointers.output),

    clusters(light ? dummyScalarPtr : *pointers.clusters),
    clusterLen(light ? dummyIntPtr : *pointers.clusterLen)
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
    model.save_selection = {"summary"};
    model.singular_stim_vars = {&stim, &obs};
    model.singular_clamp_vars = {&clampGain, &accessResistance};
    model.singular_rund_vars = {&iSettleDuration, &Imax, &dt};
    model.singular_target_vars = {&targetOffset};

    generateCode(directory, model);

    model.isUniversalLib = false;
    model.save_state_condition = "";
    model.save_selectively = false;

    // Compile
    std::string dir = directory + "/" + model.name(ModuleType::Universal) + "_CODE";
    std::ofstream makefile(dir + "/Makefile", std::ios_base::app);
    makefile << endl;
    makefile << "runner.so: runner.o" << endl;
    makefile << "\t$(CXX) -o $@ $< -shared" << endl;
    std::stringstream cmd;
    cmd << "cd " << dir << " && make runner.so";
    if ( system(cmd.str().c_str()) )
        throw std::runtime_error("Code compile failed.");

    // Load library
    return load();
}

void *UniversalLibrary::load()
{
    std::string libfile = project.dir().toStdString() + "/" + model.name(ModuleType::Universal) + "_CODE/runner.so";
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
        Variable("targetStride", "", "int")
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
    n.varNames.push_back(summary.name);
        n.varTypes.push_back(summary.type);
        variableIni.push_back(0.);

    n.simCode = simCode();
    n.supportCode = supportCode(globals);

    int numModels = nModels.size();
    nModels.push_back(n);
    nn.setName(model.name(ModuleType::Universal));
    nn.addNeuronPopulation(SUFFIX, NMODELS, numModels, fixedParamIni, variableIni);

    nn.finalize();
}

std::string UniversalLibrary::simCode()
{
    std::stringstream ss;
    ss << endl << "#ifndef _" << model.name(ModuleType::Universal) << "_neuronFnct_cc" << endl;

    ss << R"EOF(
scalar mdt = $(dt) / $(simCycles);
scalar hP = 0.;

// Settle
if ( $(iSettleDuration) > 0 ) {
    t = -$(iSettleDuration)*$(dt);
    clamp.dVClamp = 0.0;
    clamp.VClamp0 = $(stim).baseV;
    if ( $(integrator) == IntegrationMethod::RungeKutta4 ) {
        for ( int mt = 0, mtEnd = $(iSettleDuration) * $(simCycles); mt < mtEnd; mt++ )
            RK4(t + mt*mdt, mdt, state, params, clamp);
    } else if ( $(integrator) == IntegrationMethod::ForwardEuler ) {
        for ( int mt = 0, mtEnd = $(iSettleDuration) * $(simCycles); mt < mtEnd; mt++ )
            Euler(t + mt*mdt, mdt, state, params, clamp);
    } else /*if ( $(integrator) == IntegrationMethod::RungeKuttaFehlberg45 )*/ {
        RKF45(t, 0., mdt, $(iSettleDuration)*$(dt), hP, state, params, clamp);
    }
}

int iT = 0;
int tStep = 0;
int nSamples = 0;
int nextObs = 0;
t = 0.;
if ( !($(assignment)&ASSIGNMENT_SUMMARY_PERSIST) )
    $(summary) = 0;

while ( !($(assignment)&ASSIGNMENT_SETTLE_ONLY)
        && iT < $(stim).duration
        && nextObs < iObservations::maxObs ) {

    // Integrate unobserved
    while ( iT < $(obs).start[nextObs] ) {
        getiCommandSegment($(stim), iT, $(stim).duration - iT, $(dt), clamp.VClamp0, clamp.dVClamp, tStep);
        tStep = min(tStep, $(obs).start[nextObs] - iT);
        if ( $(integrator) == IntegrationMethod::RungeKutta4 ) {
            for ( int mt = 0, mtEnd = tStep * $(simCycles); mt < mtEnd; mt++ )
                RK4(t + mt*mdt, mdt, state, params, clamp);
        } else if ( $(integrator) == IntegrationMethod::ForwardEuler ) {
            for ( int mt = 0, mtEnd = tStep * $(simCycles); mt < mtEnd; mt++ )
                Euler(t + mt*mdt, mdt, state, params, clamp);
        } else /*if ( $(integrator) == IntegrationMethod::RungeKuttaFehlberg45 )*/ {
            RKF45(t, t + tStep*$(dt), mdt, tStep*$(dt), hP, state, params, clamp);
        }

        if ( ($(assignment) & ASSIGNMENT_REPORT_TIMESERIES)
          && !($(assignment) & ASSIGNMENT_TIMESERIES_COMPACT)
          && ($(assignment) & ASSIGNMENT_TIMESERIES_ZERO_UNTOUCHED_SAMPLES) )
            for ( int i = 0; i < tStep; i++ )
                dd_timeseries[id + NMODELS*(iT+i)] = 0.;

        iT += tStep;
        t = iT * $(dt);
    }

    // Process results while integrating stepwise with $(dt)
    while ( iT < $(obs).stop[nextObs] ) {
        getiCommandSegment($(stim), iT, $(stim).duration - iT, $(dt), clamp.VClamp0, clamp.dVClamp, tStep);
        tStep = min(tStep, $(obs).stop[nextObs] - iT);
        while ( tStep ) {
            // Process results
            scalar value = clip(clamp.getCurrent(t, state.V), $(Imax)), diff;

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
                    diff = __shfl_up_sync(0xffffffff, value, 1) - value;
                    if ( (threadIdx.x & 31) == 0 )
                        diff = value;
                    break;
                case ASSIGNMENT_SUMMARY_COMPARE_NONE:
                default:
                    diff = value;
                    break;
                }

                $(summary) += ($(assignment) & ASSIGNMENT_SUMMARY_SQUARED) ? (double(diff)*diff) : fabs(diff);
            }

            // Integrate forward by one $(dt)
            if ( $(integrator) == IntegrationMethod::RungeKutta4 ) {
                for ( int mt = 0; mt < $(simCycles); mt++ )
                    RK4(t + mt*mdt, mdt, state, params, clamp);
            } else if ( $(integrator) == IntegrationMethod::ForwardEuler ) {
                for ( int mt = 0; mt < $(simCycles); mt++ )
                    Euler(t + mt*mdt, mdt, state, params, clamp);
            } else /*if ( $(integrator) == IntegrationMethod::RungeKuttaFehlberg45 )*/ {
                RKF45(t, t + $(dt), mdt, $(dt), hP, state, params, clamp);
            }
            ++nSamples;
            --tStep;
            ++iT;
            t = iT * $(dt);
        }
    }
    ++nextObs;
}

if ( ($(assignment)&ASSIGNMENT_REPORT_SUMMARY) && ($(assignment)&ASSIGNMENT_SUMMARY_AVERAGE) )
    $(summary) /= nSamples;

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
    ss << "#define NMODELS " << NMODELS << endl;
    ss << "#define NPARAMS " << adjustableParams.size() << endl;
    ss << "#define MAXCLUSTERS " << maxClusters << endl;
    ss << "#include \"universallibrary.cu\"" << endl;
    ss << endl;

    ss << model.supportCode() << endl;

    ss << project.simulatorCode() << endl;

    ss << "extern \"C\" UniversalLibrary::Pointers populate(UniversalLibrary &lib) {" << endl;
    ss << "    UniversalLibrary::Pointers pointers;" << endl;
    ss << "    libInit(lib, pointers);" << endl;
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
    ss << "    lib.summary.v = summary" << SUFFIX << ";" << endl;
    ss << "    lib.summary.d_v = d_summary" << SUFFIX << ";" << endl;

    ss << endl;
    ss << "    pointers.run =& stepTimeGPU;" << endl;
    ss << "    pointers.reset =& initialize;" << endl;
    ss << "    pointers.createSim =& createSim;" << endl;
    ss << "    pointers.destroySim =& destroySim;" << endl;
    ss << "    pointers.resizeTarget =& resizeTarget;" << endl;
    ss << "    pointers.pushTarget =& pushTarget;" << endl;
    ss << "    pointers.resizeOutput =& resizeOutput;" << endl;
    ss << "    pointers.pullOutput =& pullOutput;" << endl;
    ss << "    pointers.profile =& profile;" << endl;
    ss << "    pointers.cluster =& cluster;" << endl;
    ss << "    pointers.find_deltabar =& find_deltabar;" << endl;
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
    for ( Variable &v : adjustableParams )
        push(v);
    push(stim);
    push(obs);
    push(clampGain);
    push(accessResistance);
    push(iSettleDuration);
    push(Imax);
    push(dt);
    push(targetOffset);
    push(summary);
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
    pull(summary);
}
