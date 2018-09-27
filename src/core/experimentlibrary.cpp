#include "experimentlibrary.h"
#include "modelSpec.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include "global.h"
#include "cuda_helper.h"
#include <dlfcn.h>
#include "project.h"

#define SUFFIX "EXP"

static ExperimentLibrary *_this;
static void redirect(NNmodel &n) { _this->GeNN_modelDefinition(n); }

ExperimentLibrary::ExperimentLibrary(const Project & p, bool compile, bool light) :
    project(p),
    model(project.model()),
    stateVariables(model.stateVariables),
    adjustableParams(model.adjustableParams),
    lib(light ? nullptr : compile ? compile_and_load() : load()),
    populate(light ? nullptr : (decltype(populate))dlsym(lib, "populate")),
    pointers(light ? Pointers() : populate(stateVariables, adjustableParams)),
    isLight(light),
    t(light ? dummyScalar : *pointers.t),
    iT(light ? dummyULL : *pointers.iT),
    simCycles(light ? dummyInt : *pointers.simCycles),
    clampGain(light ? dummyScalar : *pointers.clampGain),
    accessResistance(light ? dummyScalar : *pointers.accessResistance),
    Imax(light ? dummyScalar : *pointers.Imax),
    integrator(light ? dummyInt : *pointers.integrator),
    Vmem(light ? dummyScalar : *pointers.Vmem),
    Vprev(light ? dummyScalar : *pointers.Vprev),
    Imem(light ? dummyScalar : *pointers.Imem),
    VC(light ? dummyBool : *pointers.VC),
    getErr(light ? dummyBool : *pointers.getErr),
    VClamp0(light ? dummyScalar : *pointers.VClamp0),
    dVClamp(light ? dummyScalar : *pointers.dVClamp),
    tStep(light ? dummyScalar : *pointers.tStep),
    setVariance(light ? dummyBool : *pointers.setVariance),
    variance(light ? dummyScalar : *pointers.variance),
    getLikelihood(light ? dummyBool : *pointers.getLikelihood),

    err(pointers.err),
    meta_hP(pointers.meta_hP),
    ext_variance(pointers.ext_variance)
{
}

ExperimentLibrary::~ExperimentLibrary()
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

void *ExperimentLibrary::compile_and_load()
{
    std::string directory = project.dir().toStdString();

    // Generate code
    _this = this;
    MetaModel::modelDef = redirect;
    generateCode(directory, model);

    // Compile
    std::string dir = directory + "/" + model.name(ModuleType::Experiment) + "_CODE";
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

void *ExperimentLibrary::load()
{
    std::string libfile = project.dir().toStdString() + "/" + model.name(ModuleType::Experiment) + "_CODE/runner.so";
    dlerror();
    void *libp;
    if ( ! (libp = dlopen(libfile.c_str(), RTLD_NOW)) )
        throw std::runtime_error(std::string("Library load failed: ") + dlerror());

    ++MetaModel::numLibs;
    return libp;
}

void ExperimentLibrary::GeNN_modelDefinition(NNmodel &nn)
{
    std::vector<double> fixedParamIni, variableIni;
    neuronModel n = model.generate(nn, fixedParamIni, variableIni);

    std::vector<Variable> globals = {
        Variable("simCycles", "", "int"),
        Variable("clampGain"),
        Variable("accessResistance"),
        Variable("Imax"),
        Variable("integrator", "", "int"),
        Variable("Vmem"),
        Variable("Vprev"),
        Variable("Imem"),
        Variable("VC", "", "bool"),
        Variable("getErr", "", "bool"),
        Variable("VClamp0"),
        Variable("dVClamp"),
        Variable("tStep"),
        Variable("setVariance", "", "bool"),
        Variable("variance"),
        Variable("getLikelihood", "", "bool")
    };
    for ( Variable &p : globals ) {
        n.extraGlobalNeuronKernelParameters.push_back(p.name);
        n.extraGlobalNeuronKernelParameterTypes.push_back(p.type);
    }

    std::vector<Variable> vars = {
        Variable("err"),
        Variable("meta_hP"),
        Variable("ext_variance")
    };
    for ( Variable &v : vars ) {
        n.varNames.push_back(v.name);
        n.varTypes.push_back(v.type);
        variableIni.push_back(0.0);
    }

    n.simCode = simCode();
    n.supportCode = supportCode(globals, vars);

    int numModels = nModels.size();
    nModels.push_back(n);
    nn.setName(model.name(ModuleType::Experiment));
    nn.addNeuronPopulation(SUFFIX, project.expNumCandidates(), numModels, fixedParamIni, variableIni);

    nn.finalize();
}

std::string ExperimentLibrary::simCode()
{
    std::stringstream ss;
    ss << endl << "#ifndef _" << model.name(ModuleType::Experiment) << "_neuronFnct_cc" << endl;
    ss << "// Mention $(clampGain) and $(accessResistance) to ensure they are present." << endl;

    ss << R"EOF(
clamp.VClamp0 = $(VClamp0);
clamp.dVClamp = $(dVClamp);

if ( $(setVariance) ) {
    $(ext_variance) = fmax(0., $(variance) - state.state__variance(params));
}

if ( $(getErr) ) {
    scalar tmp = clip(clamp.getCurrent(t, state.V), $(Imax)) - $(Imem);
    $(err) += tmp*tmp;
} else if ( $(getLikelihood) ) {
    $(err) += state.state__negLogLikelihood(clip(clamp.getCurrent(t, state.V), $(Imax)) - $(Imem), $(ext_variance), params);
}

scalar mdt = $(tStep)/$(simCycles);

if ( $(integrator) == (int)IntegrationMethod::RungeKutta4 ) {
    for ( unsigned int mt = 0; mt < $(simCycles); mt++ )
        RK4(t + mt*mdt, mdt, state, params, clamp);
} else if ( $(integrator) == (int)IntegrationMethod::ForwardEuler ) {
    for ( unsigned int mt = 0; mt < $(simCycles); mt++ )
        Euler(t + mt*mdt, mdt, state, params, clamp);
} else /*if ( $(integrator) == (int)IntegrationMethod::RungeKuttaFehlberg45 )*/ {
    RKF45(t, t+$(tStep), mdt, $(tStep), $(meta_hP), state, params, clamp);
}
#endif
)EOF";

    return ss.str();

//   } else { // Pattern clamp: Inject current $(Imem) and clamp to voltage $(Vprev) to $(Vmem), linearly interpolated
//       scalar Vcmd = (($(simCycles)-mt) * $(Vprev) + mt * $(Vmem)) / $(simCycles);
//       Isyn = $(clampGain)*(Vcmd-$(V)) / $(accessResistance) + $(Imem); // Perfect clamp
//   }
}

std::string ExperimentLibrary::supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars)
{
    std::stringstream ss;

    ss << "} // break namespace for STL includes:" << endl;

    ss << "#define NVAR " << stateVariables.size() << endl;
    ss << "#define NPARAM " << adjustableParams.size() << endl;
    ss << "#include \"definitions.h\"" << endl;
    ss << "#include \"experimentlibrary.h\"" << endl;
    ss << "#include \"../core/supportcode.cpp\"" << endl;
    ss << "#include \"experimentlibrary.cu\"" << endl;
    ss << endl;

    ss << model.supportCode() << endl;

    ss << project.simulatorCode();

    ss << "extern \"C\" ExperimentLibrary::Pointers populate(std::vector<StateVariable> &state, "
                                                         << "std::vector<AdjustableParam> &param) {" << endl;
    ss << "    ExperimentLibrary::Pointers pointers;" << endl;
    ss << "    libInit(pointers, " << project.expNumCandidates() << ");" << endl;
    int i = 0;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    state[" << i << "].v = " << v.name << SUFFIX << ";" << endl;
        ss << "    state[" << i << "].d_v = d_" << v.name << SUFFIX << ";" << endl;
        ++i;
    }
    i = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    param[" << i << "].v = " << p.name << SUFFIX << ";" << endl;
        ss << "    param[" << i << "].d_v = d_" << p.name << SUFFIX << ";" << endl;
        ++i;
    }
    ss << endl;
    for ( const Variable &p : globals ) {
        ss << "    pointers." << p.name << " =& " << p.name << SUFFIX << ";" << endl;
    }
    for ( const Variable &v : vars ) {
        ss << "    pointers." << v.name << " = " << v.name << SUFFIX << ";" << endl;
        ss << "    pointers.d_" << v.name << " = d_" << v.name << SUFFIX << ";" << endl;
    }
    ss << endl;
    ss << "    pointers.t =& t;" << endl;
    ss << "    pointers.iT =& iT;" << endl;
    ss << "    pointers.push =& push" << SUFFIX << "StateToDevice;" << endl;
    ss << "    pointers.pull =& pull" << SUFFIX << "StateFromDevice;" << endl;
    ss << "    pointers.step =& stepTimeGPU;" << endl;
    ss << "    pointers.reset =& initialize;" << endl;
    ss << "    pointers.createSim =& createSim;" << endl;
    ss << "    pointers.destroySim =& destroySim;" << endl;
    ss << "    return pointers;" << endl;
    ss << "}" << endl;

    ss << endl;
    ss << "namespace " << SUFFIX << "_neuron {" << endl;

    return ss.str();
}

void ExperimentLibrary::setRunData(RunData rund)
{
    clampGain = rund.clampGain;
    accessResistance = rund.accessResistance;
    Imax = rund.Imax;
    simCycles = rund.simCycles;
    integrator = int(rund.integrator);
    tStep = rund.dt;
}

void ExperimentLibrary::step(double dt, int cycles, bool advance_iT)
{
    scalar tPrev = t, tStepPrev = tStep, simCyclesPrev = simCycles;
    tStep = dt;
    simCycles = cycles;

    pointers.step();
    t = tPrev + tStep;
    if ( !advance_iT )
        --iT;

    tStep = tStepPrev;
    simCycles = simCyclesPrev;
}
