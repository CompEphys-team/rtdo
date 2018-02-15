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

ExperimentLibrary::ExperimentLibrary(const Project & p, bool compile) :
    project(p),
    model(project.model()),
    stateVariables(model.stateVariables),
    adjustableParams(model.adjustableParams),
    lib(compile ? compile_and_load() : load()),
    populate((decltype(populate))dlsym(lib, "populate")),
    pointers(populate(stateVariables, adjustableParams)),
    t(*(pointers.t)),
    iT(*(pointers.iT)),
    simCycles(*(pointers.simCycles)),
    clampGain(*(pointers.clampGain)),
    accessResistance(*(pointers.accessResistance)),
    Vmem(*(pointers.Vmem)),
    Vprev(*(pointers.Vprev)),
    Imem(*(pointers.Imem)),
    VC(*(pointers.VC)),
    getErr(*(pointers.getErr)),
    VClamp0(*pointers.VClamp0),
    dVClamp(*pointers.dVClamp),
    tStep(*pointers.tStep),
    err(pointers.err),
    meta_hP(pointers.meta_hP)
{
}

ExperimentLibrary::~ExperimentLibrary()
{
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
    std::string name = model.name(ModuleType::Experiment);
    std::string arg1 = std::string("generating ") + name + " in";
    char *argv[2] = {const_cast<char*>(arg1.c_str()), const_cast<char*>(directory.c_str())};
    if ( generateAll(2, argv) )
        throw std::runtime_error("Code generation failed.");

    // Compile
    std::string dir = directory + "/" + name + "_CODE";
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
        Variable("Vmem"),
        Variable("Vprev"),
        Variable("Imem"),
        Variable("VC", "", "bool"),
        Variable("getErr", "", "bool"),
        Variable("VClamp0"),
        Variable("dVClamp"),
        Variable("tStep")
    };
    for ( Variable &p : globals ) {
        n.extraGlobalNeuronKernelParameters.push_back(p.name);
        n.extraGlobalNeuronKernelParameterTypes.push_back(p.type);
    }

    std::vector<Variable> vars = {
        Variable("err")
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

    ss << model.populateStructs() << endl;

    ss << R"EOF(
clamp.VClamp0 = $(VClamp0);
clamp.dVClamp = $(dVClamp);

if ( $(getErr) ) {
    scalar tmp = clamp.getCurrent(t, $(V)) - $(Imem);
    $(err) += tmp*tmp;
}

RKF45(t, t+$(tStep), DT/$(simCycles), $(tStep), $(meta_hP), state, params, clamp);

)EOF";

    ss << model.extractState();

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
        ss << "    state[" << i++ << "].v = " << v.name << SUFFIX << ";" << endl;
    }
    i = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    param[" << i++ << "].v = " << p.name << SUFFIX << ";" << endl;
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
    ss << "    pointers.meta_hP = meta_hP" << SUFFIX << ";" << endl;
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
    simCycles = rund.simCycles;
}
