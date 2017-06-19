#include "profilerlibrary.h"
#include "modelSpec.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include "global.h"
#include "cuda_helper.h"
#include <dlfcn.h>
#include "project.h"

#define SUFFIX "PROF"

static ProfilerLibrary *_this;
static void redirect(NNmodel &n) { _this->GeNN_modelDefinition(n); }

ProfilerLibrary::ProfilerLibrary(const Project & p, bool compile) :
    project(p),
    model(project.model()),
    stateVariables(model.stateVariables),
    adjustableParams(model.adjustableParams),
    lib(compile ? compile_and_load() : load()),
    populate((decltype(populate))dlsym(lib, "populate")),
    pointers(populate(stateVariables, adjustableParams)),
    simCycles(*(pointers.simCycles)),
    samplingInterval(*(pointers.samplingInterval)),
    clampGain(*(pointers.clampGain)),
    accessResistance(*(pointers.accessResistance))
{
}

ProfilerLibrary::~ProfilerLibrary()
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

void *ProfilerLibrary::compile_and_load()
{
    std::string directory = project.dir().toStdString();

    // Generate code
    _this = this;
    MetaModel::modelDef = redirect;
    std::string name = model.name(ModuleType::Profiler);
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

void *ProfilerLibrary::load()
{
    std::string libfile = project.dir().toStdString() + "/" + model.name(ModuleType::Profiler) + "_CODE/runner.so";
    dlerror();
    void *libp;
    if ( ! (libp = dlopen(libfile.c_str(), RTLD_NOW)) )
        throw std::runtime_error(std::string("Library load failed: ") + dlerror());

    ++MetaModel::numLibs;
    return libp;
}

void ProfilerLibrary::GeNN_modelDefinition(NNmodel &nn)
{
    std::vector<double> fixedParamIni, variableIni;
    neuronModel n = model.generate(nn, fixedParamIni, variableIni);

    std::vector<Variable> globals = {
        Variable("simCycles", "", "int"),
        Variable("samplingInterval", "", "int"),
        Variable("clampGain"),
        Variable("accessResistance")
    };
    for ( Variable &p : globals ) {
        n.extraGlobalNeuronKernelParameters.push_back(p.name);
        n.extraGlobalNeuronKernelParameterTypes.push_back(p.type);
    }

    std::vector<Variable> stimArg = {
        Variable("current", "" , "scalar*"),
        Variable("stim", "", "void*")

//        Variable("stim_duration"),
//        Variable("stim_tObsBegin"),
//        Variable("stim_tObsEnd"),
//        Variable("stim_baseV"),
//        Variable("stim_numSteps", "", "size_t")
    };
//    for ( size_t i = 0; i < 10; i++ ) {
//        stimArg.push_back(Variable(QString("stim_step%1_t").arg(i).toStdString()));
//        stimArg.push_back(Variable(QString("stim_step%1_V").arg(i).toStdString()));
//        stimArg.push_back(Variable(QString("stim_step%1_ramp").arg(i).toStdString(), "", "bool"));
//    }
    for ( Variable &p : stimArg ) {
        n.extraGlobalNeuronKernelParameters.push_back(p.name);
        n.extraGlobalNeuronKernelParameterTypes.push_back(p.type);
    }

    n.simCode = simCode();
    n.supportCode = supportCode(globals, {});

    int numModels = nModels.size();
    nModels.push_back(n);
    nn.setName(model.name(ModuleType::Profiler));
    nn.addNeuronPopulation(SUFFIX, project.profNumPairs()*2, numModels, fixedParamIni, variableIni);

    nn.finalize();
}

std::string ProfilerLibrary::simCode()
{
    return std::string("\n#ifndef _") + model.name(ModuleType::Profiler) + "_neuronFnct_cc\n"
    + R"EOF(
const Stimulation stim = *(Stimulation*)$(stim);
scalar mdt = DT/$(simCycles);
t = 0.;
unsigned int samples = 0;
for ( unsigned int mt = 0; t < stim.duration && t < stim.tObsEnd; mt++ ) {
    t = mt*mdt;
    Isyn = ($(clampGain)*(getCommandVoltage(stim, t) - $(V)) - $(V)) / $(accessResistance);
)EOF"
    + model.kernel("    ", true, false)
    + R"EOF(
    if ( (mt+1) % $(samplingInterval) == 0 && t > stim.tObsBegin && t < stim.tObsEnd )
        $(current)[id + NMODELS * samples++] = Isyn;
}
#else
Isyn += 0.;
#endif
)EOF";
}

std::string ProfilerLibrary::supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars)
{
    std::stringstream ss;

    ss << "} // break namespace for STL includes:" << endl;

    ss << "#define NVAR " << stateVariables.size() << endl;
    ss << "#define NPARAM " << adjustableParams.size() << endl;
    ss << "#define NPAIRS " << project.profNumPairs() << endl;
    ss << "#define NMODELS " << project.profNumPairs()*2 << endl;
    ss << "#define NCOMP " << project.profNumPairs()*project.profNumPairs() << endl;
    ss << "#include \"definitions.h\"" << endl;
    ss << "#include \"profilerlibrary.h\"" << endl;
    ss << "#include \"../core/supportcode.cpp\"" << endl;
    ss << "#include \"profilerlibrary.cu\"" << endl;
    ss << endl;

    ss << "extern \"C\" ProfilerLibrary::Pointers populate(std::vector<StateVariable> &state, "
                                                         << "std::vector<AdjustableParam> &param) {" << endl;
    ss << "    ProfilerLibrary::Pointers pointers;" << endl;
    ss << "    libInit(pointers);" << endl;
    ss << "    pointers.d_param.resize(" << adjustableParams.size() << ");" << endl;
    int i = 0;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    state[" << i++ << "].v = " << v.name << SUFFIX << ";" << endl;
    }
    i = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    param[" << i << "].v = " << p.name << SUFFIX << ";" << endl;
        ss << "    pointers.d_param[" << i << "] = d_" << p.name << SUFFIX << ";" << endl;
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
    ss << "    pointers.push =& push" << SUFFIX << "StateToDevice;" << endl;
    ss << "    pointers.pull =& pull" << SUFFIX << "StateFromDevice;" << endl;
    ss << "    pointers.doProfile =& doProfile;" << endl;
    ss << "    pointers.reset =& initialize;" << endl;
    ss << "    return pointers;" << endl;
    ss << "}" << endl;

    ss << endl;
    ss << "namespace " << SUFFIX << "_neuron {" << endl;

    return ss.str();
}

void ProfilerLibrary::profile(Stimulation stim, size_t targetParam, double &accuracy, double &median_norm_gradient)
{
    unsigned int nSamples = std::ceil((stim.tObsEnd-stim.tObsBegin) / (samplingInterval * project.dt() / simCycles));
    *pointers.stim = stim;
    pointers.doProfile(pointers, targetParam, nSamples, accuracy, median_norm_gradient);
}

void ProfilerLibrary::setRunData(RunData rund)
{
    clampGain = rund.clampGain;
    accessResistance = rund.accessResistance;
    simCycles = rund.simCycles;
}

