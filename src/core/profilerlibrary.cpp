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

ProfilerLibrary::ProfilerLibrary(const Project & p, bool compile, bool light) :
    project(p),
    model(project.model()),
    stateVariables(model.stateVariables),
    adjustableParams(model.adjustableParams),
    lib(light ? nullptr : compile ? compile_and_load() : load()),
    populate(light ? nullptr : (decltype(populate))dlsym(lib, "populate")),
    pointers(light ? Pointers() : populate(stateVariables, adjustableParams)),
    isLight(light),
    dt(light ? dummyScalar : *pointers.dt),
    samplingInterval(light ? dummyInt : *pointers.samplingInterval),
    clampGain(light ? dummyScalar : *pointers.clampGain),
    accessResistance(light ? dummyScalar : *pointers.accessResistance)
{
}

ProfilerLibrary::~ProfilerLibrary()
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

void *ProfilerLibrary::compile_and_load()
{
    std::string directory = project.dir().toStdString();

    // Generate code
    _this = this;
    MetaModel::modelDef = redirect;
    generateCode(directory, model);

    // Compile
    std::string dir = directory + "/" + model.name(ModuleType::Profiler) + "_CODE";
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
        Variable("dt"),
        Variable("samplingInterval", "", "int"),
        Variable("clampGain"),
        Variable("accessResistance"),
        Variable("current", "" , "scalar*"),
        Variable("settling", "", "bool")
    };
    for ( Variable &p : globals ) {
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
    std::stringstream ss;
    ss << std::string("\n#ifndef _") + model.name(ModuleType::Profiler) + "_neuronFnct_cc\n";
    ss << "// Mention $(clampGain) and $(accessResistance) to ensure they are present." << endl;
    ss << R"EOF(
unsigned int samples = 0;
int mt = 0, tStep = 0;
while ( mt < stim.duration ) {
    getiCommandSegment(stim, mt, stim.duration - mt, $(dt), clamp.VClamp0, clamp.dVClamp, tStep);

    for ( int i = 0; i < tStep; i++ ) {
        t = mt * $(dt);
        if ( !$(settling) && mt >= stim.tObsBegin && (mt-stim.tObsBegin) % $(samplingInterval) == 0 )
            $(current)[id + NMODELS * samples++] = clamp.getCurrent(t, state.V);

        // Integrate
        RK4(t, $(dt), state, params, clamp);
        ++mt;
    }
}

if ( !$(settling) )
    return; // do not overwrite settled initial state
)EOF";

    ss << "#endif";
    return ss.str();
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

    ss << model.supportCode() << endl;

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
    ss << "    pointers.pushStim =& pushStim;" << endl;
    ss << "    pointers.step =& stepTimeGPU;" << endl;
    ss << "    pointers.doProfile =& doProfile;" << endl;
    ss << "    pointers.reset =& initialize;" << endl;
    ss << "    libInitPost(pointers);" << endl;
    ss << "    return pointers;" << endl;
    ss << "}" << endl;

    ss << endl;
    ss << "namespace " << SUFFIX << "_neuron {" << endl;

    return ss.str();
}

void ProfilerLibrary::settle(iStimulation stim)
{
    pointers.pushStim(stim);
    *pointers.settling = true;
    pointers.step();
}

void ProfilerLibrary::profile(iStimulation stim, size_t targetParam, double &accuracy, double &median_norm_gradient)
{
    unsigned int nSamples = (stim.tObsEnd - stim.tObsBegin) / samplingInterval;
    stim.duration = stim.tObsEnd;
    pointers.pushStim(stim);
    *pointers.settling = false;
    pointers.doProfile(pointers, targetParam, nSamples, accuracy, median_norm_gradient);
}

void ProfilerLibrary::setRunData(RunData rund)
{
    clampGain = rund.clampGain;
    accessResistance = rund.accessResistance;
}

