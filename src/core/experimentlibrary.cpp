#include "experimentlibrary.h"
#include "modelSpec.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include "global.h"
#include "cuda_helper.h"
#include <dlfcn.h>

#define SUFFIX "EXP"

static ExperimentLibrary *_this;
static void redirect(NNmodel &n) { _this->GeNN_modelDefinition(n); }

ExperimentLibrary::ExperimentLibrary(MetaModel &m, const std::string &directory, const ExperimentData &expd) :
    expd(expd),
    model(m),
    stateVariables(m.stateVariables),
    adjustableParams(m.adjustableParams),
    lib(loadLibrary(directory)),
    populate((decltype(populate))dlsym(lib, "populate")),
    pointers(populate(stateVariables, adjustableParams)),
    t(*(pointers.t)),
    iT(*(pointers.iT)),
    simCycles(*(pointers.simCycles)),
    clampGain(*(pointers.clampGain)),
    accessResistance(*(pointers.accessResistance)),
    Vmem(*(pointers.Vmem)),
    Imem(*(pointers.Imem)),
    VC(*(pointers.VC)),
    getErr(*(pointers.getErr)),
    err(pointers.err)
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

void *ExperimentLibrary::loadLibrary(const string &directory)
{
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
    dlerror();
    void *libp;
    if ( ! (libp = dlopen((dir + "/runner.so").c_str(), RTLD_NOW)) )
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
        Variable("Imem"),
        Variable("VC", "", "bool"),
        Variable("getErr", "", "bool")
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
    nn.addNeuronPopulation(SUFFIX, expd.numCandidates, numModels, fixedParamIni, variableIni);

    nn.finalize();
}

std::string ExperimentLibrary::simCode()
{
    return R"EOF(
scalar mdt = DT/$(simCycles);
for ( unsigned int mt = 0; mt < $(simCycles); mt++ ) {
   if ( $(VC) ) {
       Isyn = ($(clampGain)*($(Vmem)-$(V)) - $(V)) / $(accessResistance);
   } else {
       Isyn = $(Imem);
   }
)EOF"
   + model.kernel("    ", true, false)
   + R"EOF(
}

if ( $(getErr) ) {
   if ( $(VC) )
       $(err) += fabs(Isyn - $(Imem));
   else
       $(err) += fabs($(V) - $(Vmem));
}
)EOF";
}

std::string ExperimentLibrary::daqCode()
{
    std::stringstream ss;
    ss << R"EOF(
class Simulator : public DAQ
{
private:
    double t;

public:
    Simulator() : DAQ(nullptr), t(0.0)
    {
        initialise();
    }

    ~Simulator() {}

    void run(Stimulation s)
    {
        if ( running )
            return;
        currentStim = s;
        running = true;
    }

    void next()
    {
        if ( !running )
            return;
        double Isyn;
        scalar Vcmd = getCommandVoltage(currentStim, t);
)EOF";
    ss << "        const float mdt = DT/simCycles" << SUFFIX << ";" << endl;
    ss << "        for ( unsigned int mt = 0; mt < simCycles" << SUFFIX << "; mt++ ) {" << endl;
    ss << "            Isyn = (clampGain" << SUFFIX << "*(Vcmd-V) - V) / accessResistance" << SUFFIX << ";" << endl;
    ss << model.kernel("            ", false, false);
    ss << R"EOF(
        }

        current = Isyn;
        voltage = V;
        t += DT;
    }

    void reset()
    {
        if ( !running )
            return;
        running = false;
        voltage = current = t = 0.0;
    }

    void initialise()
    {
)EOF";
    for ( const StateVariable &v : stateVariables )
        ss << "        " << v.name << " = " << v.initial << ";" << endl;
    ss << endl;
    for ( const AdjustableParam &p : adjustableParams )
        ss << "        " << p.name << " = " << p.initial << ";" << endl;
    ss << "    }" << endl << endl; // End initialise

    // Declarations
    for ( const StateVariable &v : stateVariables )
        ss << "    " << v.type << " " << v.name << ";" << endl;
    ss << endl;
    for ( const AdjustableParam &p : adjustableParams )
        ss << "    " << p.type << " " << p.name << ";" << endl;

    ss << "};" << endl;
    return ss.str();
}

std::string ExperimentLibrary::supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars)
{
    std::stringstream ss;

    ss << "} // break namespace for STL includes:" << endl;

    ss << "#define NVAR " << stateVariables.size() << endl;
    ss << "#define NPARAM " << adjustableParams.size() << endl;
    ss << "#include \"definitions.h\"" << endl;
    ss << "#include \"experimentlibrary.h\"" << endl;
    ss << "#include \"supportcode.cu\"" << endl;
    ss << "#include \"experimentlibrary.cu\"" << endl;
    ss << endl;

    ss << daqCode();
    ss << endl;
    ss << "inline DAQ *createSim() { return new Simulator(); }" << endl;
    ss << "inline void destroySim(DAQ *sim) { delete sim; }" << endl;
    ss << endl;

    ss << "extern \"C\" ExperimentLibrary::Pointers populate(std::vector<StateVariable> &state, "
                                                         << "std::vector<AdjustableParam> &param) {" << endl;
    ss << "    ExperimentLibrary::Pointers pointers;" << endl;
    ss << "    libInit(pointers, " << expd.numCandidates << ");" << endl;
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
