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
    nn.addNeuronPopulation(SUFFIX, project.expNumCandidates(), numModels, fixedParamIni, variableIni);

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
// + std::string("#ifndef _") + model.name(ModuleType::Experiment) + "_neuronFnct_cc\nif(id==1400){ printf(\"Device %2.2f + %2d: %f\t%.1f\t%f\t%f\\n\", t, mt, $(V), $(Vmem), Isyn, $(Imem)); }\n#endif\n"
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
class SimulatorImpl : public DAQ
{
private:
    struct CacheStruct {
        CacheStruct(Stimulation s, double sDt, int extraSamples) :
            _stim(s),
            _voltage(s.duration/sDt + extraSamples),
            _current(s.duration/sDt + extraSamples)
        {}
        Stimulation _stim;
        std::vector<scalar> _voltage;
        std::vector<scalar> _current;
)EOF";
    for ( const StateVariable &v : stateVariables )
        ss << "        " << v.type << " " << v.name << ";" << endl;
    ss << R"EOF(
    };

    std::list<CacheStruct> cache;
    std::list<CacheStruct>::iterator currentCacheEntry;
    size_t currentSample;
    bool useRealism;

public:
    SimulatorImpl(Session &session, bool useRealism) : DAQ(session), useRealism(useRealism)
    {
        initialise();
    }

    ~SimulatorImpl() {}

    int throttledFor(const Stimulation &) { return 0; }

    void run(Stimulation s)
    {
        currentStim = s;
        currentSample = 0;

        double sDt = DT;
        int extraSamples = 0;
        scalar tStart = 0, tEnd = s.duration;

        if ( useRealism ) {
            sDt = samplingDt();
            if ( p.filter.active ) {
                extraSamples = p.filter.width;
                tStart = -int(p.filter.width/2) * sDt;
                tEnd = s.duration - tStart;
            }
        }

        samplesRemaining = s.duration/sDt + extraSamples;

        // Check if requested stimulation has been used before
        for ( currentCacheEntry = cache.begin(); currentCacheEntry != cache.end(); ++currentCacheEntry ) {
            if ( currentCacheEntry->_stim == s ) {
                restoreState();
                return;
            }
        }
        currentCacheEntry = cache.insert(currentCacheEntry, CacheStruct(s, sDt, extraSamples));

        scalar t = tStart;
        unsigned int iT = 0;
        scalar Isyn;
)EOF";
    ss << "        const scalar mdt = sDt/simCycles" << SUFFIX << ";" << endl;
    ss << "        for ( t = tStart; t <= tEnd; ++iT, t = tStart + iT*sDt ) {" << endl;
    ss << "            scalar Vcmd = getCommandVoltage(s, t);" << endl;
    ss << "            for ( unsigned int mt = 0; mt < simCycles" << SUFFIX << "; mt++ ) {" << endl;
    ss << "                Isyn = (clampGain" << SUFFIX << "*(Vcmd-V) - V) / accessResistance" << SUFFIX << ";" << endl;
    ss << model.kernel("                ", false, false);
    ss << R"EOF(
            } // end for mt

            currentCacheEntry->_voltage[iT] = V;
            currentCacheEntry->_current[iT] = Isyn;
        } // end for t

        saveState();
    }

    void next()
    {
        current = currentCacheEntry->_current[currentSample];
        voltage = currentCacheEntry->_voltage[currentSample];
        ++currentSample;
        --samplesRemaining;
    }

    void reset()
    {
        currentSample = 0;
        voltage = current = 0.0;
    }

    void saveState()
    {
)EOF";
    for ( const StateVariable &v : stateVariables )
        ss << "        currentCacheEntry->" << v.name << " = " << v.name << ";" << endl;
    ss << "    }" << endl;
    ss << endl;

    ss << "    void restoreState()" << endl;
    ss << "    {" << endl;
    for ( const StateVariable &v : stateVariables )
        ss << "        " << v.name << " = currentCacheEntry->" << v.name << ";" << endl;
    ss << "    }" << endl;
    ss << endl;

    ss << "    void initialise()" << endl;
    ss << "    {" << endl;
    for ( const StateVariable &v : stateVariables )
        ss << "        " << v.name << " = " << v.initial << ";" << endl;
    ss << "    }" << endl << endl;

    ss << "    void setAdjustableParam(size_t idx, double value)" << endl;
    ss << "    {" << endl;
    ss << "        switch ( idx ) {" << endl;
    for ( size_t i = 0; i < adjustableParams.size(); i++ )
        ss << "        case " << i << ": " << adjustableParams[i].name << " = value; break;" << endl;
    ss << "        }" << endl;
    ss << "    cache.clear();" << endl;
    ss << "    initialise();" << endl;
    ss << "    }" << endl << endl;

    // Declarations
    for ( const StateVariable &v : stateVariables )
        ss << "    " << v.type << " " << v.name << ";" << endl;

    ss << endl;
    for ( const AdjustableParam &p : adjustableParams )
        ss << "    " << p.type << " " << p.name << " = " << p.initial << ";" << endl;

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
    ss << "#include \"../core/supportcode.cpp\"" << endl;
    ss << "#include \"experimentlibrary.cu\"" << endl;
    ss << endl;

    ss << daqCode();
    ss << endl;
    ss << "inline DAQ *createSim(Session &session, bool useRealism) { return new SimulatorImpl(session, useRealism); }" << endl;
    ss << "inline void destroySim(DAQ *sim) { delete sim; }" << endl;
    ss << endl;

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
