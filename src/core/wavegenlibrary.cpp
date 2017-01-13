#include "wavegenlibrary.h"
#include "modelSpec.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include "global.h"
#include "cuda_helper.h"
#include <dlfcn.h>

#define SUFFIX "WG"

static WavegenLibrary *_this;
static void redirect(NNmodel &n) { _this->GeNN_modelDefinition(n); }

WavegenLibrary::WavegenLibrary(MetaModel &model, const std::string &directory, const WavegenData &searchd) :
    searchd(searchd),
    model(model),
    stateVariables(model.stateVariables),
    adjustableParams(model.adjustableParams),
    currents(model.currents),
    lib(loadLibrary(directory)),
    populate((decltype(populate))dlsym(lib, "populate")),
    pointers(populate(stateVariables, adjustableParams, currents)),
    t(*(pointers.t)),
    iT(*(pointers.iT)),
    simCycles(*(pointers.simCycles)),
    clampGain(*(pointers.clampGain)),
    accessResistance(*(pointers.accessResistance)),
    targetParam(*(pointers.targetParam)),
    final(*(pointers.final)),
    getErr(*(pointers.getErr)),
    err(pointers.err),
    waveforms(pointers.waveforms),
    wavestats(pointers.wavestats)
{

}

WavegenLibrary::~WavegenLibrary()
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

void *WavegenLibrary::loadLibrary(const std::string &directory)
{
    // Generate code
    _this = this;
    MetaModel::modelDef = redirect;
    std::string name = model.name(ModuleType::Wavegen);
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

void WavegenLibrary::GeNN_modelDefinition(NNmodel &nn)
{
    std::vector<double> fixedParamIni, variableIni;
    neuronModel n = model.generate(nn, fixedParamIni, variableIni);

    std::vector<Variable> globals = {
        Variable("simCycles", "", "int"),
        Variable("clampGain"),
        Variable("accessResistance"),
        Variable("targetParam", "", "int"),
        Variable("getErr", "", "bool"),
        Variable("final", "", "bool")
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

    for ( const Variable &c : model.currents ) {
        n.varNames.push_back(c.name);
        n.varTypes.push_back(c.type);
        variableIni.push_back(0.0);
    }

    // Allocate model groups such that target param models go into a single warp:
    // i.e., model groups are interleaved with stride (numGroupsPerBlock = warpsize/2^n, n>=0),
    // which means that models detuned in a given parameter are warp-aligned.
    cudaDeviceProp prop;
    int deviceCount, maxThreadsPerBlock = 0;
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
    for (int device = 0; device < deviceCount; device++) {
        CHECK_CUDA_ERRORS(cudaSetDevice(device));
        CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&prop, device));
        if ( prop.maxThreadsPerBlock > maxThreadsPerBlock ) {
            // somewhat arbitrarily pick a device with the most threads per block - a.k.a no support for multiGPU.
            maxThreadsPerBlock = prop.maxThreadsPerBlock;
            numGroupsPerBlock = prop.warpSize;
            GENN_PREFERENCES::defaultDevice = device;
        }
    }
    while ( (int)model.adjustableParams.size() + 1 > maxThreadsPerBlock / numGroupsPerBlock )
        numGroupsPerBlock /= 2;
    numModelsPerBlock = numGroupsPerBlock * (model.adjustableParams.size() + 1);
    GENN_PREFERENCES::autoChooseDevice = 0;
    GENN_PREFERENCES::optimiseBlockSize = 0;
    GENN_PREFERENCES::neuronBlockSize = numModelsPerBlock;

    if ( searchd.permute ) {
        numGroups = 1;
        for ( AdjustableParam &p : model.adjustableParams ) {
            numGroups *= p.wgPermutations + 1;
        }
    } else {
        numGroups = searchd.numWavesPerEpoch;
    }
    // Round up to nearest multiple of numGroupsPerBlock to achieve full occupancy and regular interleaving:
    numGroups = ((numGroups + numGroupsPerBlock - 1) / numGroupsPerBlock) * numGroupsPerBlock;
    numBlocks = numGroups / numGroupsPerBlock;
    numModels = numGroups * (model.adjustableParams.size() + 1);

    n.simCode = simCode();
    n.supportCode = supportCode(globals, vars);

    int idx = nModels.size();
    nModels.push_back(n);
    nn.setName(model.name(ModuleType::Wavegen));
    nn.addNeuronPopulation(SUFFIX, numModels, idx, fixedParamIni, variableIni);

    nn.finalize();
}

std::string WavegenLibrary::simCode()
{
    stringstream ss;
    ss << endl << "#ifndef _" << model.name(ModuleType::Wavegen) << "_neuronFnct_cc" << endl; // No Wavegen on the CPU
    ss << R"EOF(
const int groupID = id % MM_NumGroupsPerBlock;                                  // Block-local group id
const int group = groupID + (id/MM_NumModelsPerBlock) * MM_NumGroupsPerBlock;   // Global group id
const int paramID = (id % MM_NumModelsPerBlock) / MM_NumGroupsPerBlock;
WaveStats *stats;
if ( $(getErr) && paramID == $(targetParam) ) // Preload for @fn processStats - other threads don't need this
    stats =& dd_wavestats[group];
scalar Vcmd = getCommandVoltage(dd_waveforms[group], t);

scalar mdt = DT/$(simCycles);
for ( unsigned int mt = 0; mt < $(simCycles); mt++ ) {
    Isyn = ($(clampGain)*(Vcmd-$(V)) - $(V)) / $(accessResistance);
)EOF";
    ss << model.kernel("    ", true, true) << endl;
    ss <<   R"EOF(
    if ( $(getErr) ) {
        __shared__ double errShare[MM_NumModelsPerBlock];
        scalar err;

        // Make base model (paramID==0) Isyn available
        if ( !paramID )
            errShare[groupID] = Isyn;
        __syncthreads();

        // Get deviation from the base model
        if ( paramID ) {
            err = fabs(Isyn - errShare[groupID]);
            if ( $(targetParam) < 0 )
                $(err) += err * mdt;
            errShare[paramID*MM_NumGroupsPerBlock + groupID] = err;
        }
        __syncthreads();

        // Collect statistics for target param
        if ( paramID && paramID == $(targetParam) ) {
            scalar total = 0.;
            for ( int i = 1; i < NPARAM+1; i++ ) {
                total += errShare[i*MM_NumGroupsPerBlock + groupID];
                if ( i == paramID )
                    continue;
            }
            processStats(err, total / NPARAM, t + mt*mdt, *stats, $(final) && mt == $(simCycles)-1 );
        }
    }
} // end for mt

if ( $(getErr) && paramID == $(targetParam) )
    dd_wavestats[group] = *stats;

#else
Isyn += 0.; // Squelch Wunused in neuronFnct.cc
#endif
)EOF";

    return ss.str();
}

std::string WavegenLibrary::supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars)
{
    std::stringstream ss;

    ss << "} // break namespace for STL includes:" << endl;

    ss << "#define MM_NumGroupsPerBlock " << numGroupsPerBlock << endl;
    ss << "#define MM_NumModelsPerBlock " << numModelsPerBlock << endl;
    ss << "#define MM_NumGroups " << numGroups << endl;
    ss << "#define NVAR " << stateVariables.size() << endl;
    ss << "#define NPARAM " << adjustableParams.size() << endl;
    ss << "#include \"definitions.h\"" << endl;
    ss << "#include \"wavegenlibrary.h\"" << endl;
    ss << "#include \"supportcode.cu\"" << endl;
    ss << "#include \"wavegenlibrary.cu\"" << endl;
    ss << endl;

    ss << "extern \"C\" WavegenLibrary::Pointers populate(std::vector<StateVariable> &state, "
                                                      << "std::vector<AdjustableParam> &param, "
                                                      << "std::vector<Variable> &current) {" << endl;
    ss << "    WavegenLibrary::Pointers pointers;" << endl;
    ss << "    libInit(pointers, " << numGroups << ", " << numModels << ");" << endl;
    int i = 0;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    state[" << i++ << "].v = " << v.name << SUFFIX << ";" << endl;
    }
    i = 0;
    for ( const Variable &c : currents ) {
        ss << "    current[" << i++ << "].v = " << c.name << SUFFIX << ";" << endl;
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
    ss << "    return pointers;" << endl;
    ss << "}" << endl;

    ss << endl;
    ss << "namespace " << SUFFIX << "_neuron {" << endl;

    return ss.str();
}
