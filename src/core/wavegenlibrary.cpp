#include "wavegenlibrary.h"
#include "modelSpec.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include "global.h"
#include "cuda_helper.h"
#include <dlfcn.h>
#include "project.h"

#define SUFFIX "WG"

static WavegenLibrary *_this;
static void redirect(NNmodel &n) { _this->GeNN_modelDefinition(n); }

WavegenLibrary::WavegenLibrary(Project &p, bool compile) :
    project(p),
    model(p.model()),
    stateVariables(model.stateVariables),
    adjustableParams(model.adjustableParams),
    currents(model.currents),
    lib(compile ? compile_and_load() : load()),
    populate((decltype(populate))dlsym(lib, "populate")),
    pointers(populate(*this)),
    simCycles(*(pointers.simCycles)),
    clampGain(*(pointers.clampGain)),
    accessResistance(*(pointers.accessResistance)),
    targetParam(*(pointers.targetParam)),
    settling(*(pointers.settling)),
    getErr(*(pointers.getErr)),
    err(pointers.err),
    waveforms(pointers.waveforms),
    bubbles(pointers.bubbles)
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

void *WavegenLibrary::compile_and_load()
{
    std::string directory = project.dir().toStdString();
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
    return load();
}

void *WavegenLibrary::load()
{
    std::string libfile = project.dir().toStdString() + "/" + model.name(ModuleType::Wavegen) + "_CODE/runner.so";
    dlerror();
    void *libp;
    if ( ! (libp = dlopen(libfile.c_str(), RTLD_NOW)) )
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
        Variable("settling", "", "bool"),
        Variable("nGroupsPerStim", "", "int")
    };
    for ( Variable &p : globals ) {
        n.extraGlobalNeuronKernelParameters.push_back(p.name);
        n.extraGlobalNeuronKernelParameterTypes.push_back(p.type);
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

    // Round up to nearest multiple of numGroupsPerBlock to achieve full occupancy and regular interleaving:
    numGroups = ((project.wgNumGroups() + numGroupsPerBlock - 1) / numGroupsPerBlock) * numGroupsPerBlock;
    project.setWgNumGroups(numGroups);
    numBlocks = numGroups / numGroupsPerBlock;
    numModels = numGroups * (model.adjustableParams.size() + 1);

    n.simCode = simCode();
    n.supportCode = supportCode(globals, {});

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
const scalar mdt = DT/$(simCycles);
scalar gathErr = 0.;
const Stimulation stim = dd_waveforms[group];
Stimulation::Step nextStep;
Bubble bestBubble = {-1,0,0}, currentBubble = {-1,0,0};
t = 0.;
scalar Vcmd = getStep(nextStep, stim, t, mdt);
for ( unsigned int mt = 0; t < stim.duration; mt++ ) {
    t = mt*mdt;
    if ( t >= nextStep.t )
        Vcmd = getStep(nextStep, stim, t, mdt);
    else if ( nextStep.ramp )
        Vcmd += nextStep.V;
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
                gathErr += err * mdt;
            errShare[paramID*MM_NumGroupsPerBlock + groupID] = err;
        }
        __syncthreads();

        // Collect statistics for target param
        if ( paramID == $(targetParam) ) {
            scalar value = 0.;
            for ( int i = 1; i < NPARAM+1; i++ ) {
                value += errShare[i*MM_NumGroupsPerBlock + groupID];
                if ( i == paramID )
                    continue;
            }
            value = fitnessPartial(err, value/NPARAM);
            if ( $(nGroupsPerStim) == 1 )
                extendBubble(bestBubble, currentBubble, value, mt);
            else if ( $(nGroupsPerStim) <= MM_NumGroupsPerBlock ) {
                value = warpReduceSum(value, $(nGroupsPerStim))/$(nGroupsPerStim);
                if ( groupID % $(nGroupsPerStim) == 0 )
                    extendBubble(bestBubble, currentBubble, value, mt);
            } else {
                value = warpReduceSum(value, MM_NumGroupsPerBlock);
                if ( groupID == 0 )
                    dd_parfitBlock[int(group/MM_NumGroupsPerBlock) + mt * int(MM_NumGroups / MM_NumGroupsPerBlock)] = value;
            }
        }
    }
} // end for mt

if ( $(getErr) && $(targetParam) < 0 ) {
    dd_err[id] = gathErr;
}

if ( paramID == $(targetParam) ) {
    if ( bestBubble.cycles )
        bestBubble.value /= bestBubble.cycles;
    if ( $(nGroupsPerStim) == 1 )
        dd_bubbles[group] = bestBubble.cycles ? bestBubble : Bubble {0,0,0};
    else if ( $(nGroupsPerStim) <= MM_NumGroupsPerBlock && groupID % $(nGroupsPerStim) == 0 )
        dd_bubbles[group/$(nGroupsPerStim)] = bestBubble.cycles ? bestBubble : Bubble {0,0,0};
}

if ( !$(settling) )
    return;

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
    ss << "#define MM_NumBlocks " << numBlocks << endl;
    ss << "#define MM_NumModels " << numModels << endl;
    ss << "#define NVAR " << stateVariables.size() << endl;
    ss << "#define NPARAM " << adjustableParams.size() << endl;
    ss << "#include \"definitions.h\"" << endl;
    ss << "#include \"wavegenlibrary.h\"" << endl;
    ss << "#include \"../core/supportcode.cpp\"" << endl;
    ss << "#include \"wavegenlibrary.cu\"" << endl;
    ss << endl;

    ss << "extern \"C\" WavegenLibrary::Pointers populate(WavegenLibrary &lib) {" << endl;
    ss << "    WavegenLibrary::Pointers pointers;" << endl;
    ss << "    libInit(pointers, MM_NumGroups, MM_NumModels);" << endl;

    // Re-set num* indicators for lazy loading
    ss << "    lib.numGroupsPerBlock = MM_NumGroupsPerBlock;" << endl;
    ss << "    lib.numModelsPerBlock = MM_NumModelsPerBlock;" << endl;
    ss << "    lib.numGroups = MM_NumGroups;" << endl;
    ss << "    lib.numBlocks = MM_NumBlocks;" << endl;
    ss << "    lib.numModels = MM_NumModels;" << endl;

    int i = 0;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    lib.stateVariables[" << i++ << "].v = " << v.name << SUFFIX << ";" << endl;
    }
    i = 0;
    for ( const Variable &c : currents ) {
        ss << "    lib.currents[" << i++ << "].v = " << c.name << SUFFIX << ";" << endl;
    }
    i = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    lib.adjustableParams[" << i++ << "].v = " << p.name << SUFFIX << ";" << endl;
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
    ss << "    pointers.step =& stepTimeGPU;" << endl;
    ss << "    pointers.reset =& initialize;" << endl;
    ss << "    pointers.generateBubbles =& generateBubbles;" << endl;
    ss << "    return pointers;" << endl;
    ss << "}" << endl;

    ss << endl;
    ss << "namespace " << SUFFIX << "_neuron {" << endl;

    return ss.str();
}

void WavegenLibrary::generateBubbles(scalar duration)
{
    unsigned int nSamples = duration / (project.dt() / simCycles);
    pointers.generateBubbles(nSamples, nStim, pointers);
}

void WavegenLibrary::setRunData(RunData rund)
{
    clampGain = rund.clampGain;
    accessResistance = rund.accessResistance;
    simCycles = rund.simCycles;
}
