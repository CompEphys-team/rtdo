#include "metamodel.h"
#include "tinyxml2.h"
#include <stdexcept>
#include "modelSpec.h"
#include "global.h"
#include "stringUtils.h"
#include <sstream>
#include <fstream>
#include "cuda_helper.h"
#include <QString>
#include <QRegularExpression>
#include <cstdlib>

#define HH "HH"

MetaModel *genn_target_generator = nullptr;

MetaModel::MetaModel(std::string xmlfile)
{
    tinyxml2::XMLDocument doc;
    if ( doc.LoadFile(xmlfile.c_str()) ) {
        doc.PrintError();
        throw std::runtime_error("Load XML failed with error " + doc.ErrorID() );
    }

    const tinyxml2::XMLElement *model, *el, *sub;

/// Base tag and model name:
/// <model name="MyModel">
///     ...
/// </model>
    model = doc.FirstChildElement("model");
    if ( !model )
        throw std::runtime_error("Invalid model file: No <model> base tag found.");

    _name = model->Attribute("name");
    if ( _name.empty() )
        throw std::runtime_error("Invalid model file: model name not set.");

/// State variables:
/// <variable name="V" value="-63.5">           <!-- refer to as `$(name)` in code sections; value is the initial default -->
///     <tmp name="foo">code section</tmp>      <!-- temporary variable, refer to as `name` in code sections -->
///     <tmp current="1" name="I_Na">code section</tmp>     <!-- Saved for diagnostic purposes during wavegen -->
///     <dYdt>code section</dYdt>               <!-- diff. eqn d(var)/dt. -->
/// </variable>
    bool hasV = false;
    for ( el = model->FirstChildElement("variable"); el; el = el->NextSiblingElement("variable") ) {
        StateVariable p(el->Attribute("name"), el->FirstChildElement("dYdt")->GetText());
        el->QueryDoubleAttribute("value", &p.initial);
        for ( sub = el->FirstChildElement("tmp"); sub; sub = sub->NextSiblingElement("tmp") ) {
            Variable tmp(sub->Attribute("name"), sub->GetText());
            bool isCur;
            if ( sub->QueryBoolAttribute("current", &isCur) == tinyxml2::XML_SUCCESS && isCur ) {
                currents.push_back(tmp);
            }
            p.tmp.push_back(tmp);
        }

        if ( p.name == "V" ) {
            hasV = true;
            _baseV = p.initial;
            stateVariables.insert(stateVariables.begin(), p);
        } else {
            stateVariables.push_back(p);
        }
    }
    if ( !hasV ) {
        throw std::runtime_error("Error: Model fails to define a voltage variable V.");
    }

/// Fixed parameters:
/// <parameter name="m_VSlope" value="-21.3" />     <!-- refer to as `$(name)`. Value can not change at runtime. -->
    for ( el = model->FirstChildElement("parameter"); el; el = el->NextSiblingElement("parameter") ) {
        Variable p(el->Attribute("name"));
        el->QueryDoubleAttribute("value", &p.initial);
        _params.push_back(p);
    }

/// Adjustable parameters:
/// <adjustableParam name="gNa" value="7.2">    <!-- refer to as `$(name)`. Value is the initial default. -->
///     <range min="5" max="10" />              <!-- Permissible value range, edges inclusive -->
///     <perturbation rate="0.2" type="*|+|multiplicative|additive (default)" />
///                 <!-- Learning rate for this parameter (max change for +/additive, stddev for */multiplicative) -->
///     <wavegen permutations="2 (default)" distribution="normal(default)|uniform" standarddev="1.0" />
///                 <!-- Number of permutations during wavegen. Normal distribution is random around the default value,
///                     uniform distribution is an even spread across the value range. The default standard deviation
///                     is 5 times the perturbation rate.
///                     The default value is always used and does not count towards the total. -->
/// </adjustableParam>
    for ( el = model->FirstChildElement("adjustableParam"); el; el = el->NextSiblingElement("adjustableParam") ) {
        AdjustableParam p(el->Attribute("name"));
        el->QueryDoubleAttribute("value", &p.initial);
        if ( (sub = el->FirstChildElement("range")) ) {
            sub->QueryDoubleAttribute("min", &p.min);
            sub->QueryDoubleAttribute("max", &p.max);
        }
        if ( (sub = el->FirstChildElement("perturbation")) ) {
            sub->QueryDoubleAttribute("rate", &p.sigma);
            std::string ptype = sub->Attribute("type");
            p.multiplicative = !ptype.compare("*") || !ptype.compare("multiplicative");
        }
        p.wgPermutations = 2;
        p.wgNormal = true;
        p.wgSD = 5*p.sigma;
        if ( (sub = el->FirstChildElement("wavegen")) ) {
            sub->QueryIntAttribute("permutations", &p.wgPermutations);
            if ( (p.wgNormal = !sub->Attribute("distribution", "uniform")) )
                sub->QueryDoubleAttribute("standarddev", &p.wgSD);
        }
        adjustableParams.push_back(p);
    }
}

bool MetaModel::createModules(string directory)
{
    putenv("GENN_PATH=" LOCAL_GENN_PATH);
    _dir = directory;
    cfg.type = ModuleType::Wavegen;
    if ( !createModule() )
        return false;
    cfg.type = ModuleType::Experiment;
    if ( !createModule() )
        return false;
    return true;
}

bool MetaModel::createModule()
{
    genn_target_generator = this;
    std::string arg1 = std::string("generating ") + name() + " in";
    char *argv[2] = {const_cast<char*>(arg1.c_str()), const_cast<char*>(_dir.c_str())};
    if ( generateAll(2, argv) ) {
        std::cerr << "Code generation failed." << std::endl;
        return false;
    }

    std::string dir = _dir + "/" + name() + "_CODE";
    std::ofstream makefile(dir + "/Makefile", std::ios_base::app);
    makefile << endl;
    makefile << "runner.so: runner.o" << endl;
    makefile << "\t$(CXX) -o $@ $< -shared" << endl;

    std::stringstream cmd;
    cmd << "cd " << dir << " && GENN_PATH=" << LOCAL_GENN_PATH << " make runner.so";
    if ( system(cmd.str().c_str()) ) {
        std::cerr << "Code compile failed." << std::endl;
        return false;
    }
    return true;
}

void MetaModel::generateExperimentCode(NNmodel &m, neuronModel &n,
                                             std::vector<double> &fixedParamIni,
                                             std::vector<double> &variableIni)
{
    std::vector<Variable> globals = {
        Variable("simCycles", "", "int"),
        Variable("getErrG", "", "bool"),
        Variable("clampGain"),
        Variable("accessResistance"),
        Variable("VmemG"),
        Variable("ImemG"),
        Variable("VC", "", "bool")
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

    n.supportCode = generateExperimentBridge(globals, vars);

    n.simCode = R"EOF(
scalar mdt = DT/$(simCycles);
for ( unsigned int mt = 0; mt < $(simCycles); mt++ ) {
    if ( $(VC) ) {
        Isyn = ($(clampGain)*($(VmemG)-$(V)) - $(V)) / $(accessResistance);
    } else {
        Isyn = $(ImemG);
    }
)EOF"
    + kernel("    ", true)
    + R"EOF(
}

if ( $(getErrG) ) {
    if ( $(VC) )
        $(err) += abs(Isyn - $(ImemG));
    else
        $(err) += abs($(V) - $(VmemG));
}
)EOF";

    int numModels = nModels.size();
    nModels.push_back(n);
    m.addNeuronPopulation(HH, cfg.npop, numModels, fixedParamIni, variableIni);
}

void MetaModel::generateWavegenCode(NNmodel &m, neuronModel &n,
                                          std::vector<double> &fixedParamIni,
                                          std::vector<double> &variableIni)
{
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

    for ( const Variable &c : currents ) {
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
    while ( (int)adjustableParams.size() + 1 > maxThreadsPerBlock / numGroupsPerBlock )
        numGroupsPerBlock /= 2;
    GENN_PREFERENCES::autoChooseDevice = 0;
    GENN_PREFERENCES::optimiseBlockSize = 0;
    GENN_PREFERENCES::neuronBlockSize = numGroupsPerBlock * (adjustableParams.size() + 1);

    if ( cfg.permute ) {
        numGroups = 1;
        for ( AdjustableParam &p : adjustableParams ) {
            numGroups *= p.wgPermutations + 1;
        }
    } else {
        numGroups = cfg.npop;
    }
    // Round up to nearest multiple of numGroupsPerBlock to achieve full occupancy and regular interleaving:
    numGroups = ((numGroups + numGroupsPerBlock - 1) / numGroupsPerBlock) * numGroupsPerBlock;
    numBlocks = numGroups / numGroupsPerBlock;

    stringstream ss;
    ss << endl << "#ifndef _" << name() << "_neuronFnct_cc" << endl; // No Wavegen on the CPU
    ss << R"EOF(
const int groupID = id % MM_NumGroupsPerBlock;                       // Block-local group id
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
    ss << kernel("    ", true) << endl;
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
            err = abs(Isyn - errShare[groupID]);
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

    n.simCode = ss.str();

    n.supportCode = generateWavegenBridge(globals, vars);

    int numModels = nModels.size();
    nModels.push_back(n);
    m.addNeuronPopulation(HH, numGroups * (adjustableParams.size()+1), numModels, fixedParamIni, variableIni);
}

void MetaModel::generate(NNmodel &m)
{
    if ( !GeNNReady )
        initGeNN();

    neuronModel n;
    std::vector<double> fixedParamIni;
    std::vector<double> variableIni;

    for ( const StateVariable &v : stateVariables ) {
        n.varNames.push_back(v.name);
        n.varTypes.push_back(v.type);
        variableIni.push_back(v.initial);
    }
    for ( const AdjustableParam &p : adjustableParams ) {
        n.varNames.push_back(p.name);
        n.varTypes.push_back(p.type);
        variableIni.push_back(p.initial);
    }
    for ( const Variable &p : _params ) {
        n.pNames.push_back(p.name);
        fixedParamIni.push_back(p.initial);
    }

#ifdef DEBUG
    GENN_PREFERENCES::debugCode = true;
    GENN_PREFERENCES::optimizeCode = false;
#else
    GENN_PREFERENCES::debugCode = false;
    GENN_PREFERENCES::optimizeCode = true;
#endif
    GENN_PREFERENCES::optimiseBlockSize = 1;
    GENN_PREFERENCES::userNvccFlags = "-std c++11 -Xcompiler \"-fPIC\" -I" CORE_INCLUDE_PATH;

    m.setDT(cfg.dt);
    m.setName(name());
#ifdef USEDOUBLE
    m.setPrecision(GENN_DOUBLE);
#else
    m.setPrecision(GENN_FLOAT);
#endif

    switch ( cfg.type ) {
    case ModuleType::Experiment: generateExperimentCode(m, n, fixedParamIni, variableIni); break;
    case ModuleType::Wavegen:    generateWavegenCode   (m, n, fixedParamIni, variableIni); break;
    }

    m.finalize();
    m.resetKernel = GENN_FLAGS::calcSynapses; // i.e., never.

}

string MetaModel::name() const
{
    switch ( cfg.type ) {
    case ModuleType::Experiment:
        return _name + "_experiment";
        break;
    case ModuleType::Wavegen:
        return _name + "_wavegen";
        break;
    default:
        return _name + "_no_such_type";
    }
}

// Runge-Kutta k variable names
static inline std::string k(std::string Y, int i) { return std::string("k") + std::to_string(i) + "__" + Y; }

std::string MetaModel::kernel(const std::string &tab, bool wrapVariables) const
{
    auto wrap = [=](const std::string &name) {
        if ( wrapVariables )
            return std::string("$(") + name + std::string(")");
        else
            return name;
    };
    auto unwrap = [=](const std::string &code) {
        if ( wrapVariables )
            return code;
        else
            return QString::fromStdString(code).replace(QRegularExpression("\\$\\((\\w+)\\)"), "\\1").toStdString();
    };

    std::stringstream ss;
    switch ( cfg.method ) {
    case IntegrationMethod::ForwardEuler:
        for ( const StateVariable &v : stateVariables ) {
            ss << tab << v.type << " ddt__" << v.name << ";" << endl;
            ss << tab << "{" << endl;
            for ( const Variable &t : v.tmp ) {
                ss << tab << tab << t.type << " " << t.name << " = " << unwrap(t.code) << ";" << endl;
                if ( isCurrent(t) )
                    ss << tab << tab << wrap(t.name) << " = " << t.name << ";" << endl;
            }
            ss << tab << tab << "ddt__" << v.name << " = " << unwrap(v.code) << ";" << endl;
            ss << tab << "}" << endl;
        }
        for ( const StateVariable &v : stateVariables )
            ss << tab << wrap(v.name) << " += ddt__" << v.name << " * mdt;" << endl;
        break;
    case IntegrationMethod::RungeKutta4:
        // Y0:
        for ( const StateVariable &v : stateVariables ) {
            ss << tab << v.type << " Y0__" << v.name << " = " << wrap(v.name) << ";" << endl;
        }

        for ( int i = 1; i < 5; i++ ) {
            ss << endl;
            ss << tab << "// k_i = dYdt(Y_(i-1)), i = " << i << ":" << endl;
            for ( const StateVariable &v : stateVariables ) {
                ss << tab << v.type << " " << k(v.name, i) << ";" << endl;
                ss << tab << "{" << endl;
                for ( const Variable &t : v.tmp ) {
                    ss << tab << tab << t.type << " " << t.name << " = " << unwrap(t.code) << ";" << endl;
                    if ( i == 1 && isCurrent(t) )
                        ss << tab << tab << wrap(t.name) << " = " << t.name << ";" << endl;
                }
                ss << tab << tab << k(v.name, i) << " = " << unwrap(v.code) << ";" << endl;
                ss << tab << "}" << endl;
            }
            ss << endl;

            ss << tab << "// Y_i = Y0 + k_i * {h/2, h/2, h, ...}, i = " << i << ":" << endl;
            for ( const StateVariable &v : stateVariables ) {
                if ( i == 1 || i == 2 ) {
                    ss << tab << wrap(v.name) << " = Y0__" << v.name << " + " << k(v.name, i) << " * mdt * 0.5;" << endl;
                } else if ( i == 3 ) {
                    ss << tab << wrap(v.name) << " = Y0__" << v.name << " + " << k(v.name, i) << " * mdt;" << endl;
                } else {
                    ss << tab << wrap(v.name) << " = Y0__" << v.name << " + mdt / 6.0 * ("
                       << k(v.name, 1)
                       << " + 2*" << k(v.name, 2)
                       << " + 2*" << k(v.name, 3)
                       << " + " << k(v.name, 4)
                       << ");" << endl;
                }
            }
        }
        break;
    }
    return ss.str();
}

bool MetaModel::isCurrent(const Variable &tmp) const
{
    if ( cfg.type == ModuleType::Wavegen ) {
        for ( const Variable &c : currents ) {
            if ( c.name == tmp.name ) {
                return true;
            }
        }
    }
    return false;
}

std::string MetaModel::generateWavegenBridge(std::vector<Variable> const& globals, std::vector<Variable> const& vars) const
{
    std::stringstream ss;

    ss << "} // break namespace for STL includes:" << endl;

    ss << "#define MM_NumGroupsPerBlock " << numGroupsPerBlock << endl;
    ss << "#define MM_NumModelsPerBlock " << GENN_PREFERENCES::neuronBlockSize << endl;
    ss << "#define MM_NumGroups " << numGroups << endl;
    ss << "#define NVAR " << stateVariables.size() << endl;
    ss << "#define NPARAM " << adjustableParams.size() << endl;
    ss << "#include \"definitions.h\"" << endl;
    ss << "#include \"wavegen_globals.h\"" << endl;
    ss << "#include \"supportcode.cu\"" << endl;
    ss << "#include \"wavegen.cu\"" << endl;
    ss << endl;

    ss << "namespace Wavegen_Global {" << endl;
    ss << "void populate(MetaModel &m) {" << endl;
    ss << "    NPOP = " << numGroups * (adjustableParams.size()+1) << ";" << endl;
    int i = 0;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    m.stateVariables[" << i++ << "].v = " << v.name << HH << ";" << endl;
    }
    i = 0;
    for ( const Variable &c : currents ) {
        ss << "    m.currents[" << i++ << "].v = " << c.name << HH << ";" << endl;
    }
    i = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    m.adjustableParams[" << i++ << "].v = " << p.name << HH << ";" << endl;
    }
    ss << endl;
    for ( const Variable &p : globals ) {
        ss << "    " << p.name << " =& " << p.name << HH << ";" << endl;
    }
    for ( const Variable &v : vars ) {
        ss << "    " << v.name << " = " << v.name << HH << ";" << endl;
        ss << "    d_" << v.name << " = d_" << v.name << HH << ";" << endl;
    }
    ss << "}" << endl;
    ss << "}" << endl;

    ss << endl;
    ss << "namespace " << HH << "_neuron {" << endl;

    return ss.str();
}

std::string MetaModel::generateExperimentBridge(const std::vector<Variable> &globals, const std::vector<Variable> &vars) const
{
    std::stringstream ss;

    ss << "} // break namespace for STL includes:" << endl;

    ss << "#define NVAR " << stateVariables.size() << endl;
    ss << "#define NPARAM " << adjustableParams.size() << endl;
    ss << "#include \"definitions.h\"" << endl;
    ss << "#include \"experiment_globals.h\"" << endl;
    ss << "#include \"daq.h\"" << endl;
    ss << "#include \"supportcode.cu\"" << endl;
    ss << "#include \"experiment.cu\"" << endl;
    ss << endl;

    ss << "namespace Experiment_Global {" << endl;
    ss << "void populate(MetaModel &m) {" << endl;
    ss << "    NPOP = " << cfg.npop << ";" << endl;

    int i = 0;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    m.stateVariables[" << i++ << "].v = " << v.name << HH << ";" << endl;
    }
    i = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    m.adjustableParams[" << i++ << "].v = " << p.name << HH << ";" << endl;
    }
    ss << endl;
    for ( const Variable &p : globals ) {
        ss << "    " << p.name << " =& " << p.name << HH << ";" << endl;
    }
    for ( const Variable &v : vars ) {
        ss << "    " << v.name << " = " << v.name << HH << ";" << endl;
        ss << "    d_" << v.name << " = d_" << v.name << HH << ";" << endl;
    }
    ss << "}" << endl;
    ss << endl;

    ss << R"EOF(
class Simulator : public DAQ
{
private:
    RunData *r;
    double t;

public:
    Simulator(DAQData *p, RunData *r) : DAQ(p), r(r), t(0.0)
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
        const float mdt = p->dt/r->simCycles;
        double Isyn;
        scalar Vcmd = getCommandVoltage(currentStim, t);
        for ( unsigned int mt = 0; mt < r->simCycles; mt++ ) {
            Isyn = (r->clampGain*(Vcmd-V) - V) / r->accessResistance;

)EOF";
    ss << kernel("            ", false);
    ss << R"EOF(
        }

        current = Isyn;
        voltage = V;
        t += p->dt;
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

    ss << R"EOF(
};

DAQ *createSim(DAQData *p, RunData *r) { return new Simulator(p,r); }
void destroySim(DAQ *sim) { delete sim; }

} // end namespace Experiment_Global
)EOF";

    ss << endl;
    ss << "namespace " << HH << "_neuron {" << endl;

    return ss.str();
}
