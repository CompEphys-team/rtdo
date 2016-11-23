#include "metamodel.h"
#include "tinyxml2.h"
#include <stdexcept>
#include "modelSpec.h"
#include "global.h"
#include "stringUtils.h"
#include <sstream>

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
///     <wavegen permutations="2 (default)" distribution="normal|uniform (default)" standarddev="0.2" />
///                 <!-- Number of permutations during wavegen. Normal distribution is random around the default value,
///                     uniform distribution is an even spread across the value range.
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
        p.wgNormal = false;
        p.wgSD = p.sigma;
        if ( (sub = el->FirstChildElement("wavegen")) ) {
            sub->QueryIntAttribute("permutations", &p.wgPermutations);
            if ( (p.wgNormal = sub->Attribute("distribution", "normal")) )
                sub->QueryDoubleAttribute("standarddev", &p.wgSD);
        }
        adjustableParams.push_back(p);
    }
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

    n.supportCode = bridge(globals, vars);

    n.simCode = R"EOF(
scalar mdt = DT/$(simCycles);
for ( unsigned int mt = 0; mt < $(simCycles); mt++ ) {
    if ( $(VC) ) {
        Isyn = ($(clampGain)*($(VmemG)-$(V)) - $(V)) / $(accessResistance);
    } else {
        Isyn = $(ImemG);
    }
)EOF"
    + kernel("    ")
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
        Variable("final", "", "bool")
    };
    for ( Variable &p : globals ) {
        n.extraGlobalNeuronKernelParameters.push_back(p.name);
        n.extraGlobalNeuronKernelParameterTypes.push_back(p.type);
    }

    std::vector<Variable> vars = {
        Variable("Vmem"),
        Variable("err"),
        Variable("getErr", "", "bool"),
        Variable("bubbles", "", "int"), // Number of bubbles

        // Totals
        Variable("cyclesWon", "", "int"), // Number of cycles won (= target param err > other param errs)
        Variable("wonByAbs"), // Absolute distance to the next param err down (= target err - next err)
        Variable("wonByRel"), // Relative distance to the next param err down (= (target err - next err)/next err)
        Variable("wonOverMean"), // Relative distance to mean param err

        // Current winning streak ("bubble")
        Variable("cCyclesWon", "", "int"),
        Variable("cWonByAbs"),
        Variable("cWonByRel"),
        Variable("cWonOverMean"),

        // Longest bubble
        Variable("bCyclesWon", "", "int"),
        Variable("bCyclesWonT"), // Time at end
        Variable("bCyclesWonA"), // Absolute distance to next param err within this bubble
        Variable("bCyclesWonR"), // Relative distance to next param err within this bubble
        Variable("bCyclesWonM"), // Relative distance to mean param err within this bubble

        // Greatest absolute distance bubble
        Variable("bWonByAbs"),
        Variable("bWonByAbsT"),
        Variable("bWonByAbsC", "", "int"), // Number of cycles within bubble
        Variable("bWonByAbsR"),
        Variable("bWonByAbsM"),

        // Greatest relative distance to next bubble
        Variable("bWonByRel"),
        Variable("bWonByRelT"),
        Variable("bWonByRelC", "", "int"),
        Variable("bWonByRelA"),
        Variable("bWonByRelM"),

        // Greatest relative distance to mean bubble
        Variable("bWonOverMean"),
        Variable("bWonOverMeanT"),
        Variable("bWonOverMeanC", "", "int"),
        Variable("bWonOverMeanA"),
        Variable("bWonOverMeanR")
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

    stringstream ss;
    ss << R"EOF(
scalar mdt = DT/$(simCycles);
for ( unsigned int mt = 0; mt < $(simCycles); mt++ ) {
    Isyn = ($(clampGain)*($(Vmem)-$(V)) - $(V)) / $(accessResistance);
)EOF";
    ss << kernel("    ") << endl;
    ss << "#ifndef _" << name() << "_neuronFnct_cc" << endl; // Don't compile this part in calcNeuronsCPU
    ss <<   R"EOF(
    __shared__ double errShare[NPARAM+1];
    __shared__ double IsynShare[NPARAM+1];
    if ( $(getErr) ) {
        IsynShare[threadIdx.x] = Isyn;
        __syncthreads();

        // Get deviation from the base model
        scalar err = abs(Isyn-IsynShare[0]);
        $(err) += err * mdt;
        errShare[threadIdx.x] = err;
        __syncthreads();

        if ( threadIdx.x == $(targetParam) ) {
            scalar next = 0.;
            scalar total = 0.;
            for ( int i = 1; i < NPARAM+1; i++ ) {
                total += errShare[i];
                if ( i == threadIdx.x )
                    continue;
                if ( errShare[i] > err ) {
                    next = -1.f;
                    break;
                }
                if ( errShare[i] > next )
                    next = errShare[i];
            }
            if ( (next < 0. && $(cCyclesWon)) || ($(final) && mt == $(simCycles)-1 && next >= 0.) ) {
            //    a bubble has just ended     or  this is the final cycle and a bubble is still open
            // ==> Close the bubble, collect stats
                $(bubbles)++;
                if ( $(cCyclesWon) > $(bCyclesWon) ) {
                    $(bCyclesWon) = $(cCyclesWon);
                    $(bCyclesWonT) = t + mt*mdt;
                    $(bCyclesWonA) = $(cWonByAbs);
                    $(bCyclesWonR) = $(cWonByRel);
                    $(bCyclesWonM) = $(cWonOverMean);
                }
                if ( $(cWonByAbs) > $(bWonByAbs) ) {
                    $(bWonByAbs) = $(cWonByAbs);
                    $(bWonByAbsT) = t + mt*mdt;
                    $(bWonByAbsC) = $(cCyclesWon);
                    $(bWonByAbsR) = $(cWonByRel);
                    $(bWonByAbsM) = $(cWonOverMean);
                }
                if ( $(cWonByRel) > $(bWonByRel) ) {
                    $(bWonByRel) = $(cWonByRel);
                    $(bWonByRelT) = t + mt*mdt;
                    $(bWonByRelC) = $(cCyclesWon);
                    $(bWonByRelA) = $(cWonByAbs);
                    $(bWonByRelM) = $(cWonOverMean);
                }
                if ( $(cWonOverMean) > $(bWonOverMean) ) {
                    $(bWonOverMean) = $(cWonOverMean);
                    $(bWonOverMeanT) = t + mt*mdt;
                    $(bWonOverMeanC) = $(cCyclesWon);
                    $(bWonOverMeanA) = $(cWonByAbs);
                    $(bWonOverMeanR) = $(cWonByRel);
                }
                $(cCyclesWon) = 0;
                $(cWonByAbs) = 0.;
                $(cWonByRel) = 0.;
                $(cWonOverMean) = 0.;
            } else if ( next >= 0. ) { // Process open bubble
                $(cyclesWon)++;
                $(cCyclesWon)++;
                $(wonByAbs) += err;
                $(cWonByAbs) += err;
                {
                    scalar rel = 1 - err / (total/NPARAM);
                    $(wonOverMean) += rel;
                    $(cWonOverMean) += rel;
                }
                if ( next > 0. ) {
                    scalar rel = 1 - err / next;
                    $(wonByRel) += rel;
                    $(cWonByRel) += rel;
                }
            }
        }
    }
#endif
}
)EOF";

    n.simCode = ss.str();

    if ( cfg.permute ) {
        cfg.npop = 1;
        for ( AdjustableParam &p : adjustableParams ) {
            cfg.npop *= p.wgPermutations + 1;
        }
    }

    n.supportCode = bridge(globals, vars);

    int numModels = nModels.size();
    nModels.push_back(n);
    m.addNeuronPopulation(HH, cfg.npop * (adjustableParams.size()+1), numModels, fixedParamIni, variableIni);

    GENN_PREFERENCES::optimiseBlockSize = 0;
    GENN_PREFERENCES::neuronBlockSize = adjustableParams.size() + 1;
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

std::string MetaModel::kernel(const std::string &tab) const
{
    std::stringstream ss;
    switch ( cfg.method ) {
    case IntegrationMethod::ForwardEuler:
        for ( const StateVariable &v : stateVariables ) {
            ss << tab << v.type << " ddt__" << v.name << ";" << endl;
            ss << tab << "{" << endl;
            for ( const Variable &t : v.tmp ) {
                ss << tab << tab << t.type << " " << t.name << " = " << t.code << ";" << endl;
                if ( isCurrent(t) )
                    ss << tab << tab << "$(" << t.name << ") = " << t.name << ";" << endl;
            }
            ss << tab << tab << "ddt__" << v.name << " = " << v.code << ";" << endl;
            ss << tab << "}" << endl;
        }
        for ( const StateVariable &v : stateVariables )
            ss << tab << "$(" << v.name << ") += ddt__" << v.name << " * mdt;" << endl;
        break;
    case IntegrationMethod::RungeKutta4:
        // Y0:
        for ( const StateVariable &v : stateVariables ) {
            ss << tab << v.type << " Y0__" << v.name << " = $("  << v.name << ");" << endl;
        }

        for ( int i = 1; i < 5; i++ ) {
            ss << endl;
            ss << tab << "// k_i = dYdt(Y_(i-1)), i = " << i << ":" << endl;
            for ( const StateVariable &v : stateVariables ) {
                ss << tab << v.type << " " << k(v.name, i) << ";" << endl;
                ss << tab << "{" << endl;
                for ( const Variable &t : v.tmp ) {
                    ss << tab << tab << t.type << " " << t.name << " = " << t.code << ";" << endl;
                    if ( i == 1 && isCurrent(t) )
                        ss << tab << tab << "$(" << t.name << ") = " << t.name << ";" << endl;
                }
                ss << tab << tab << k(v.name, i) << " = " << v.code << ";" << endl;
                ss << tab << "}" << endl;
            }
            ss << endl;

            ss << tab << "// Y_i = Y0 + k_i * {h/2, h/2, h, ...}, i = " << i << ":" << endl;
            for ( const StateVariable &v : stateVariables ) {
                if ( i == 1 || i == 2 ) {
                    ss << tab << "$(" << v.name << ") = Y0__" << v.name << " + " << k(v.name, i) << " * mdt * 0.5;" << endl;
                } else if ( i == 3 ) {
                    ss << tab << "$(" << v.name << ") = Y0__" << v.name << " + " << k(v.name, i) << " * mdt;" << endl;
                } else {
                    ss << tab << "$(" << v.name << ") = Y0__" << v.name << " + mdt / 6.0 * ("
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

std::string MetaModel::bridge(std::vector<Variable> const& globals, std::vector<Variable> const& vars) const
{
    std::stringstream ss;

    ss << "} // break namespace for STL includes:" << endl;
    ss << "#include \"kernelhelper.h\"" << endl;
    ss << "#include \"definitions.h\"" << endl;
    ss << "#define NVAR " << stateVariables.size() << endl;
    ss << "#define NPARAM " << adjustableParams.size() << endl;
    ss << endl;

    ss << "void populate(MetaModel &m) {" << endl;
    if ( cfg.type == ModuleType::Experiment )
        ss << "    GeNN_Bridge::NPOP = " << cfg.npop << ";" << endl;
    else
        ss << "    GeNN_Bridge::NPOP = " << cfg.npop * (adjustableParams.size()+1) << ";" << endl;
    int i = 0;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    m.stateVariables[" << i++ << "].v = " << v.name << HH << ";" << endl;
    }
    i = 0;
    for ( const Variable &c : currents ) {
        if ( !isCurrent(c) )
            break;
        ss << "    m.currents[" << i++ << "].v = " << c.name << HH << ";" << endl;
    }
    i = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    m.adjustableParams[" << i++ << "].v = " << p.name << HH << ";" << endl;
    }
    ss << endl;
    for ( const Variable &p : globals ) {
        ss << "    GeNN_Bridge::" << p.name << " =& " << p.name << HH << ";" << endl;
    }
    for ( const Variable &v : vars ) {
        ss << "    GeNN_Bridge::" << v.name << " = " << v.name << HH << ";" << endl;
        ss << "    GeNN_Bridge::d_" << v.name << " = d_" << v.name << HH << ";" << endl;
    }
    ss << "}" << endl;

    ss << endl;
    ss << "namespace " << HH << "_neuron {" << endl;

    return ss.str();
}
