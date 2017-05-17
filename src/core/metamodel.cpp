#include "metamodel.h"
#include "tinyxml2.h"
#include <stdexcept>
#include "modelSpec.h"
#include "global.h"
#include <sstream>
#include "cuda_helper.h"
#include <QString>
#include <QRegularExpression>
#include "project.h"

void (*MetaModel::modelDef)(NNmodel&);

size_t MetaModel::numLibs = 0;

MetaModel::MetaModel(const Project &p) :
    project(p)
{
    tinyxml2::XMLDocument doc;
    if ( doc.LoadFile(project.modelfile().toStdString().c_str()) ) {
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
///     <tmp current="1" name="I_Na">code section</tmp>     <!-- Saved for diagnostic purposes during wavegen, and available to all state vars -->
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
            using std::swap;
            if ( p.min > p.max )
                swap(p.min, p.max);
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

neuronModel MetaModel::generate(NNmodel &m, std::vector<double> &fixedParamIni, std::vector<double> &variableIni)
{
    if ( !GeNNReady )
        initGeNN();

    neuronModel n;

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

    m.setDT(project.dt());

#ifdef USEDOUBLE
    m.setPrecision(GENN_DOUBLE);
#else
    m.setPrecision(GENN_FLOAT);
#endif

    m.resetKernel = GENN_FLAGS::calcSynapses; // i.e., never.

    return n;

}

std::string MetaModel::name() const
{
    return _name;
}

string MetaModel::name(ModuleType type) const
{
    switch ( type ) {
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

std::string MetaModel::kernel(const std::string &tab, bool wrapVariables, bool defineCurrents) const
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
    switch ( project.method() ) {
    case IntegrationMethod::ForwardEuler:
        // Locally declare currents
        for ( const StateVariable &v : stateVariables ) {
            for ( const Variable &t : v.tmp ) {
                if ( isCurrent(t) ) {
                    ss << tab << t.type << " " << t.name << ";" << endl;
                }
            }
        }
        // Define dYdt for each state variable
        for ( const StateVariable &v : stateVariables ) {
            ss << tab << v.type << " ddt__" << v.name << ";" << endl;
            ss << tab << "{" << endl;
            for ( const Variable &t : v.tmp ) {
                if ( isCurrent(t) ) {
                    ss << tab << tab << t.name << " = " << unwrap(t.code) << ";" << endl;
                    if ( defineCurrents )
                        ss << tab << tab << wrap(t.name) << " = " << t.name << ";" << endl;
                } else {
                    ss << tab << tab << t.type << " " << t.name << " = " << unwrap(t.code) << ";" << endl;
                }

            }
            ss << tab << tab << "ddt__" << v.name << " = " << unwrap(v.code) << ";" << endl;
            ss << tab << "}" << endl;
        }
        // Update state variables
        for ( const StateVariable &v : stateVariables )
            ss << tab << wrap(v.name) << " += ddt__" << v.name << " * mdt;" << endl;
        break;

    case IntegrationMethod::RungeKutta4:
        // Y0:
        for ( const StateVariable &v : stateVariables ) {
            ss << tab << v.type << " Y0__" << v.name << " = " << wrap(v.name) << ";" << endl;
            for ( int i = 1; i < 5; i++ ) {
                ss << tab << v.type << " " << k(v.name, i) << ";" << endl;
            }
        }

        for ( int i = 1; i < 5; i++ ) {
            ss << endl;
            ss << tab << "{ // Begin Runge-Kutta, step " << i << endl;
            // Locally declare currents
            for ( const StateVariable &v : stateVariables ) {
                for ( const Variable &t : v.tmp ) {
                    if ( isCurrent(t) ) {
                        ss << tab << t.type << " " << t.name << ";" << endl;
                    }
                }
            }

            // Define k_i
            ss << tab << "// k_i = dYdt(Y_(i-1)), i = " << i << ":" << endl;
            for ( const StateVariable &v : stateVariables ) {
                ss << tab << "{" << endl;
                for ( const Variable &t : v.tmp ) {
                    if ( isCurrent(t) ) {
                        ss << tab << tab << t.name << " = " << unwrap(t.code) << ";" << endl;
                        if ( defineCurrents && i == 1 )
                            ss << tab << tab << wrap(t.name) << " = " << t.name << ";" << endl;
                    } else {
                        ss << tab << tab << t.type << " " << t.name << " = " << unwrap(t.code) << ";" << endl;
                    }
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
            ss << tab << "} // End Runge-Kutta, step " << i << endl;
        }
        break;
    }
    return ss.str();
}

bool MetaModel::isCurrent(const Variable &tmp) const
{
    for ( const Variable &c : currents ) {
        if ( c.name == tmp.name ) {
            return true;
        }
    }
    return false;
}
