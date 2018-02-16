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

MetaModel::MetaModel(const Project &p, std::string file) :
    project(p)
{
    if ( file.empty() )
        file = project.modelfile().toStdString();

    tinyxml2::XMLDocument doc;
    if ( doc.LoadFile(file.c_str()) ) {
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
///     <range min="0" max="1" />               <!-- Optional: Permissible value range, inclusive of bounds -->
///     <tolerance>1e-6</tolerance>             <!-- Optional: RKF45 error tolerance. Defaults to 1e-3. -->
/// </variable>
    bool hasV = false;
    for ( el = model->FirstChildElement("variable"); el; el = el->NextSiblingElement("variable") ) {
        StateVariable p(el->Attribute("name"), el->FirstChildElement("dYdt")->GetText());
        el->QueryDoubleAttribute("value", &p.initial);
        if ( (sub = el->FirstChildElement("range")) ) {
            sub->QueryDoubleAttribute("min", &p.min);
            sub->QueryDoubleAttribute("max", &p.max);
            using std::swap;
            if ( p.min > p.max )
                swap(p.min, p.max);
        }
        if ( (sub = el->FirstChildElement("tolerance")) && sub->DoubleText() > 0 )
            p.tolerance = sub->DoubleText();
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
            p.adjustedSigma = p.sigma;
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

neuronModel MetaModel::generate(NNmodel &m, std::vector<double> &fixedParamIni, std::vector<double> &variableIni) const
{
    if ( !GeNNReady )
        initGeNN();

    neuronModel n;

    for ( const StateVariable &v : stateVariables ) {
        n.varNames.push_back(v.name);
        n.varTypes.push_back(v.type);
        variableIni.push_back(v.initial);
    }
    n.varNames.push_back("meta_hP");
    n.varTypes.push_back("scalar");
    variableIni.push_back(project.dt()/10);
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
    GENN_PREFERENCES::userNvccFlags = "-std c++11 -Xcompiler \"-fPIC\" -I" CORE_INCLUDE_PATH " -I" CORE_INCLUDE_PATH "/../../lib/randutils";

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
    case ModuleType::Profiler:
        return _name + "_profiler";
    default:
        return _name + "_no_such_type";
    }
}

std::string MetaModel::resolveCode(const std::string &code) const
{
    QString qcode = QString::fromStdString(code);
    for ( const StateVariable &v : stateVariables ) {
        QString sub = "%1";
        if ( v.name == "t" || v.name == "params" || v.name == "clamp" )
            sub = "this->%1";
        qcode.replace(QString("$(%1)").arg(QString::fromStdString(v.name)),
                      sub.arg(QString::fromStdString(v.name)));
    }
    for ( const AdjustableParam &p : adjustableParams )
        qcode.replace(QString("$(%1)").arg(QString::fromStdString(p.name)),
                      QString("params.%1").arg(QString::fromStdString(p.name)));
    for ( const Variable &p : _params )
        qcode.replace(QString("$(%1)").arg(QString::fromStdString(p.name)),
                      QString::number(p.initial, 'g', 10));
    return qcode.toStdString();
}

std::string MetaModel::structDeclarations() const
{
    std::stringstream ss;
    ss.setf(std::ios_base::scientific, std::ios_base::floatfield);
    ss.precision(10);

    ss << "struct Parameters {" << endl;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    " << p.type << " " << p.name << " = " << p.initial << ";" << endl;
    }
    ss << "};" << endl;
    ss << endl;

    ss << "struct State {" << endl;
    for ( const StateVariable &v : stateVariables ) {
        ss << "    " << v.type << " " << v.name << " = " << v.initial << ";" << endl;
    }
    ss << endl;

    ss << "    __host__ __device__ State operator+(const State &state) const {" << endl;
    ss << "        State retval;" << endl;
    for ( const StateVariable &v : stateVariables ) {
        ss << "        retval." << v.name << " = this->" << v.name << " + state." << v.name << ";" << endl;
    }
    ss << "        return retval;" << endl;
    ss << "    }" << endl;
    ss << endl;

    ss << "    __host__ __device__ State operator*(scalar factor) const {" << endl;
    ss << "        State retval;" << endl;
    for ( const StateVariable &v : stateVariables ) {
        ss << "        retval." << v.name << " = this->" << v.name << " * factor;" << endl;
    }
    ss << "        return retval;" << endl;
    ss << "    }" << endl;

    ss << "    __host__ __device__ State state__f(scalar t, const Parameters &params, scalar Isyn) const {" << endl;
    ss << "        State retval;" << endl;
    for ( const StateVariable &v : stateVariables ) {
        ss << "        " << "{" << endl;
        for ( const Variable &t : v.tmp ) {
            ss << "            " << t.type << " " << t.name << " = " << resolveCode(t.code) << ";" << endl;
        }
        ss << "            retval." << v.name << " = " << resolveCode(v.code) << ";" << endl;
        ss << "        }" << endl;
    }
    ss << "        return retval;" << endl;
    ss << "    }" << endl;

    ss << "    __host__ __device__ inline State state__f(scalar t, const Parameters &params, const ClampParameters &clamp) const {" << endl;
    ss << "        return state__f(t, params, clamp.getCurrent(t, this->V));" << endl;
    ss << "    }" << endl;
    ss << endl;

    ss << "    __host__ __device__ scalar state__delta(scalar h, bool &success) const {" << endl;
    ss << "        scalar err, maxErr = 0;" << endl;
    for ( const StateVariable &v : stateVariables ) {
        ss << "        err = fabs(" << v.name << ") / (h * " << v.tolerance << ");" << endl;
        ss << "        if ( err > maxErr ) maxErr = err;" << endl;
    }
    ss << "        success = (maxErr <= 1);" << endl;
    ss << "        if ( maxErr <= 0.03125 ) return 2;" << endl;
    ss << "        return 0.84 * sqrt(sqrt(1/maxErr));" << endl;
    ss << "    }" << endl;
    ss << endl;

    ss << "    __host__ __device__ void state__limit() {" << endl;
    for ( const StateVariable &v : stateVariables ) {
        ss << "        if ( " << v.name << " > " << v.max << " ) " << v.name << " = " << v.max << ";" << endl;
        ss << "        if ( " << v.name << " < " << v.min << " ) " << v.name << " = " << v.min << ";" << endl;
    }
    ss << "    }" << endl;

    ss << "};" << endl;

    return ss.str();
}

std::string MetaModel::supportCode() const
{
    return structDeclarations();
}

std::string MetaModel::populateStructs(std::string paramPre, std::string paramPost, std::string rundPre, std::string rundPost) const
{
    std::stringstream ss;
    ss << "State state;" << endl;
    for ( const Variable &v : stateVariables ) {
        ss << "    state." << v.name << " = " << paramPre << v.name << paramPost << ";" << endl;
    }
    ss << endl;

    ss << "Parameters params;" << endl;
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "    params." << p.name << " = " << paramPre << p.name << paramPost << ";" << endl;
    }
    ss << endl;

    ss << "ClampParameters clamp;" << endl;
    ss << "    clamp.clampGain = " << rundPre << "clampGain" << rundPost << ";" << endl;
    ss << "    clamp.accessResistance = " << rundPre << "accessResistance" << rundPost << ";" << endl;
    // Note, params.VClamp0 and params.dVClamp to be populated by caller.

    return ss.str();
}

std::string MetaModel::extractState(std::string pre, std::string post) const
{
    std::stringstream ss;
    for ( const Variable &v : stateVariables ) {
        ss << pre << v.name << post << " = state." << v.name << ";" << endl;
    }
    return ss.str();
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
    ss.setf(std::ios_base::scientific, std::ios_base::floatfield);
    ss.precision(10);

    if ( !wrapVariables )
        for ( const Variable &p : _params )
            ss << tab << "constexpr " << p.type << " " << p.name << " = " << p.initial << ";" << endl;

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
        for ( const StateVariable &v : stateVariables ) {
            ss << tab << wrap(v.name) << " += ddt__" << v.name << " * mdt;" << endl;
            if ( v.max > v.min ) {
                ss << tab << "if ( " << wrap(v.name) << " < " << v.min << " ) "
                   << wrap(v.name) << " = " << v.min << ";" << endl;
                ss << tab << "else if ( " << wrap(v.name) << " > " << v.max << " ) "
                   << wrap(v.name) << " = " << v.max << ";" << endl;
            }
        }
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
                    if ( v.max > v.min ) {
                        ss << tab << "if ( " << wrap(v.name) << " < " << v.min << " ) "
                           << wrap(v.name) << " = " << v.min << ";" << endl;
                        ss << tab << "else if ( " << wrap(v.name) << " > " << v.max << " ) "
                           << wrap(v.name) << " = " << v.max << ";" << endl;
                    }
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

string MetaModel::daqCode(int ordinal) const
{
    std::stringstream ss;
    ss << endl << "#define Simulator_numbered Simulator_" << ordinal << endl;
    ss << "class Simulator_numbered : public DAQ {" << endl;
    ss << "private:" << endl;

    if ( ordinal > 1 ) // Ordinal 1 is the original model, use the public structs
        ss << structDeclarations() << endl;

    ss << R"EOF(
    struct CacheStruct {
        CacheStruct(Stimulation s, bool VC, double sDt, int extraSamples) :
            _stim(s),
            _VC(VC),
            _voltage(s.duration/sDt + extraSamples),
            _current(s.duration/sDt + extraSamples)
        {}
        Stimulation _stim;
        bool _VC;
        std::vector<scalar> _voltage;
        std::vector<scalar> _current;
        State state;
    };

    std::list<CacheStruct> cache;
    std::list<CacheStruct>::iterator currentCacheEntry;
    size_t iT;
    bool useRealism;

    scalar tStart, sDt, mdt, meta_hP, noiseI[3], noiseExp, noiseA, noiseDefaultH;
    bool caching = false, generating = false;
    int skipSamples;

    State state;
    Parameters params;
    ClampParameters clamp;

public:
    Simulator_numbered(Session &session, bool useRealism) : DAQ(session), useRealism(useRealism)
    {
        if ( p.simd.paramSet == 1 ) {
)EOF";
    for ( const AdjustableParam &p : adjustableParams ) {
        ss << "            params." << p.name << " = RNG.uniform<" << p.type << ">(" << p.min << ", " << p.max << ");" << endl;
    }
    ss << "        } else if ( p.simd.paramSet == 2 ) {" << endl;
    for ( size_t i = 0; i < adjustableParams.size(); i++ ) {
        ss << "            params." << adjustableParams[i].name << " = p.simd.paramValues[" << i << "];" << endl;
    }
    ss << R"EOF(
        }
    }

    ~Simulator_numbered() {}

    int throttledFor(const Stimulation &) { return 0; }

    void getNoiseParams(scalar h, scalar &Exp, scalar &A)
    {
      if ( p.simd.noiseTau > 0 ) { // Ornstein-Uhlenbeck
          scalar noiseD = 2 * p.simd.noiseStd * p.simd.noiseStd / p.simd.noiseTau; // nA^2/ms
          Exp = std::exp(-h/p.simd.noiseTau);
          A = std::sqrt(noiseD * p.simd.noiseTau * (1 - Exp*Exp) / 2);
      } else { // White noise
          Exp = 0;
          A = p.simd.noiseStd;
      }
    }

    scalar getNextNoise(scalar I_t, scalar h)
    {
        scalar Exp, A;
        if ( h == noiseDefaultH || p.simd.noiseTau == 0 ) {
            Exp = noiseExp;
            A = noiseA;
        } else {
            getNoiseParams(h, Exp, A);
        }
        return I_t * Exp + A * RNG.variate<scalar>(0, 1); // I(t+h) = I0 + (I(t)-I0)*exp(-h/tau) + A*X(0,1), I0 = 0
    }

    void run(Stimulation s, double settle)
    {
        currentStim = s;
        iT = 0;
        tStart = 0;
        sDt = DT;
        int extraSamples = 0;

        clamp.clampGain = rund.clampGain;
        clamp.accessResistance = rund.accessResistance;

        outputResolution = 0;

        if ( useRealism ) {
            sDt = samplingDt();
            if ( p.filter.active ) {
                extraSamples = p.filter.width;
                tStart = -int(p.filter.width/2) * sDt;
            }
            if ( p.simd.noise ) {
                mdt = sDt/rund.simCycles;
                noiseI[2] = p.simd.noiseStd * RNG.variate<scalar>(0,1);
                noiseDefaultH = mdt/2; // Fixed-step RK4 has two evenly spaced evaluations per step
                getNoiseParams(noiseDefaultH, noiseExp, noiseA);
                outputResolution = sDt;
            }
        }

        samplesRemaining = s.duration/sDt + extraSamples;
        meta_hP = sDt/rund.simCycles;

        if ( settle > 0 ) {
            Stimulation settlingStim;
            settlingStim.baseV = s.baseV;
            settlingStim.duration = settle;
            for ( currentCacheEntry = cache.begin(); currentCacheEntry != cache.end(); ++currentCacheEntry )
                if ( currentCacheEntry->_stim == settlingStim && currentCacheEntry->_VC == VC )
                    break;
            if ( currentCacheEntry == cache.end() ) {
                clamp.VClamp0 = s.baseV;
                clamp.dVClamp = 0;
                RKF45(0, settle, meta_hP, settle, meta_hP, state, params, clamp);
                currentCacheEntry = cache.insert(currentCacheEntry, CacheStruct(settlingStim, VC, settle, 0));
                currentCacheEntry->state = state;
                currentCacheEntry->_current[0] = clamp.getCurrent(0, state.V);
                currentCacheEntry->_voltage[0] = state.V;
            } else {
                state = currentCacheEntry->state;
            }

            current = currentCacheEntry->_current[0];
            voltage = currentCacheEntry->_voltage[0];

            skipSamples = settle/sDt;
            samplesRemaining += skipSamples;
        }

        if ( useRealism && p.simd.noise ) {
            caching = false;
            generating = true;
        } else {
            caching = true;
            // Check if requested stimulation has been used before
            for ( currentCacheEntry = cache.begin(); currentCacheEntry != cache.end(); ++currentCacheEntry )
                if ( currentCacheEntry->_stim == s && currentCacheEntry->_VC == VC )
                    break;
            if ( currentCacheEntry == cache.end() ) {
                generating = true;
                currentCacheEntry = cache.insert(currentCacheEntry, CacheStruct(s, VC, sDt, extraSamples));
            } else {
                generating = false;
                state = currentCacheEntry->state;
            }
        }
    }

    void next()
    {
        if ( skipSamples-- > 0 )
            return;

        if ( generating ) {
            scalar t = tStart + iT*sDt;
            scalar VClamp0_2, dVClamp_2, t_2 = getCommandVoltages(currentStim, t, sDt, clamp.VClamp0, clamp.dVClamp, VClamp0_2, dVClamp_2);

            voltage = state.V;
            current = clamp.getCurrent(t, state.V);

            if ( useRealism && p.simd.noise ) {
                scalar pureIsyn = current;
                current += noiseI[2];
                for ( unsigned int mt = 0; mt < rund.simCycles; mt++ ) {
                    t = tStart + iT*sDt + mt*mdt;
//                    if ( t > t_2 ) {
//                        clamp.VClamp0 = VClamp0_2;
//                        clamp.dVClamp = dVClamp_2;
//                        t_2 += sDt; // ignore for the rest of the loop
//                    }

                    // Keep noise constant across evaluations for the same time point
                    noiseI[0] = noiseI[2];
                    noiseI[1] = getNextNoise(noiseI[0], mdt/2);
                    noiseI[2] = getNextNoise(noiseI[1], mdt/2);
//                    scalar pureIsyn = clamp.getCurrent(t, state.V);

                    // RK4, fixed step:
                    constexpr scalar half = 1/2.;
                    constexpr scalar third = 1/3.;
                    constexpr scalar sixth = 1/6.;
                    State k1 = state.state__f(t, params, pureIsyn + noiseI[0]) * mdt;
                    State k2 = State(state + k1*half).state__f(t + mdt*half, params, pureIsyn + noiseI[1]) * mdt;
                    State k3 = State(state + k2*half).state__f(t + mdt*half, params, pureIsyn + noiseI[1]) * mdt;
                    State k4 = State(state + k3).state__f(t + mdt, params, pureIsyn + noiseI[2]) * mdt;
                    state = state + k1*sixth + k2*third + k3*third + k4*sixth;
                    state.state__limit();
                }
            } else {
                if ( t_2 == 0 ) {
                    RKF45(t, t + sDt, sDt/rund.simCycles, sDt, meta_hP, state, params, clamp);
                } else {
                    RKF45(t, t_2, sDt/rund.simCycles, sDt, meta_hP, state, params, clamp);
                    clamp.VClamp0 = VClamp0_2;
                    clamp.dVClamp = dVClamp_2;
                    RKF45(t_2, t + sDt, sDt/rund.simCycles, sDt, meta_hP, state, params, clamp);
                }
            }

            if ( caching ) {
                currentCacheEntry->_voltage[iT] = voltage;
                currentCacheEntry->_current[iT] = current;
            }
        } else {
            current = currentCacheEntry->_current[iT];
            voltage = currentCacheEntry->_voltage[iT];
        }
        ++iT;
        --samplesRemaining;
    }

    void reset()
    {
        iT = 0;
        voltage = current = 0.0;
        if ( caching && generating )
            currentCacheEntry->state = state;
        caching = generating = false;
    }

)EOF";

    ss << "    void setAdjustableParam(size_t idx, double value)" << endl;
    ss << "    {" << endl;
    ss << "        switch ( idx ) {" << endl;
    for ( size_t i = 0; i < adjustableParams.size(); i++ )
        ss << "        case " << i << ": params." << adjustableParams[i].name << " = value; break;" << endl;
    ss << "        }" << endl;
    ss << "        cache.clear();" << endl;
    ss << "    }" << endl << endl;

    ss << "    double getAdjustableParam(size_t idx)" << endl;
    ss << "    {" << endl;
    ss << "        switch ( idx ) {" << endl;
    for ( size_t i = 0; i < adjustableParams.size(); i++ )
        ss << "        case " << i << ": return params." << adjustableParams[i].name << ";" << endl;
    ss << "        }" << endl;
    ss << "        return 0;" << endl;
    ss << "    }" << endl << endl;

    ss << "};" << endl;
    ss << "#undef Simulator_numbered" << endl;
    return ss.str();
}
