/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#include "xmlmodel.h"
#include "tinyxml.h"
#include <fstream>
#include <exception>
#include <iomanip>
#include "config.h"

#define POPNAME "HH"
#define CUTMARK "///--- Cut simulator here ---///"

using namespace std;

XMLModel::XMLModel()
{
}

XMLModel::XMLModel(string filename)
{
    if ( !load(filename) )
        throw(exception());
}

bool XMLModel::load(string filename)
{
    TiXmlDocument doc;
    doc.SetCondenseWhiteSpace(false);
    doc.LoadFile(filename);

    TiXmlHandle hDoc(&doc);
    TiXmlElement *el;
    TiXmlElement *sub;

    el = hDoc.FirstChild().Element();
    if ( !el )
        return false;
    TiXmlHandle hRoot(el);

    _name.clear();
    _name = el->Attribute("name");

    code.clear();
    if ( (el = hRoot.FirstChild("code").Element()) )
        code = string(el->GetText());

    precision = 6;
    genn_double = false;
    if ( (el = hRoot.FirstChild("precision").Element()) ) {
        el->QueryIntAttribute("digits", &precision);
        genn_double = !string("double").compare(el->Attribute("scalar"));
    }

    _vars.clear();
    bool hasV = false;
    for ( el = hRoot.FirstChild("variable").Element(); el; el = el->NextSiblingElement("variable") ) {
        struct param p;
        p.name = el->Attribute("name");
        p.type = el->Attribute("type");
        el->QueryDoubleAttribute("value", &p.initial);
        _vars.push_back(p);

        if ( !p.name.compare("V") ) {
            hasV = true;
            _baseV = p.initial;
        }
    }
    if ( !hasV ) {
        cerr << "Model fails to define a voltage variable V." << endl;
        return false;
    }

    _params.clear();
    for ( el = hRoot.FirstChild("parameter").Element(); el; el = el->NextSiblingElement("parameter") ) {
        struct param p;
        p.name = el->Attribute("name");
        el->QueryDoubleAttribute("value", &p.initial);
        _params.push_back(p);
    }

    _adjustableParams.clear();
    for ( el = hRoot.FirstChild("adjustableParam").Element(); el; el = el->NextSiblingElement("adjustableParam") ) {
        struct param p;
        p.name = el->Attribute("name");
        p.type = el->Attribute("type");
        el->QueryDoubleAttribute("value", &p.initial);
        if ( (sub = el->FirstChildElement("range")) ) {
            sub->QueryDoubleAttribute("min", &p.min);
            sub->QueryDoubleAttribute("max", &p.max);
        }
        if ( (sub = el->FirstChildElement("perturbation")) ) {
            sub->QueryDoubleAttribute("rate", &p.sigma);
            string ptype = sub->Attribute("type");
            p.multiplicative = !ptype.compare("*") || !ptype.compare("multiplicative");
        }
        _adjustableParams.push_back(p);
    }

    _currents.clear();
    for ( el = hRoot.FirstChild("current").Element(); el; el = el->NextSiblingElement("current") ) {
        struct Current c;
        c.name = el->Attribute("name");
        c.code = string(el->GetText());
        _currents.push_back(c);
    }

    return true;
}

std::string XMLModel::generateDefinition(XMLModel::outputType type, int npop, string path, bool single)
{
    string modelname = name(type, single);

    if ( single ) {
        npop = 1;
    }

    if ( path.back() != '/' )
        path += "/";
    ofstream of(path + modelname + string(".cc"));

    of << setprecision(precision) << scientific;

    of << "#include \"modelSpec.h\"" << endl;
    of << "#include \"global.h\"" << endl;
    of << "#include \"stringUtils.h\"" << endl;

    switch ( type ) {
    case VClamp:
        gendef_VClamp(of, npop, modelname);
        break;
    case WaveGen:
        gendef_wavegen(of, npop, modelname);
        break;
    case WaveGenNoveltySearch:
        gendef_wavegenNS(of, npop, modelname);
        break;
    }

    return modelname;
}

void XMLModel::gendef_variables(ofstream &of, int nExtraVars, bool addCurrents)
{
    // Initial values

    of << "double variableIni[" << to_string(_vars.size() + _adjustableParams.size() + nExtraVars) << "] = {" << endl;
    for ( vector<param>::iterator it = _vars.begin(); it != _vars.end(); ++it )
        of << "  " << it->initial << "," << endl;
    for ( vector<param>::iterator it = _adjustableParams.begin(); it != _adjustableParams.end(); ++it )
        of << "  " << it->initial << "," << endl;
    of << "};" << endl;

    of << "double fixedParamIni[" << to_string(_params.size() + 1) << "] = {" << endl;
    for ( vector<param>::iterator it = _params.begin(); it != _params.end(); ++it )
        of << "  " << it->initial << "," << endl;
    of << "  " << config->model.cycles << endl;
    of << "};" << endl;

    of << "double *aParamIni = variableIni + NVAR;" << endl;

    of << "double aParamRange[NPARAM*2] = {" << endl;
    for ( vector<param>::iterator it = _adjustableParams.begin(); it != _adjustableParams.end(); ++it )
        of << "  " << it->min << ", " << it->max << "," << endl;
    of << "};" << endl;

    of << "double aParamSigma[NPARAM] = {" << endl;
    for ( vector<param>::iterator it = _adjustableParams.begin(); it != _adjustableParams.end(); ++it )
        of << "  " << it->sigma << "," << endl;
    of << "};" << endl;

    of << "bool aParamPMult[NPARAM] = {" << endl;
    for ( vector<param>::iterator it = _adjustableParams.begin(); it != _adjustableParams.end(); ++it )
        of << "  " << (it->multiplicative ? "true" : "false") << "," << endl;
    of << "};" << endl;

    of << endl;

    // Iterables

    of << "#ifdef DEFINITIONS_H" << endl;
        of << "scalar *mvar[NVAR];" << endl;
        of << "scalar *d_mvar[NVAR];" << endl;
        of << "scalar *mparam[NPARAM];" << endl;
        of << "scalar *d_mparam[NPARAM];" << endl;
        if ( addCurrents ) {
            of << "scalar *mcurrents[NCURRENTS];" << endl;
            of << "scalar *d_mcurrents[NCURRENTS];" << endl;
        }

        // Iterable initialisation

        of << "void rtdo_init_bridge() {" << endl;
            int i = 0;
            for ( vector<param>::iterator it = _vars.begin(); it != _vars.end(); ++it, ++i ) {
                of << "mvar[" << i << "] = " << it->name << POPNAME << ";" << endl;
                of << "d_mvar[" << i << "] = d_" << it->name << POPNAME << ";" << endl;
            }
            i = 0;
            for ( vector<param>::iterator it = _adjustableParams.begin(); it != _adjustableParams.end(); ++it, ++i ) {
                of << "mparam[" << i << "] = " << it->name << POPNAME << ";" << endl;
                of << "d_mparam[" << i << "] = d_" << it->name << POPNAME << ";" << endl;
            }
            i = 0;
            if ( addCurrents ) {
                for ( vector<Current>::iterator it = _currents.begin(); it != _currents.end(); ++it, ++i ) {
                    of << "mcurrents[" << i << "] = " << it->name << POPNAME << ";" << endl;
                    of << "d_mcurrents[" << i << "] = d_" << it->name << POPNAME << ";" << endl;
                }
            }
        of << "}" << endl;
    of << "#endif" << endl;
    of << endl;
}

void XMLModel::gendef_pushVar(ofstream &of, const param &p)
{
    of << "n.varNames.push_back(\"" << p.name << "\");" << endl;
    of << "n.varTypes.push_back(\"" << p.type << "\");" << endl;
}
void XMLModel::gendef_pushParam(ofstream &of, const param &p)
{
    of << "n.pNames.push_back(\"" << p.name << "\");" << endl;
}
void XMLModel::gendef_pushGlobalParam(ofstream &of, const param &p)
{
    of << "n.extraGlobalNeuronKernelParameters.push_back(\"" << p.name << "\");" << endl;
    of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"" << p.type << "\");" << endl;
}

void XMLModel::gendef_VClamp(ofstream &of, int npop, string modelname)
{
    of << "#define NVAR " << _vars.size() << endl;
    of << "#define NPARAM " << _adjustableParams.size() << endl;
    of << "#define NCURRENTS " << _currents.size() << endl;
    of << "#define NPOP " << npop << endl;
    of << endl;

    gendef_variables(of, 1, false);

    of << "void modelDefinition(NNmodel &model) {" << endl;
    of << "neuronModel n;" << endl;
    of << "initGeNN();" << endl;
    of << "n.varNames.clear();" << endl;
    of << "n.varTypes.clear();" << endl;

    of << endl;
    for ( const param &p : _vars ) {
        gendef_pushVar(of, p);
    }
    of << endl;
    for ( const param &p : _adjustableParams ) {
        gendef_pushVar(of, p);
    }
    of << endl;
    for ( const param &p : _params ) {
        gendef_pushParam(of, p);
    }
    of << endl;

    gendef_pushVar(of, param("err"));

    gendef_pushParam(of, param("simCycles"));

    gendef_pushGlobalParam(of, param("ot"));
    gendef_pushGlobalParam(of, param("clampGain"));
    gendef_pushGlobalParam(of, param("accessResistance"));

    gendef_pushGlobalParam(of, param("ote"));
    gendef_pushGlobalParam(of, param("stepVG"));
    gendef_pushGlobalParam(of, param("IsynG"));
    gendef_pushGlobalParam(of, param("VC", "bool"));

    // Code
    of << "n.simCode = R\"EOF(" << endl;
    for ( Current c : _currents ) {
        of << "scalar " << c.name << ";" << endl;
    }
    of << CUTMARK << endl
       << "unsigned int mt;" << endl
       << "scalar mdt= DT/$(simCycles);" << endl
       << "for (mt=0; mt < $(simCycles); mt++) {" << endl
       << "  if ( $(VC) ) {" << endl
       << "    Isyn = ($(clampGain)*($(stepVG)-$(V)) - $(V)) / $(accessResistance);" << endl
       << "  } else {" << endl
       << "    Isyn = $(IsynG);" << endl
       << "  }" << endl;
    for ( Current c : _currents ) {
        of << "  " << c.name << " = " << c.code << ";" << endl;
    }
    of << code << endl
       << "}" << endl
       << CUTMARK << endl
       << "if ((t > $(ot)) && (t < $(ote))) {" << endl
       << "  $(err) += abs(Isyn-$(IsynG));" << endl
       << "}" << endl
       << ")EOF\";" << endl;
    for ( Current c : _currents ) {
        of << "substitute( n.simCode, tS(\"$(" << c.name << ")\"), tS(\"" << c.name << "\") );" << endl;
    }


#ifdef _DEBUG
        of << "GENN_PREFERENCES::debugCode = true;" << endl;
#else
        of << "GENN_PREFERENCES::optimizeCode = true;" << endl;
#endif

        of << endl;
        of << "n.thresholdConditionCode = (\"false\");" << endl;
        of << "int modelNum = nModels.size();" << endl;
        of << "nModels.push_back(n);" << endl;
        of << "model.setDT(" << to_string(config->io.dt) << ");" << endl;
        of << "model.setName(\"" << modelname << "\");" << endl;
        of << "model.setPrecision(" << (genn_double ? "GENN_DOUBLE" : "GENN_FLOAT") << ");" << endl;
        of << "model.addNeuronPopulation(\"" << POPNAME << "\", " << npop << ", modelNum, fixedParamIni, variableIni);" << endl;
        of << "model.finalize();" << endl;
        of << "}" << endl;
}

void XMLModel::gendef_wavegen(ofstream &of, int npop, string modelname)
{
    of << "#define NVAR " << _vars.size() << endl;
    of << "#define NPARAM " << _adjustableParams.size() << endl;
    of << "#define NCURRENTS " << _currents.size() << endl;
    of << "#define GAPOP " << npop << endl;
    of << "#define NPOP " << npop*(_adjustableParams.size() + 1) << endl;
    of << endl;

    gendef_variables(of, _currents.size() + 3, true);

    of << "void modelDefinition(NNmodel &model) {" << endl;
    of << "neuronModel n;" << endl;
    of << "initGeNN();" << endl;
    of << "n.varNames.clear();" << endl;
    of << "n.varTypes.clear();" << endl;

    of << endl;
    for ( const param &p : _vars ) {
        gendef_pushVar(of, p);
    }
    of << endl;
    for ( const param &p : _adjustableParams ) {
        gendef_pushVar(of, p);
    }
    of << endl;
    for ( const Current &c : _currents ) {
        gendef_pushVar(of, param(c.name));
    }
    of << endl;
    for ( const param &p : _params ) {
        gendef_pushParam(of, p);
    }
    of << endl;

    gendef_pushVar(of, param("err"));
    gendef_pushVar(of, param("stepVG"));
    gendef_pushVar(of, param("ote"));

    gendef_pushParam(of, param("simCycles"));

    gendef_pushGlobalParam(of, param("ot"));
    gendef_pushGlobalParam(of, param("clampGain"));
    gendef_pushGlobalParam(of, param("accessResistance"));

    // Code
    of << "n.simCode = R\"EOF(" << endl
       << CUTMARK << endl
       << "unsigned int mt;" << endl
       << "scalar mdt= DT/$(simCycles);" << endl
       << "for (mt=0; mt < $(simCycles); mt++) {" << endl
       << "  Isyn = ($(clampGain)*($(stepVG)-$(V)) - $(V)) / $(accessResistance);" << endl;
    for ( Current c : _currents ) {
        of << "  $(" << c.name << ") = " << c.code << ";" << endl;
    }
    of << code << endl
       << "#ifndef _" << modelname << "_neuronFnct_cc" << endl // Don't compile this part in calcNeuronsCPU
       << "#ifdef _" << modelname << "_neuronKrnl_cc" << endl // Also, don't compile in simulateSingleNeuron
       << "  __shared__ double IsynShare[" + to_string(_adjustableParams.size() + 1) + "];" << endl
       << "  if ((t > $(ot)) && (t < $(ote))) {" << endl
       << "    IsynShare[threadIdx.x] = Isyn;" << endl
       << "    __syncthreads();" << endl
       << "    $(err)+= abs(Isyn-IsynShare[0]) * mdt * DT;" << endl
       << "  }" << endl
       << "#endif" << endl
       << "#endif" << endl
       << "}" << endl
       << CUTMARK << endl
       << ")EOF\";" << endl;


    // Global preferences
    of << "GENN_PREFERENCES::optimiseBlockSize = 0;" << endl;
    of << "GENN_PREFERENCES::neuronBlockSize = " << to_string(_adjustableParams.size() + 1) << ";" << endl;
    of << "GENN_PREFERENCES::synapseBlockSize = 1;" << endl;
    of << "GENN_PREFERENCES::learningBlockSize = 1;" << endl;
#ifdef _DEBUG
        of << "GENN_PREFERENCES::debugCode = true;" << endl;
#else
        of << "GENN_PREFERENCES::optimizeCode = true;" << endl;
#endif

    of << endl;
    of << "n.thresholdConditionCode = (\"false\");" << endl;
    of << "int modelNum = nModels.size();" << endl;
    of << "nModels.push_back(n);" << endl;
    of << "model.setDT(" << to_string(config->io.dt) << ");" << endl;
    of << "model.setName(\"" << modelname << "\");" << endl;
    of << "model.setPrecision(" << (genn_double ? "GENN_DOUBLE" : "GENN_FLOAT") << ");" << endl;
    of << "model.addNeuronPopulation(\"" << POPNAME << "\", " << npop*(_adjustableParams.size() + 1)
        << ", modelNum, fixedParamIni, variableIni);" << endl;
    of << "model.finalize();" << endl;
    of << "}" << endl;
}

void XMLModel::gendef_wavegenNS(ofstream &of, int npop, string modelname)
{
    of << "#define NVAR " << _vars.size() << endl;
    of << "#define NPARAM " << _adjustableParams.size() << endl;
    of << "#define NCURRENTS " << _currents.size() << endl;
    of << "#define GAPOP " << npop << endl;
    of << "#define NPOP " << npop*(_adjustableParams.size() + 1) << endl;
    of << endl;

    gendef_variables(of, _currents.size() + 18, true);

    of << "void modelDefinition(NNmodel &model) {" << endl;
    of << "neuronModel n;" << endl;
    of << "initGeNN();" << endl;
    of << "n.varNames.clear();" << endl;
    of << "n.varTypes.clear();" << endl;

    of << endl;
    for ( const param &p : _vars ) {
        gendef_pushVar(of, p);
    }
    of << endl;
    for ( const param &p : _adjustableParams ) {
        gendef_pushVar(of, p);
    }
    of << endl;
    for ( const Current &c : _currents ) {
        gendef_pushVar(of, param(c.name));
    }
    of << endl;
    for ( const param &p : _params ) {
        gendef_pushParam(of, p);
    }
    of << endl;

    gendef_pushVar(of, param("err"));
    gendef_pushVar(of, param("stepVG"));

    // Exceed: Windows where the error signal for a parameter exceeds the mean error across all parameters
    // The selected value reflects the greatest aggregate error above the mean (the area between the error and the mean error curves)
    gendef_pushVar(of, param("exceed"));
    gendef_pushVar(of, param("exceedCurrent"));
    gendef_pushVar(of, param("nExceed", "int"));
    gendef_pushVar(of, param("nExceedCurrent", "int"));

    // Best: Windows where the error signal for a parameter exceeds all other parameters' error signal
    // The selected value reflects the longest 'winning streak', rather than the greatest aggregate margin.
    gendef_pushVar(of, param("best"));
    gendef_pushVar(of, param("bestCurrent"));
    gendef_pushVar(of, param("nBest", "int"));
    gendef_pushVar(of, param("nBestCurrent", "int"));

    // Separation: Windows where the error signal for a parameter exceeds all other parameters' error signal
    // The selected value reflects the greatest aggregate separation. See also comment in code generation
    gendef_pushVar(of, param("separation"));
    gendef_pushVar(of, param("sepCurrent"));
    gendef_pushVar(of, param("nSeparation", "int"));
    gendef_pushVar(of, param("nSepCurrent", "int"));
    gendef_pushVar(of, param("sepDecay"));

    // Waveform start/end (for stObservationWindow)
    gendef_pushVar(of, param("tStart"));
    gendef_pushVar(of, param("tStartCurrent"));
    gendef_pushVar(of, param("tEnd"));

    gendef_pushParam(of, param("simCycles"));

    gendef_pushGlobalParam(of, param("ot"));
    gendef_pushGlobalParam(of, param("clampGain"));
    gendef_pushGlobalParam(of, param("accessResistance"));

    // Switches (for stWaveformOptimise)
    gendef_pushGlobalParam(of, param("calcBest", "bool"));
    gendef_pushGlobalParam(of, param("calcExceed", "bool"));
    gendef_pushGlobalParam(of, param("calcSeparation", "bool"));

    // Tolerance
    gendef_pushGlobalParam(of, param("timeTolerance"));
    gendef_pushGlobalParam(of, param("currentTolerance"));
    gendef_pushGlobalParam(of, param("deltaTolerance"));

    gendef_pushGlobalParam(of, param("ote"));
    gendef_pushGlobalParam(of, param("stage", "int"));


    // Code
    of << "n.simCode = R\"EOF(" << endl
       << "#ifndef _" << modelname << "_neuronFnct_cc" << endl
       << "scalar IsynErr = 0;" << endl
       << "bool gatherError = ( $(stage) == " << stDetuneAdjust << " || ((t > $(ot)) && (t < $(ote))) );" << endl
       << "#endif" << endl

       // Internal loop
       << CUTMARK << endl
       << "unsigned int mt;" << endl
       << "scalar mdt= DT/$(simCycles);" << endl
       << "for (mt=0; mt < $(simCycles); mt++) {" << endl
       << "  Isyn = ($(clampGain)*($(stepVG)-$(V)) - $(V)) / $(accessResistance);" << endl;
    for ( Current c : _currents ) {
      of << "  $(" << c.name << ") = " << c.code << ";" << endl;
    }
    of << code << endl
       << "#ifndef _" << modelname << "_neuronFnct_cc" << endl // Don't compile this part in calcNeuronsCPU
       << "#ifdef _" << modelname << "_neuronKrnl_cc" << endl // Also, don't compile in simulateSingleNeuron
       << "  __shared__ double IsynShare[" + to_string(_adjustableParams.size() + 1) + "];" << endl
       << "  if ( gatherError || ($(stage) > " << stObservationWindow__start
             << " && $(stage) < " << stObservationWindow__end << ") ) {" << endl
       << "    IsynShare[threadIdx.x] = Isyn;" << endl
       << "    __syncthreads();" << endl
       << "    IsynErr += abs(Isyn-IsynShare[0]) * mdt;" << endl
       << "  }" << endl
       << "#endif" << endl
       << "#endif" << endl
       << "}" << endl
       << CUTMARK << endl

       << "#ifndef _" << modelname << "_neuronFnct_cc" << endl

       // Report Isyn (reference) or IsynErr (detuned) for validation
       << "if ( $(stage) > " << stObservationWindow__start << " && $(stage) < " << stObservationWindow__end << " ) {" << endl
       << "  $(err) = (threadIdx.x ? IsynErr : Isyn);" << endl
       << "}" << endl

       // DetuneAdjust stage: Add up this parameter's deviation from tuned model
       << "if ( $(stage) == " << stDetuneAdjust << " ) {" << endl
       << "  $(err) += IsynErr * DT;" << endl

       // Calculate average deviation from tuned model across all parameters
       << "} else if ( gatherError && threadIdx.x > 0 ) {" << endl
       << "  __shared__ scalar IsynErrShare[" << to_string(_adjustableParams.size() + 1) << "];" << endl
       << "  IsynErrShare[threadIdx.x] = IsynErr;" << endl
       << "  __syncthreads();" << endl
       << "  scalar avgIsynErr = 0;" << endl
       << "  scalar maxIsynErr = 0;" << endl
       << "  int runnerUp = 0;"
       << "  for ( int i = 1; i < " << to_string(_adjustableParams.size() + 1) << "; ++i ) {" << endl
       << "    avgIsynErr += IsynErrShare[i];" << endl
       << "    if ( maxIsynErr <= IsynErrShare[i] ) {" << endl
       << "      maxIsynErr = IsynErrShare[i];" << endl
       << "      if ( i != threadIdx.x )  runnerUp = i;" << endl
       << "    }" << endl
       << "  }" << endl
          // Apply current tolerance to all further operations
       << "  if ( IsynErr < $(currentTolerance) ) IsynErr = 0;" << endl
       << "  avgIsynErr /= " << to_string(_adjustableParams.size()) << ";" << endl
       << "  scalar RUratio = IsynErr > 0 ? (IsynErr - IsynErrShare[runnerUp]) / IsynErr : 0;" << endl

       // Calculate $(separation) as the square of the absolute separation from the runner-up,
       // normalised by the absolute separation from the reference.
       // This rewards both good separation from the other parameters, and good (large, robust-to-noise) current deflections
       // Timing tolerance and delta tolerance apply here.
       << "  if ( $(calcSeparation) ) {" << endl
       << "    if ( IsynErr > IsynErrShare[runnerUp] + $(deltaTolerance) ) {" << endl
       << "      if ( $(stage) == " << stObservationWindowSeparation << " && $(nSepCurrent) == 0 )" << endl
       << "        $(tStartCurrent) = t;" << endl
       << "      $(nSepCurrent)++;" << endl
       << "      $(sepCurrent) += RUratio * (IsynErr - IsynErrShare[runnerUp]);" << endl
       << "      if ( $(sepDecay) < $(timeTolerance) )" << endl
       << "        $(sepDecay) += DT;" << endl
       << "      if ( $(sepCurrent) > $(separation) ) {" << endl
       << "        $(nSeparation) = $(nSepCurrent);" << endl
       << "        $(separation) = $(sepCurrent);" << endl
       << "        if ( $(stage) == " << stObservationWindowSeparation << " ) {" << endl
       << "          $(tStart) = $(tStartCurrent);" << endl
       << "          $(tEnd) = t + DT;" << endl
       << "        }" << endl
       << "      }" << endl
       << "    } else {" << endl
       << "      if ( $(sepDecay) > 0 ) {" << endl
       << "        $(sepDecay) -= DT;" << endl
       << "      } else { " << endl
       << "        $(nSepCurrent) = 0;" << endl
       << "        $(sepCurrent) = 0;" << endl
       << "        $(sepDecay) = 0;" << endl
       << "      }" << endl
       << "    }" << endl
       << "  }" << endl

       // Calculate $(best), normalised by the second-greatest error signal
       // Delta tolerance applies here.
       << "  if ( $(calcBest) ) {" << endl
       << "    if ( IsynErr > IsynErrShare[runnerUp] + $(deltaTolerance) ) {" << endl
       << "      if ( $(stage) == " << stObservationWindowBest << " && $(nBestCurrent) == 0 )" << endl
       << "        $(tStartCurrent) = t;" << endl
       << "      $(nBestCurrent)++;" << endl
       << "      $(bestCurrent) += RUratio;" << endl
       << "      if ( $(nBestCurrent) > $(nBest) ) {" << endl
       << "        $(nBest) = $(nBestCurrent);" << endl
       << "        $(best) = $(bestCurrent);" << endl
       << "        if ( $(stage) == " << stObservationWindowBest << " ) {" << endl
       << "          $(tStart) = $(tStartCurrent);" << endl
       << "          $(tEnd) = t + DT;" << endl
       << "        }" << endl
       << "      }" << endl
       << "    } else {" << endl
       << "      $(nBestCurrent) = 0;" << endl
       << "      $(bestCurrent) = 0;" << endl
       << "    }" << endl
       << "  }" << endl

       // NoveltySearch and WaveformOptimise stage:
       // Find the largest positive deviation from the average, in terms of normalised area under the curve
       << "  if ( $(calcExceed) ) {" << endl
       << "    if ( IsynErr > avgIsynErr && avgIsynErr > 0 ) {" << endl
       << "      if ( $(stage) == " << stObservationWindowExceed << " && $(nExceedCurrent) == 0 )" << endl
       << "        $(tStartCurrent) = t;" << endl
       << "      $(nExceedCurrent)++;" << endl
       << "      $(exceedCurrent) += (IsynErr / avgIsynErr) - 1.0;" << endl
       << "      if ( $(exceedCurrent) > $(exceed) ) {" << endl
       << "        $(nExceed) = $(nExceedCurrent);" << endl
       << "        $(exceed) = $(exceedCurrent);" << endl
       << "        if ( $(stage) == " << stObservationWindowExceed << " ) {" << endl
       << "          $(tStart) = $(tStartCurrent);" << endl
       << "          $(tEnd) = t + DT;" << endl
       << "        }" << endl
       << "      }" << endl
       << "    } else {" << endl
       << "      $(nExceedCurrent) = 0;" << endl
       << "      $(exceedCurrent) = 0;" << endl
       << "    }" << endl
       << "  }" << endl
       << "}" << endl
       << "#endif" << endl
       << ")EOF\";" << endl;


    // Global preferences
    of << "GENN_PREFERENCES::optimiseBlockSize = 0;" << endl;
    of << "GENN_PREFERENCES::neuronBlockSize = " << to_string(_adjustableParams.size() + 1) << ";" << endl;
    of << "GENN_PREFERENCES::synapseBlockSize = 1;" << endl;
    of << "GENN_PREFERENCES::learningBlockSize = 1;" << endl;
#ifdef _DEBUG
        of << "GENN_PREFERENCES::debugCode = true;" << endl;
#else
        of << "GENN_PREFERENCES::optimizeCode = true;" << endl;
#endif

        of << endl;
        of << "n.thresholdConditionCode = (\"false\");" << endl;
        of << "int modelNum = nModels.size();" << endl;
        of << "nModels.push_back(n);" << endl;
        of << "model.setDT(" << to_string(config->io.dt) << ");" << endl;
        of << "model.setName(\"" << modelname << "\");" << endl;
        of << "model.setPrecision(" << (genn_double ? "GENN_DOUBLE" : "GENN_FLOAT") << ");" << endl;
        of << "model.addNeuronPopulation(\"" << POPNAME << "\", " << npop*(_adjustableParams.size() + 1)
            << ", modelNum, fixedParamIni, variableIni);" << endl;
        of << "model.finalize();" << endl;
    of << "}" << endl;
}

void XMLModel::generateSimulator(XMLModel::outputType type, string path)
{
    if ( path.back() != '/' )
        path += "/";
    ofstream of(path + name(type) + string(".cc"), ios_base::out | ios_base::app);
    ifstream is(path + name(type) + string("_CODE/neuronFnct.cc"));

    string scalar(genn_double ? "double" : "float");

    of << endl;
    of << "extern \"C\" " << scalar << " simulateSingleNeuron("
       << scalar << " *variables, "
       << scalar << " *adjustableParams, "
       << scalar << " *currents, "
       << scalar << " lstepVG)" << endl;
    of << "{" << endl;
    of << "\t" << scalar << " Isyn = 0;" << endl;

    int i = 0;
    for ( param& v : _vars ) {
        of << "\t" << scalar << " &l" << v.name << " = variables[" << i << "];" << endl;
        ++i;
    }
    i = 0;
    for ( param &p : _adjustableParams ) {
        of << "\t" << scalar << " l" << p.name << " = adjustableParams[" << i << "];" << endl;
        ++i;
    }
    i = 0;
    for ( Current &c : _currents ) {
        if ( type == WaveGen || type == WaveGenNoveltySearch )
            of << "\t" << scalar << " &l" << c.name << " = currents[" << i << "];" << endl;
        else
            of << "\t" << scalar << " &" << c.name << " = currents[" << i << "];" << endl;
        ++i;
    }
    of << endl;

    char buffer[1024];
    bool cut = false;
    while ( is.good() ) {
        is.getline(buffer, 1024);
        if ( !string(buffer).compare(CUTMARK) ) {
            cut = !cut;
        } else if ( cut ) {
            of << buffer << endl;
        }
    }

    of << "\treturn Isyn;" << endl;
    of << "}" << endl;
}

string XMLModel::name(XMLModel::outputType type, bool single) const
{
    string modelname = _name + "_";
    switch ( type ) {
    case VClamp:
        modelname += "vclamp";
        break;
    case WaveGen:
        modelname += "wavegen";
        break;
    case WaveGenNoveltySearch:
        modelname += "wavegenNS";
        break;
    }
    if ( single ) {
        modelname += "_single";
    }
    return modelname;
}
