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
    for ( el = hRoot.FirstChild("variable").Element(); el; el = el->NextSiblingElement("variable") ) {
        struct param p;
        p.name = el->Attribute("name");
        p.type = el->Attribute("type");
        el->QueryDoubleAttribute("value", &p.initial);
        _vars.push_back(p);
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

    return true;
}

std::string XMLModel::generateDefinition(XMLModel::outputType type, int npop, string path, bool single)
{
    string modelname = name(type, single);

    if ( single ) {
        npop = 1;
    }

    int nExtraVars = 2;

    if ( path.back() != '/' )
        path += "/";
    ofstream of(path + modelname + string(".cc"));

    of << setprecision(precision) << scientific;

    of << "#define NVAR " << _vars.size() + 1 << endl; // +1 for V
    of << "#define NPARAM " << _adjustableParams.size() << endl;
    switch( type ) {
    case VClamp:
        of << "#define NPOP " << npop << endl;
        nExtraVars += 0;
        break;
    case WaveGen:
        of << "#define GAPOP " << npop << endl;
        if ( !single ) {
            npop *= (_adjustableParams.size() + 1);
        }
        of << "#define NPOP " << npop << endl;
        nExtraVars += 2;
        break;
    case WaveGenNoveltySearch:
        of << "#define GAPOP " << npop << endl;
        if ( !single ) {
            npop *= (_adjustableParams.size() + 1);
        }
        of << "#define NPOP " << npop << endl;
        nExtraVars += 12;
        break;
    }

    of << endl;
    of << "#include \"modelSpec.h\"" << endl;
    of << "#include \"modelSpec.cc\"" << endl;

    of << endl;
    of << "double variableIni[" << to_string(_vars.size() + _adjustableParams.size() + nExtraVars) << "] = {" << endl;
    of << "  -60.0," << endl;
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
    of << "#ifdef DEFINITIONS_H" << endl;
    of << "#include \"realtimeenvironment.h\"" << endl;
    of << "scalar *mvar[NVAR];" << endl;
    of << "scalar *d_mvar[NVAR];" << endl;
    of << "scalar *mparam[NPARAM];" << endl;
    of << "scalar *d_mparam[NPARAM];" << endl;
    of << "void rtdo_init_bridge() {" << endl;
    of << "  mvar[0] = VHH;" << endl;
    of << "  d_mvar[0] = d_VHH;" << endl;
    int i = 1;
    for ( vector<param>::iterator it = _vars.begin(); it != _vars.end(); ++it, ++i ) {
        of << "mvar[" << i << "] = " << it->name << POPNAME << ";" << endl;
        of << "d_mvar[" << i << "] = d_" << it->name << POPNAME << ";" << endl;
    }
    i = 0;
    for ( vector<param>::iterator it = _adjustableParams.begin(); it != _adjustableParams.end(); ++it, ++i ) {
        of << "mparam[" << i << "] = " << it->name << POPNAME << ";" << endl;
        of << "d_mparam[" << i << "] = d_" << it->name << POPNAME << ";" << endl;
    }
    of << "RealtimeEnvironment::env().setClampGainParameter(&clampGainHH);" << endl;
    of << "}" << endl;
    of << "#endif" << endl;

    of << endl;
    of << "void modelDefinition(NNmodel &model) {" << endl;
    of << "neuronModel n;" << endl;
    of << "initGeNN();" << endl;
    of << "n.varNames.clear();" << endl;
    of << "n.varTypes.clear();" << endl;

    of << endl;
    of << "n.varNames.push_back(\"V\");" << endl;
    of << "n.varTypes.push_back(\"scalar\");" << endl;
    for ( vector<param>::iterator it = _vars.begin(); it != _vars.end(); ++it ) {
        of << "n.varNames.push_back(\"" << it->name << "\");" << endl;
        of << "n.varTypes.push_back(\"" << it->type << "\");" << endl;
    }
    of << endl;
    for ( vector<param>::iterator it = _adjustableParams.begin(); it != _adjustableParams.end(); ++it ) {
        of << "n.varNames.push_back(\"" << it->name << "\");" << endl;
        of << "n.varTypes.push_back(\"" << it->type << "\");" << endl;
    }
    of << endl;
    for ( vector<param>::iterator it = _params.begin(); it != _params.end(); ++it ) {
        of << "n.pNames.push_back(\"" << it->name << "\");" << endl;
    }
    of << "n.pNames.push_back(\"simCycles\");" << endl;
    of << endl;

    of << "n.varNames.push_back(\"err\");" << endl;
    of << "n.varTypes.push_back(\"scalar\");" << endl;
    of << "n.extraGlobalNeuronKernelParameters.push_back(\"ot\");" << endl;
    of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"scalar\");" << endl;
    of << "n.extraGlobalNeuronKernelParameters.push_back(\"clampGain\");" << endl;
    of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"scalar\");" << endl;
    of << endl;

    switch ( type ) {
    case VClamp:
        of << "n.simCode = R\"EOF(" << endl
           << CUTMARK << endl
           << "unsigned int mt;" << endl
           << "scalar mdt= DT/$(simCycles);" << endl
           << "for (mt=0; mt < $(simCycles); mt++) {" << endl
           << "  Isyn= $(clampGain)*($(stepVG)-$(V));" << endl
           << code << endl
           << "}" << endl
           << CUTMARK << endl
           << "if ((t > $(ot)) && (t < $(ote))) {" << endl
           << "  $(err) += abs(Isyn-$(IsynG));" << endl
           << "}" << endl
           << ")EOF\";" << endl;
        break;
    case WaveGen:
        of << "n.simCode = R\"EOF(" << endl
           << CUTMARK << endl
           << "unsigned int mt;" << endl
           << "scalar mdt= DT/$(simCycles);" << endl
           << "for (mt=0; mt < $(simCycles); mt++) {" << endl
           << "  Isyn= $(clampGain)*($(stepVG)-$(V));" << endl
           << code << endl
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
        break;
    case WaveGenNoveltySearch:
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
           << "  Isyn= $(clampGain)*($(stepVG)-$(V));" << endl
           << code << endl
           << "#ifndef _" << modelname << "_neuronFnct_cc" << endl // Don't compile this part in calcNeuronsCPU
           << "#ifdef _" << modelname << "_neuronKrnl_cc" << endl // Also, don't compile in simulateSingleNeuron
           << "  __shared__ double IsynShare[" + to_string(_adjustableParams.size() + 1) + "];" << endl
           << "  if ( gatherError ) {" << endl
           << "    IsynShare[threadIdx.x] = Isyn;" << endl
           << "    __syncthreads();" << endl
           << "    IsynErr += abs(Isyn-IsynShare[0]) * mdt;" << endl
           << "  }" << endl
           << "#endif" << endl
           << "#endif" << endl
           << "}" << endl
           << CUTMARK << endl

           << "#ifndef _" << modelname << "_neuronFnct_cc" << endl

           // Report Isyn for validation
           << "if ( $(stage) == " << stObservationWindow << " ) {" << endl
           << "  $(err) = Isyn;" << endl
           << "}" << endl

           // DetuneAdjust stage: Add up this parameter's deviation from tuned model
           << "if ( $(stage) == " << stDetuneAdjust << " ) {" << endl
           << "  $(err) += IsynErr * DT;" << endl

           // Calculate average deviation from tuned model across all parameters
           << "} else if ( gatherError ) {" << endl
           << "  __shared__ scalar IsynErrShare[" << to_string(_adjustableParams.size() + 1) << "];" << endl
           << "  IsynErrShare[threadIdx.x] = IsynErr;" << endl
           << "  __syncthreads();" << endl
           << "  scalar avgIsynErr = 0;" << endl
           << "  scalar maxIsynErr = 0;" << endl
           << "  int runnerUp = 0;"
           << "  for ( int i = 1; i < " << to_string(_adjustableParams.size() + 1) << "; ++i ) {" << endl
           << "    avgIsynErr += IsynErrShare[i];" << endl
           << "    maxIsynErr = maxIsynErr > IsynErrShare[i] ? maxIsynErr : IsynErrShare[i];" << endl
           << "    if ( i != threadIdx.x )  runnerUp = i;" << endl
           << "  }" << endl
           << "  avgIsynErr /= " << to_string(_adjustableParams.size()) << ";" << endl

           // Calculate $(best), normalised by the second-greatest error signal
           << "  if ( $(calcBest) && threadIdx.x > 0 ) {" << endl
           << "    if ( IsynErr == maxIsynErr ) {" << endl
           << "      if ( $(stage) == " << stObservationWindow << " && !$(calcExceed) && $(nBestCurrent) == 0 )" << endl
           << "        $(tStartCurrent) = t;" << endl
           << "      $(nBestCurrent)++;" << endl
           << "      $(bestCurrent) += (IsynErr - IsynErrShare[runnerUp]) / IsynErr;" << endl
           << "      if ( $(nBestCurrent) > $(nBest) ) {" << endl
           << "        $(nBest) = $(nBestCurrent);" << endl
           << "        $(best) = $(bestCurrent);" << endl
           << "        if ( $(stage) == " << stObservationWindow << " && !$(calcExceed) ) {" << endl
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
           << "  if ( $(calcExceed) && threadIdx.x > 0 ) {" << endl
           << "    if ( IsynErr > avgIsynErr && avgIsynErr > 0 ) {" << endl
           << "      if ( $(stage) == " << stObservationWindow << " && $(nExceedCurrent) == 0 )" << endl
           << "        $(tStartCurrent) = t;" << endl
           << "      $(nExceedCurrent)++;" << endl
           << "      $(exceedCurrent) += (IsynErr / avgIsynErr) - 1.0;" << endl
           << "      if ( $(exceedCurrent) > $(exceed) ) {" << endl
           << "        $(nExceed) = $(nExceedCurrent);" << endl
           << "        $(exceed) = $(exceedCurrent);" << endl
           << "        if ( $(stage) == " << stObservationWindow << " ) {" << endl
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
    }
    of << endl;

    switch ( type ) {
    case VClamp:
        of << "n.extraGlobalNeuronKernelParameters.push_back(\"ote\");" << endl;
        of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"scalar\");" << endl;
        of << "n.extraGlobalNeuronKernelParameters.push_back(\"stepVG\");" << endl;
        of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"scalar\");" << endl;
        of << "n.extraGlobalNeuronKernelParameters.push_back(\"IsynG\");" << endl;
        of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"scalar\");" << endl;
        break;
    case WaveGen:
        of << "n.varNames.push_back(\"stepVG\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;
        of << "n.varNames.push_back(\"ote\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;
        of << "optimiseBlockSize = 0;" << endl;
        of << "neuronBlkSz = " << to_string(_adjustableParams.size() + 1) << ";" << endl;
        of << "synapseBlkSz = 1;" << endl;
        of << "learnBlkSz = 1;" << endl;
        break;
    case WaveGenNoveltySearch:
        of << "n.varNames.push_back(\"stepVG\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;

        // Exceed: Windows where the error signal for a parameter exceeds the mean error across all parameters
        // The selected value reflects the greatest aggregate error above the mean (the area between the error and the mean error curves)
        of << "n.varNames.push_back(\"exceed\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;
        of << "n.varNames.push_back(\"exceedCurrent\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;
        of << "n.varNames.push_back(\"nExceed\");" << endl;
        of << "n.varTypes.push_back(\"int\");" << endl;
        of << "n.varNames.push_back(\"nExceedCurrent\");" << endl;
        of << "n.varTypes.push_back(\"int\");" << endl;

        // Best: Windows where the error signal for a parameter exceeds all other parameters' error signal
        // The selected value reflects the longest 'winning streak', rather than the greatest aggregate margin.
        of << "n.varNames.push_back(\"best\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;
        of << "n.varNames.push_back(\"bestCurrent\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;
        of << "n.varNames.push_back(\"nBest\");" << endl;
        of << "n.varTypes.push_back(\"int\");" << endl;
        of << "n.varNames.push_back(\"nBestCurrent\");" << endl;
        of << "n.varTypes.push_back(\"int\");" << endl;

        // Waveform start/end (for stObservationWindow)
        of << "n.varNames.push_back(\"tStart\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;
        of << "n.varNames.push_back(\"tStartCurrent\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;
        of << "n.varNames.push_back(\"tEnd\");" << endl;
        of << "n.varTypes.push_back(\"scalar\");" << endl;

        // Switches (for stWaveformOptimise)
        of << "n.extraGlobalNeuronKernelParameters.push_back(\"calcBest\");" << endl;
        of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"bool\");" << endl;
        of << "n.extraGlobalNeuronKernelParameters.push_back(\"calcExceed\");" << endl;
        of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"bool\");" << endl;

        of << "n.extraGlobalNeuronKernelParameters.push_back(\"ote\");" << endl;
        of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"scalar\");" << endl;
        of << "n.extraGlobalNeuronKernelParameters.push_back(\"stage\");" << endl;
        of << "n.extraGlobalNeuronKernelParameterTypes.push_back(\"int\");" << endl;
        of << "optimiseBlockSize = 0;" << endl;
        of << "neuronBlkSz = " << to_string(_adjustableParams.size() + 1) << ";" << endl;
        of << "synapseBlkSz = 1;" << endl;
        of << "learnBlkSz = 1;" << endl;
        break;
    }

    of << endl;
    of << "n.thresholdConditionCode = (\"false\");" << endl;
    of << "int modelNum = nModels.size();" << endl;
    of << "nModels.push_back(n);" << endl;
    of << "model.setName(\"" << modelname << "\");" << endl;
    of << "model.setPrecision(" << (genn_double ? "GENN_DOUBLE" : "GENN_FLOAT") << ");" << endl;
    of << "model.addNeuronPopulation(\"" << POPNAME << "\", " << npop << ", modelNum, fixedParamIni, variableIni);" << endl;
    of << "model.finalize();" << endl;
    of << "}" << endl;

    return modelname;
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
       << scalar << " lstepVG)" << endl;
    of << "{" << endl;
    of << "\t" << scalar << " Isyn = 0;" << endl;

    int i = 1;
    of << "\t" << scalar << " &lV = variables[0];" << endl;
    for ( param& v : _vars ) {
        of << "\t" << scalar << " &l" << v.name << " = variables[" << i << "];" << endl;
        ++i;
    }
    i = 0;
    for ( param &p : _adjustableParams ) {
        of << "\t" << scalar << " l" << p.name << " = adjustableParams[" << i << "];" << endl;
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
