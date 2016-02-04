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

    // Assume std >= c++11 for raw string literal:
    of << "n.simCode = R\"EOF(" << endl
       << CUTMARK << endl
       << "unsigned int mt;" << endl
       << "scalar mdt= DT/$(simCycles);" << endl
       << "for (mt=0; mt < $(simCycles); mt++) {" << endl
       << "  Isyn= $(clampGain)*($(stepVG)-$(V));" << endl
       << code
       << endl;
    switch ( type ) {
    case VClamp:
        of << "}" << endl
           << CUTMARK << endl
           << "if ((t > $(ot)) && (t < $(ote))) {" << endl
           << "  $(err) += abs(Isyn-$(IsynG));" << endl
           << "}" << endl
           << ")EOF\";" << endl;
        break;
    case WaveGen:
        of << "#ifndef _" << modelname << "_neuronFnct_cc" << endl // Don't compile this part in calcNeuronsCPU
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
    }
    if ( single ) {
        modelname += "_single";
    }
    return modelname;
}
