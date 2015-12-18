#include <string>
#include <iostream>
#include "modelSpec.h"
#include "modelSpec.cc"
using namespace std;

void rtdo_generate_bridge(const neuronModel& n, const string& pop_name, int nVars, int nParams, double *var_ini) {
    assert(nVars + nParams <= n.varNames.size() + 1);
    static bool printed = false;
    if ( printed )
        return;
    cout << BRIDGE_START << endl;

    cout << "#define NVAR " << nVars << endl;
    cout << "#define NPARAM " << nParams << endl;

    size_t i = 0;
    if ( nVars > 0 ) {
        cout << "scalar *mvar[NVAR] = {" << endl;
        for ( i = 0; i < nVars; i++ ) {
            cout << '\t' << n.varNames[i] << pop_name << "," << endl;
        }
        cout << "};" << endl;
        cout << "double mvar_ini[NVAR] = {" << endl;
        for ( i = 0; i < nVars; i++ ) {
            cout << '\t' << var_ini[i] << "," << endl;
        }
        cout << "};" << endl;
    }
    if ( nParams > 0 )  {
        cout << "scalar *mparam[NPARAM] = {" << endl;
        for ( i = nVars; i < nVars + nParams; i++ ) {
            cout << '\t' << n.varNames[i] << pop_name << "," << endl;
        }
        cout << "};" << endl;
    }

    cout << "scalar *merr = " << n.varNames.back() << pop_name << ";" << endl;
    cout << "scalar& ot = ot" << pop_name << ";" << endl;
    cout << "scalar& ote = ote" << pop_name << ";" << endl;
    cout << "scalar& stepVG = stepVG" << pop_name << ";" << endl;
    cout << "scalar& IsynG = IsynG" << pop_name << ";" << endl;

    cout << "scalar mrange[NPARAM][2];" << endl;
    cout << "bool mpertmult[NPARAM];" << endl;

    cout << BRIDGE_END << endl;
    printed = true;
}

void rtdo_add_extra_parameters(neuronModel& n) {
  n.varNames.push_back(tS("err"));
  n.varTypes.push_back(tS("scalar"));
  n.extraGlobalNeuronKernelParameters.push_back(tS("ot"));
  n.extraGlobalNeuronKernelParameterTypes.push_back(tS("scalar"));
  n.extraGlobalNeuronKernelParameters.push_back(tS("ote"));
  n.extraGlobalNeuronKernelParameterTypes.push_back(tS("scalar"));
  n.extraGlobalNeuronKernelParameters.push_back( tS( "stepVG" ) );
  n.extraGlobalNeuronKernelParameterTypes.push_back( tS( "scalar" ) );
  n.extraGlobalNeuronKernelParameters.push_back(tS("IsynG"));
  n.extraGlobalNeuronKernelParameterTypes.push_back(tS("scalar"));

  n.simCode += tS("\n\
    if ((t > $(ot)) && (t < $(ote))) {\n\
      $(err)+= abs(Isyn-$(IsynG));\n\
    }");
}
