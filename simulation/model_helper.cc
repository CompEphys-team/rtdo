#include <string>
#include <iostream>
#include "modelSpec.h"
#include "modelSpec.cc"
using namespace std;

void rtdo_generate_bridge(const neuronModel& n, const string& pop_name, int nVars, int nParams, double *var_ini) {
#ifndef _V_CLAMP
    assert(nVars + nParams <= n.varNames.size() + 1);
    static bool printed = false;
    if ( printed )
        return;
    cout << BRIDGE_START << endl;

    cout << "#define NVAR " << nVars << endl;
    cout << "#define NPARAM " << nParams << endl;

    size_t i = 0;
    cout << endl << "double mvar_ini[NVAR] = {" << endl;
    for ( i = 0; i < nVars; i++ ) {
        cout << '\t' << var_ini[i] << "," << endl;
    }
    cout << "};" << endl;

    cout << "scalar *mvar[NVAR];" << endl;
    cout << "scalar *mparam[NPARAM];" << endl;
    cout << "scalar *errM;" << endl;
    cout << "scalar *d_errM;" << endl;
    cout << "scalar& ot = ot" << pop_name << ";" << endl;
    cout << "scalar& ote = ote" << pop_name << ";" << endl;
    cout << "scalar& stepVG = stepVG" << pop_name << ";" << endl;
    cout << "scalar& IsynG = IsynG" << pop_name << ";" << endl;

    cout << endl << "void rtdo_init_bridge() {" << endl;
    for ( i = 0; i < nVars; i++ ) {
        cout << "\tmvar[" << i << "] = " << n.varNames[i] << pop_name << ";" << endl;
    }
    for ( i = 0; i < nParams; i++ ) {
        cout << "\tmparam[" << i << "] = " << n.varNames[i+nVars] << pop_name << ";" << endl;
    }
    cout << "\terrM = err" << pop_name << ";" << endl;
    cout << "\td_errM = d_err" << pop_name << ";" << endl;
    cout << "}" << endl;

    cout << BRIDGE_END << endl;
    printed = true;
#endif
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
