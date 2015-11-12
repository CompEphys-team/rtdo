/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file HHVClamp.cc

\brief This file contains the model definition of HHVClamp model. It is used in both the GeNN code generation and the user side simulation code. The HHVClamp model implements a population of unconnected Hodgkin-Huxley neurons that evolve to mimick a model run on the CPU, using genetic algorithm techniques.
*/
//--------------------------------------------------------------------------

#define DT 0.25  //!< This defines the global time step at which the simulation will run
#define Npara 19
#include "modelSpec.h"
#include "modelSpec.cc"
#include "HHVClampParameters.h"

double myHH_ini[Npara+6]= {
  -60.0,         // 0 - membrane potential E
  0.0529324,     // 1 - prob. for Na channel activation m
  0.3176767,     // 2 - prob. for not Na channel blocking h
  0.5961207,     // 3 - prob. for K channel activation n
  120.0,         // 4 - gNa: Na conductance in 1/(mOhms * cm^2)
  55.0,          // 5 - ENa: Na equi potential in mV
  3.5,           // 6 - maoff: offset of alpha_m curve Vmid*slope
  0.1,           // 7 - maslope: slope of alpha_m curve
  60.0,          // 8 - mboff
  18.0,          // 9 - mbslope
  3.0,           // 10 - haoff
  20.0,          // 11 - haslope
  3.0,           // 12 - hboff
  0.1,           // 13 - hbslope
  36.0,          // 14 - gK: K conductance in 1/(mOhms * cm^2)
  -72.0,         // 15 - EK: K equi potential in mV
  0.5,           // 16 - naoff
  0.01,          // 17 - naslope
  60.0,          // 18 - nboff
  80.0,          // 19 - nbslope
  0.3,           // 20 - gl: leak conductance in 1/(mOhms * cm^2)
  -50.0,         // 21 - El: leak equi potential in mV
  1.0,           // 22 - Cmem: membr. capacity density in muF/cm^2
  0.0            // 23 - err
  ,-60.0
};

double *myHH_p= NULL;


//--------------------------------------------------------------------------
/*! \brief This function defines the HH model with variable parameters.
 */
//--------------------------------------------------------------------------

void modelDefinition(NNmodel &model) 
{
	optimiseBlockSize = 0;
	neuronBlkSz = Npara + 1;
	synapseBlkSz = 1;
	learnBlkSz = 1;
	neuronModel n;
	initGeNN();
	// HH neurons with adjustable parameters (introduced as variables)
	n.varNames.clear();
	n.varTypes.clear();
	n.varNames.push_back(tS("V"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("m"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("h"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("n"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("gNa"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("ENa"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("maoff"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("maslope"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("mboff"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("mbslope"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("haoff"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("haslope"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("hboff"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("hbslope"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("gK"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("EK"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("naoff"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("naslope"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("nboff"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("nbslope"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("gl"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("El"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("C"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("err"));
	n.varTypes.push_back( tS( "scalar" ) );
	n.varNames.push_back( tS( "stepVG" ) );
	n.varTypes.push_back( tS( "scalar" ) );
	n.varNames.push_back( tS( "ote" ) );
	n.varTypes.push_back( tS( "scalar" ) );

	n.extraGlobalNeuronKernelParameters.push_back(tS("ot"));
	n.extraGlobalNeuronKernelParameterTypes.push_back(tS("scalar"));  
	n.extraGlobalNeuronKernelParameters.push_back(tS("IsynG"));
	n.extraGlobalNeuronKernelParameterTypes.push_back(tS("scalar"));

	n.simCode= tS("   scalar Imem;\n\
	unsigned int mt;\n\
	scalar mdt= DT/50.0;\n\
	for (mt=0; mt < 50; mt++) {\n\
		Isyn= 200.0*($(stepVG)-$(V));\n\
		Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
				$(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
				$(gl)*($(V)-($(El)))-Isyn);\n\
		scalar _a= ($(maoff)+$(maslope)*$(V)) / (1.0-exp(-$(maoff)-$(maslope)*$(V)));\n\
		scalar _b= 4.0*exp(-($(V)+$(mboff))/$(mbslope));\n\
		$(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n\
		_a= 0.07*exp(-$(V)/$(haslope)-$(haoff));\n\
		_b= 1.0 / (exp(-$(hboff)-$(hbslope)*$(V))+1.0);\n\
		$(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n\
		_a= (-$(naoff)-$(naslope)*$(V)) / (exp(-10.0*$(naoff)-10.0*$(naslope)*$(V))-1.0);\n\
		_b= 0.125*exp(-($(V)+$(nboff))/$(nbslope));\n\
		$(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
		$(V)+= Imem/$(C)*mdt;\n\
#ifndef _HHVClamp_neuronFnct_cc\n\
		__shared__ double IsynShare[Npara+1];\n\
		if ((t > $(ot)) && (t < $(ote))) {\n\
			IsynShare[threadIdx.x] = Isyn;\n\
			__syncthreads();\n\
			$(err)+= abs(Isyn-IsynShare[0]) * mdt * " + std::to_string(DT) + ";\n\
			if(abs($(stepVG))>200)\n\
			{\n\
			printf(\"%f\\n\",$(stepVG));\n\
			}\n\
		}\n\
#endif\n\
	}\n");

	n.thresholdConditionCode = tS("$(V) > 100");
	int HHV= nModels.size();
	nModels.push_back(n);
	model.setName("HHVClamp");
	model.setPrecision(GENN_FLOAT);
	model.setGPUDevice(fixGPU);
	model.addNeuronPopulation("HH", NPOP, HHV, myHH_p, myHH_ini);
	model.finalize();
}
