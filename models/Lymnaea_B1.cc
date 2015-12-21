/*--------------------------------------------------------------------------
   Author: Thomas Nowotny

   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
          Falmer, Brighton BN1 9QJ, UK

   email to:  T.Nowotny@sussex.ac.uk

   initial version: 2010-02-07

--------------------------------------------------------------------------*/
#include "SimulationParameters.h"
#include "modelSpec.h"
#include "modelSpec.cc"

#define Nvar 4 /*!< Number of true variables (local use only) */
#define Npara 19 /*!< Number of adjustable parameters (local use only) */

//--------------------------------------------------------------------------
/*! \brief This function defines the HH model with variable parameters.
 */
//--------------------------------------------------------------------------

void modelDefinition(NNmodel &model)
{
  /*! \brief This array sets the initial values of the model.
   *  Its size must be at least (number of true variables) + (number of model parameters) + 1.
   *  Only the first (Nvar) values are used, as the adjustable parameters are initialised
   *  from the parameter file.
   */
  double myHH_ini[Npara+Nvar+1]= {
    -60.0,         // 0 - membrane potential E
    0.0529324,     // 1 - prob. for Na channel activation m
    0.3176767,     // 2 - prob. for not Na channel blocking h
    0.5961207,     // 3 - prob. for K channel activation n
  };

  double *myHH_p= NULL; //!< This array sets the initial values of fixed parameters, if present.

  neuronModel n;
  initGeNN();
  // HH neurons with adjustable parameters (introduced as variables)
  n.varNames.clear();
  n.varTypes.clear();

  // Add true variables first...
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("scalar"));
  n.varNames.push_back(tS("m"));
  n.varTypes.push_back(tS("scalar"));
  n.varNames.push_back(tS("h"));
  n.varTypes.push_back(tS("scalar"));
  n.varNames.push_back(tS("n"));
  n.varTypes.push_back(tS("scalar"));

  // Add adjustable parameters second...
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
    }\n");

  n.thresholdConditionCode = tS("$(V) > 100");

  // Allow RTDO to add its extra parameters and sim code and generate bridge code last.
  std::string popname = tS("HH");
  rtdo_add_extra_parameters(n);
  rtdo_generate_bridge(n, popname, Nvar, Npara, myHH_ini);

  int HHV= nModels.size();
  nModels.push_back(n);

  model.setName("Lymnaea_B1");
  model.setPrecision(GENN_FLOAT);
  model.setGPUDevice(fixGPU);
  model.addNeuronPopulation(popname, NPOP, HHV, myHH_p, myHH_ini);
  model.finalize();
}
