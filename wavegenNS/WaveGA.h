/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file classol_sim.h

\brief Header file containing global variables and macros used in running the HHVClamp/VClampGA model.
*/
//--------------------------------------------------------------------------

#include <cassert>
#include <vector>
#include <fstream>

using namespace std;
#include "hr_time.cpp"
#include "utils.h" // for CHECK_CUDA_ERRORS
#include <cuda_runtime.h>
#include "numlib/randomGen.h"
#include "numlib/gauss.h"
#include "run.h"

randomGen R;
randomGauss RG;

#define BUFSZ 1024
#define STEPNO 3
#define BASEV -60.0
#define MAVGBUFSZ 10

double sigmaAdjust[NPARAM];
