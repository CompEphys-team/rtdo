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
#ifdef RTDO
#include "model.h"
#else
#ifndef _WAVE
#include "HHVClamp.cc"
#else
#include "Wave.cc"
#endif
#include "HHVClamp_CODE/runner.cc"
#endif // RTDO
#include "numlib/randomGen.h"
#include "numlib/gauss.h"
randomGen R;
randomGauss RG;

#define BUFSZ 1024
#define STEPNO 3
#define BASEV -60.0
#define MAVGBUFSZ 10

#include "helper.h"

//#define DEBUG_PROCREATE
#include "GA.cc"

#define RAND(Y,X) Y = Y * 1103515245 +12345;X= (unsigned int)(Y >> 16) & 32767

// and some global variables
double t= 0.0f;
unsigned int iT= 0;
CStopWatch timer;
