/*--------------------------------------------------------------------------
Author: Thomas Nowotny

Institute: Center for Computational Neuroscience and Robotics
University of Sussex
Falmer, Brighton BN1 9QJ, UK

email to:  T.Nowotny@sussex.ac.uk

initial version: 2010-02-07

--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generate_run.cc

\brief This file is used to run the HHVclampGA model with a single command line.


*/
//--------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>

#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#else // UNIX
#include <sys/stat.h> //needed for mkdir
#endif

using namespace std;

//--------------------------------------------------------------------------
/*! \brief Main entry point for generate_run.
*/
//--------------------------------------------------------------------------
#define _V_CLAMP
int main( int argc, char *argv[] )
{
	std::cerr << "Starting" << std::endl;
	if (argc != 11)
	{
		cerr << "usage: generate_run <CPU=0, GPU=1> <protocol> <nPop> <totalT> <outdir> <stimulus file> <sigma file> <noise amplitude (STD)> <debug mode? (0/1)> <GPU choice>" << endl;
		exit( 1 );
	}
	int which = atoi( argv[1] );
	int protocol = atoi( argv[2] );
	int nPop = atoi( argv[3] );
	double totalT = atof( argv[4] );
	string outDir = "output";
	double INoiseSTD = atof( argv[8] );
	int dbgMode = atoi( argv[9] ); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
	int GPU = atoi( argv[10] );
	int retval;
	string cmd;

        
	string GeNNPath = getenv( "GENN_PATH" );
	cerr << GeNNPath << endl;

	string fname = "model/HHVClampParameters.h";
	ofstream os( fname.c_str() );
	os << "#define NPOP " << nPop;
#ifndef _WAVE
	os << "*20"; //Number of param + 1
#endif // _WAVE
	os << endl;
	os << "#define fixGPU " << GPU << endl;
	os.close();
	
	// build it
#ifdef _WIN32
#ifdef _V_CLAMP
	cmd = "cd /d model && buildmodel.bat HHVClamp " + std::to_string( dbgMode );
#elif defined(_I_CLAMP)
	cmd = "cd /d model && buildmodel.bat HHIClamp " + std::to_string( dbgMode );
#elif defined(_WAVE)
	cmd = "cd /d model && buildmodel.bat Wave " + std::to_string( dbgMode );
#endif

	cmd += " && nmake /nologo /f WINmakefile clean && nmake /nologo /f WINmakefile";
	/*if (dbgMode == 1) {
		cmd += " DEBUG=1";
		cout << cmd << endl;
	}*/
#else // UNIX
#ifdef _V_CLAMP
	cmd = "cd model && buildmodel.sh HHVClamp " + std::to_string( dbgMode );
#elif defined(_I_CLAMP)
	cmd = "cd model && buildmodel.sh Wave " + std::to_string( dbgMode );
#elif defined(_WAVE)
	cmd = "cd model && buildmodel.sh Wave " + std::to_string( dbgMode );
#endif // !_WAVE
// 	cmd += " && make clean && make";
// 	if (dbgMode == 1) {
// 		cmd += " debug";
// 	}
// 	else {
// 		cmd += " release";
// 	}
#endif
	cerr << cmd << endl;
	retval = system( cmd.c_str() );



	// create output directory
#ifdef _WIN32
	_mkdir( outDir.c_str() );
#else // UNIX
	if (mkdir( outDir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH ) == -1) {
		cerr << "Directory cannot be created. It may exist already." << endl;
	}
#endif
	return 0;
}
