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

        string instanceDir = "model."  + std::to_string(protocol);
        system( std::string("cp -R model " + instanceDir).c_str() );

		// write model parameters
		string fname = instanceDir + "/HHVClampParameters.h";
		ofstream os( fname.c_str() );
		os << "#define NPOP " << nPop;
#ifdef _WAVE
		os << "*20"; //Number of param + 1
#endif // _WAVE

		os << endl;
#ifdef _WAVE
		os << "#define GAPOP " << nPop << endl;
#endif
		os << "#define TOTALT " << totalT << endl;
		os << "#define fixGPU " << GPU << endl;
		os << "#define INoiseSTD " << INoiseSTD << endl;
		os.close();


		// build it
#ifdef _WIN32
		cmd += "cd /d model && nmake /nologo /f WINmakefile clean && nmake /nologo /f WINmakefile";
		if (dbgMode == 1) {
		cmd += " DEBUG=1";
		cout << cmd << endl;
		}
#else // UNIX
		cmd = "cd " + instanceDir + " && make clean && make";
		if (dbgMode == 1) {
			cmd += " debug";
		}
		else {
			cmd += " release";
		}
#endif
		cerr << cmd << endl;
		retval = system( cmd.c_str() );
	if (retval != 0){
		cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
		cerr << "Exiting..." << endl;
		exit( 1 );
	}



	//run it!
	cout << "running test..." << endl;
	cmd = std::string( argv[5] ) + " "
		+ std::string( argv[6] ) + " "
		+ std::string( argv[7] ) + " "
		+ "1 "
		+ std::string( argv[8] ) + " ";
#ifdef _WIN32
	if (dbgMode == 1) {
		cmd = "devenv /debugexe model\\VClampGA.exe " + cmd;
	}
	else {
		cmd = "model\\VClampGA.exe " + cmd;
	}
#else // UNIX
	if (dbgMode == 1) {
		cmd = "cuda-gdb -tui --args " + instanceDir + "/VClampGA " + cmd;
	}
	else {
		cmd = instanceDir + "/VClampGA " + cmd;
        }
#endif
	std::cerr << cmd << std::endl;
	retval = system( cmd.c_str() );
	system( std::string("rm -R " + instanceDir).c_str() );
	if (retval != 0){
		cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
		cerr << "Exiting..." << endl;
		exit( 1 );
	}

	return 0;
}
