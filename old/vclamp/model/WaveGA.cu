/*--------------------------------------------------------------------------
Author: Daniel Saska

Institute: Informatics
University of Sussex
Brighton BN1 9QJ, UK

email to:  ds376@sussex.ac.uk

initial version: 2014-09-09

--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file VClampGA.cu

\brief Main entry point for the GeNN project demonstrating realtime fitting of a neuron with a GA running mostly on the GPU.
*/
//--------------------------------------------------------------------------

#ifdef _WAVE

// minimum duration of voltage step
#define MINSTEP 2.0  
// minimal voltage
#define MINV -100.0
// maximal voltage
#define MAXV 50.0
// minimal observation time window
#define MINT 5.0
// maximal observation time window
#define MAXT 100.0
// observation time (fixed)
#define OT 100.0
// total simulation time
#define SIM_TIME 200.0
// initial dispersion of V steps
#define VSTEPINI 40.0
// initial "baseline" for V steps
#define VSTEP0 -60.0
// initial maximum width of steps (ms)
#define STEPWDINI 100.0
// minimal position of first step
#define MINSTEPT 10.0
// target parameter to build waveforms for
int TRGP = 0;
// number of GA NGENs
int NGEN = 500;
// factor of reducing mutateA
#define MUTATEFAC 0.995
// number of steps in the voltage signals
#define NVSTEPS 3

#define Npara 19

// the size of random mutations
double mutateA = 10.0;


#include "VClampGA.h"
#include "waveHelper.h"
#include <cuda.h>

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the project
*/
//--------------------------------------------------------------------------


int main( int argc, char *argv[] )
{
	std::string basename = "wave";
	int focusParam = 1;
	bool which = 1;
	if (argc == 11)
	{
		std::string basename = argv[5];
		focusParam = atoi( argv[2] );
		bool which = atoi( argv[10] );
	}

	std::cout << basename << std::endl;
	char buf[BUFSZ];
	int experimentNo = 0;
	string OutDir = basename + "_output";
	double gaBalance = 1.0 / (Npara - 1.0);

	//-----------------------------------------------------------------
	// Initialize poulation
	vector<inputSpec> stims;
	wave_pop_init( stims, GAPOP );
	size_t * sn = new size_t[NPOP];

	//-----------------------------------------------------------------
	// build the neuronal circuitery

	NNmodel model;
	modelDefinition( model );
	allocateMem();
	initialize();
	

	unsigned int VSize = NPOP*theSize( model.ftype );

	for (size_t generation = 0; generation < NGEN; ++generation)
	{
		memset( sn, 0x00000000, NPOP * sizeof( size_t ) );
		cudaMemset( d_errHH, 0x0, VSize );
		var_init_auto_detune();
		otHH = OT;
		cudaMemset( d_errHH, 0x0, VSize );

		for (size_t i = 0; i < stims.size(); ++i)
		{
			for (size_t j = 0; j < Npara + 1; j++)
			{
				float tmp = (stims[i].ot + OT);
				CHECK_CUDA_ERRORS( cudaMemcpy( &d_oteHH[i * (Npara + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
			}
		}
		for (double t = 0.0; t < SIM_TIME; t += DT)
		{
			if (which == GPU) {
				stepTimeGPU( t );
			}
			else {
				return;//not supported
			}

			for (size_t i = 0; i < stims.size(); ++i)
			{
				if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0))) 
				{
					for (size_t j = 0; j < Npara + 1; ++j)
					{
						float tmp = stims[i].V[sn[i]];
						CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (Npara + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
					}
					++sn[i];
				}
			}

		}
		if (which == 1) {
			CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );
			for (size_t i = 0; i < stims.size(); ++i)
			{
				stims[i].fit = 0.0;
				for (size_t j = 0; j < Npara + 1; j++)
				{
					if (j == focusParam)
					{
						stims[i].fit += errHH[i * (Npara + 1) + j];
					} 
					else
					{
						stims[i].fit -= errHH[i * (Npara + 1) + j] * gaBalance;
					}
				}
			}
			std::ofstream file( basename + "." + std::to_string( focusParam - 1 ) +"." + std::to_string( generation ) + ".pop" );
			for (size_t i = 0; i < stims.size(); ++i)
			{
				file << stims[i] << std::endl;
			}
			file.close();
			procreatePop( stims );
		}
	}
	delete[] sn;

	fprintf( stderr, "DONE" );
	return 0;
}
#endif