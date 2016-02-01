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
// number of GA NGENs
int NGEN = 500;
// factor of reducing mutateA
#define MUTATEFAC 0.995
// number of steps in the voltage signals
#define NVSTEPS 3

#define TOTALT 200

// the size of random mutations
double mutateA = 10.0;


#include "WaveGA.h"
#include "waveHelper.h"
#include <cuda.h>

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the project
*/
//--------------------------------------------------------------------------


extern "C" inputSpec wavegen(int focusParam, int nGenerations, bool *stopFlag)
{
    double gaBalance = 1.0 / (NPARAM - 1.0);
    NGEN = nGenerations;

	//-----------------------------------------------------------------
    // Initialize population
	vector<inputSpec> stims;
	wave_pop_init( stims, GAPOP );
	size_t * sn = new size_t[NPOP];

	//-----------------------------------------------------------------
	// build the neuronal circuitery

	NNmodel model;
	modelDefinition( model );
	allocateMem();
	initialize();
    rtdo_init_bridge();
	
    //------------------------------------------------------
    // Get steady-state variable values at holding potential
    stepVGHH[0] = VSTEP0;
    for ( double t = 0.0; t < 10000.0; t += DT ) {
        calcSingleNeuronCPU(t);
    }
    scalar holdingVar[NVAR];
    for ( int i = 0; i < NVAR; i++ )
        holdingVar[i] = mvar[i][0];


	unsigned int VSize = NPOP*theSize( model.ftype );

    for (size_t generation = 0; generation < NGEN && !*stopFlag; ++generation)
	{
        for ( int i = 0; i < NVAR; i++ ) {
            for ( int j = 0; j < NPOP; j++ ) {
                mvar[i][j] = holdingVar[i];
            }
        }
        for ( int j = 0; j < NPOP; j++ ) {
            errHH[j] = 0.0;
        }

        var_init_auto_detune();
        memset( sn, 0x00000000, NPOP * sizeof( size_t ) );
        otHH = OT;

		for (size_t i = 0; i < stims.size(); ++i)
		{
            for (size_t j = 0; j < NPARAM + 1; j++)
			{
                float tmp = (stims[i].ot + stims[i].dur);
                CHECK_CUDA_ERRORS( cudaMemcpy( &d_oteHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
			}
		}
		for (double t = 0.0; t < SIM_TIME; t += DT)
        {
            stepTimeGPU( t );

			for (size_t i = 0; i < stims.size(); ++i)
			{
				if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0))) 
				{
                    for (size_t j = 0; j < NPARAM + 1; ++j)
					{
						float tmp = stims[i].V[sn[i]];
                        CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
					}
					++sn[i];
				}
			}

		}

        CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );
        for (size_t i = 0; i < stims.size(); ++i)
        {
            stims[i].fit = 0.0;
            for (size_t j = 1; j < NPARAM + 1; j++)
            {
                if (j == focusParam+1)
                {
                    stims[i].fit += errHH[i * (NPARAM + 1) + j];
                }
                else
                {
                    stims[i].fit -= errHH[i * (NPARAM + 1) + j] * gaBalance;
                }
            }
        }
        procreatePop( stims );
        cout << "Generation " << generation << "'s best stimulus:" << endl;
        cout << stims[0] << endl;
	}
	delete[] sn;
    return stims[0];
}
