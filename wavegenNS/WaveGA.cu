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
// factor of reducing mutateA
#define MUTATEFAC 0.995
// number of steps in the voltage signals
#define NVSTEPS 3

#define TOTALT 200

// the size of random mutations
double mutateA = 10.0;

// Novelty search param
double maxSigmaToRange = 0.1;
double pertFac = 0.1;
double noveltyThreshold = 10.0;
double optimiseInitProportion = 0.2;


#include "WaveGA.h"
#include "waveHelper.h"
#include <cuda.h>
#include <array>


//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the project
*/
//--------------------------------------------------------------------------


extern "C" void wavegenNS(int nGenerationsNS, int nGenerationsOptimise, ofstream &outfile, bool *stopFlag)
{
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
    scalar holdingVar[NVAR], singleParamIni[NPARAM];
    for ( int i = 0; i < NVAR; i++ )
        holdingVar[i] = mvar[i][0];
    for ( int i = 0; i < NPARAM; i++ )
        singleParamIni[i] = mparam[i][0];
    for ( double t = 0.0; t < 10000.0; t += DT ) {
        simulateSingleNeuron(holdingVar, singleParamIni, VSTEP0);
    }

	unsigned int VSize = NPOP*theSize( model.ftype );

    // Stage: Adjust sigmas to detune with similar results:
    cout << "Adjusting parameter sigmas..." << endl;
    for ( int r = 0; r < 2; r++ ) {
        double sigmax[NPARAM];
        for ( int j = 0; j < NPARAM; j++ ) {
            if ( aParamPMult[j] )
                sigmax[j] = maxSigmaToRange; // Range-dependent values are problematic because of zero range end points
            else
                sigmax[j] = maxSigmaToRange * (aParamRange[2*j + 1] - aParamRange[2*j]);
        }

        otHH = OT;
        oteHH = TOTALT;
        stageHH = stDetuneAdjust;
        reset(holdingVar, pertFac);
        memset( sn, 0x00000000, NPOP * sizeof( size_t ) );
        for (double t = 0.0; t < SIM_TIME; t += DT) {
            stepTimeGPU(t);
            for (size_t i = 0; i < GAPOP; ++i) {
                if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0)))
                {
                    for (size_t j = 0; j < NPARAM + 1; ++j) {
                        float tmp = stims[i].V[sn[i]];
                        CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
                    }
                    ++sn[i];
                }
            }
        }

        CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );

        double pErr[NPARAM] = {}, errTotal = 0.0;
        for ( size_t i = 0; i < GAPOP; ++i ) {
            for ( size_t j = 1; j < NPARAM + 1; ++j ) {
                pErr[j-1] += errHH[i * (NPARAM + 1) + j];
                errTotal += errHH[i * (NPARAM + 1) + j];
            }
        }
        errTotal /= NPARAM;
        double globalAdjust = 1.0;
        for ( size_t j = 0; j < NPARAM; ++j ) {
            double adjustment = errTotal / pErr[j];
            aParamSigma[j] *= adjustment;
            double reduce = aParamSigma[j]/sigmax[j]; // > 1 if aParamSigma[j] exceeds its sigmax
            if ( reduce > globalAdjust ) {
                globalAdjust = reduce;
            }
        }
        for ( int j = 0; j < NPARAM; j++ ) {
            aParamSigma[j] /= globalAdjust;
        }
    }
    for ( int i = 0; i < NPARAM; i++ ) {
        cout << "Parameter " << (i+1) << " adjusted sigma: " << aParamSigma[i] << endl;
    }
    cout << endl;

    // Stage: Novelty search
    array<vector<noveltyBundle>, NPARAM> noveltyDB;
    for ( int i = 0; i < NPARAM; i++ ) {
        noveltyDB[i].push_back({});
    }
    {
        otHH = OT;
        oteHH = TOTALT;
        stageHH = stNoveltySearch;

        for ( size_t generation = 0; generation < nGenerationsNS && !*stopFlag; ++generation ) {
            cout << "Novelty search, generation " << generation << endl;
            reset(holdingVar, pertFac);
            memset( sn, 0x00000000, NPOP * sizeof( size_t ) );
            for (double t = 0.0; t < SIM_TIME; t += DT) {
                stepTimeGPU(t);
                for (size_t i = 0; i < GAPOP; ++i) {
                    if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0)))
                    {
                        for (size_t j = 0; j < NPARAM + 1; ++j) {
                            float tmp = stims[i].V[sn[i]];
                            CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
                        }
                        ++sn[i];
                    }
                }
            }

            CHECK_CUDA_ERRORS( cudaMemcpy( exceedHH, d_exceedHH, VSize, cudaMemcpyDeviceToHost ) );
            CHECK_CUDA_ERRORS( cudaMemcpy( nExceedHH, d_nExceedHH, NPOP*sizeof(int), cudaMemcpyDeviceToHost ) );

            noveltyBundle bundle;
            double avgNovelty = 0;
            int numNew = 0;
            for ( size_t i = 0; i < GAPOP; ++i ) {
                stims[i].fit = 0.0;
                for ( size_t j = 1; j < NPARAM + 1; j++ ) {
                    bundle.novelty[0] = exceedHH[i * (NPARAM + 1) + j];
                    bundle.novelty[1] = nExceedHH[i * (NPARAM + 1) + j];
                    double least = 1e9;
                    for ( noveltyBundle &p : noveltyDB[j-1] ) {
                        double dist = noveltyDistance(p, bundle);
                        if ( least > dist ) {
                            least = dist;
                            if ( least < noveltyThreshold )
                                break;
                        }
                    }
                    stims[i].fit += least / noveltyDB[j-1].size(); // Bias fitness, but not novelty, by count
                    avgNovelty += least;
                    if ( least > noveltyThreshold ) {
                        bundle.wave = stims[i];
                        noveltyDB[j-1].push_back(bundle);
                        ++numNew;
                    }
                }
            }
            cout << "Average novelty value: " << (avgNovelty / GAPOP) << ", " << numNew << " new waves" << endl;

            procreatePop(stims);
        }
    }
    for ( int i = 0; i < NPARAM; i++ ) {
        cout << (noveltyDB[i].size()-1) << " touchstone waves for parameter " << (i+1) << endl;
        if ( noveltyDB[i].size() == 1 ) {
            cout << "Waveforms for this parameter will be generated from scratch. You may need to "
                    "decrease the novelty threshold or fit this parameter using a different method." << endl;
            noveltyDB[i].clear();
        }
    }


    // Stage: Waveform optimisation
    for ( int k = 0; k < NPARAM && !*stopFlag; k++ ) {
        stageHH = stWaveformOptimise;

        vector<inputSpec> initial;
        initial.reserve(noveltyDB[k].size() * optimiseInitProportion);
        sort(noveltyDB[k].begin(), noveltyDB[k].end(), fittestNovelty);
        for ( int i = 0; i < noveltyDB[k].size() * optimiseInitProportion; i++ ) {
            initial.push_back(noveltyDB[k].at(i).wave);
        }

        stims.clear();
        wave_pop_init_from( stims, GAPOP, initial );

        for (size_t generation = 0; generation < nGenerationsOptimise && !*stopFlag; ++generation) {
            cout << "Optimising parameter " << (k+1) << ", generation " << generation << endl;
            reset(holdingVar, pertFac);
            memset( sn, 0x00000000, NPOP * sizeof( size_t ) );
            for (double t = 0.0; t < SIM_TIME; t += DT) {
                stepTimeGPU(t);
                for (size_t i = 0; i < GAPOP; ++i) {
                    if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0)))
                    {
                        for (size_t j = 0; j < NPARAM + 1; ++j) {
                            float tmp = stims[i].V[sn[i]];
                            CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
                        }
                        ++sn[i];
                    }
                }
            }

            CHECK_CUDA_ERRORS( cudaMemcpy( exceedHH, d_exceedHH, VSize, cudaMemcpyDeviceToHost ) );
            for (size_t i = 0; i < GAPOP; ++i) {
                stims[i].fit = exceedHH[i * (NPARAM + 1) + k + 1];
            }
            procreateInitialisedPop( stims, initial );
            cout << stims[0] << endl;
        }
        if ( *stopFlag )
            break;

        // Substage: Find observation window
        stageHH = stObservationWindow;
        reset(holdingVar, pertFac);
        memset( sn, 0x00000000, NPOP * sizeof( size_t ) );
        for (double t = 0.0; t < SIM_TIME; t += DT) {
            stepTimeGPU(t);
            for (size_t i = 0; i < GAPOP; ++i) {
                if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0)))
                {
                    for (size_t j = 0; j < NPARAM + 1; ++j) {
                        float tmp = stims[i].V[sn[i]];
                        CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
                    }
                    ++sn[i];
                }
            }
        }
        CHECK_CUDA_ERRORS( cudaMemcpy( exceedHH, d_exceedHH, VSize, cudaMemcpyDeviceToHost ) );
        CHECK_CUDA_ERRORS( cudaMemcpy( tStartHH, d_tStartHH, VSize, cudaMemcpyDeviceToHost ) );
        CHECK_CUDA_ERRORS( cudaMemcpy( tEndHH, d_tEndHH, VSize, cudaMemcpyDeviceToHost ) );
        for (size_t i = 0; i < GAPOP; ++i) {
            stims[i].fit = exceedHH[i * (NPARAM + 1) + k + 1];
            stims[i].ot = tStartHH[i * (NPARAM + 1) + k + 1];
            stims[i].dur = tEndHH[i * (NPARAM + 1) + k + 1] - stims[i].ot;
        }
        sort(stims.begin(), stims.end(), larger);
        cout << "Top 5% with their observation windows:" << endl;
        for ( size_t i = 0; i < stims.size() / 20; i++ ) {
            cout << (i+1) << ": " << stims[i] << endl;
        }

        for ( int i = 0; i < NPARAM; i++ )
            outfile << (i==k) << " ";
        outfile << stims[0] << endl;

    }


    delete[] sn;
}
